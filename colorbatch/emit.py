"""Re-emit parsed G-code with segments in a new order.

Between segments the emitter inserts: tool-change block (when the tool changes, Z re-based on the
tallest printed point), a pseudo layer marker for the preview, machine-state restore, a
collision-safe lift/travel/lower, a retraction-state fix, and object-label wrapping. Travel moves
inside a segment are lifted when their path would pass too close to taller printed material.
"""
from __future__ import annotations

import re

import numpy as np

from .gcode import CELL, Block, MachineState, Parsed, Segment, cell_centres, decode, is_move, path_points, words
from .head import HeadModel

GRID = 2048  # cells per axis (covers -512..512 mm at 0.5 mm)
_Z_WORD = re.compile(r"(\bZ)\s*([-+]?(?:\d+\.?\d*|\.\d+))")
_FLUSH_E = re.compile(r"^(G1\s+E)([-+]?(?:\d+\.?\d*|\.\d+))(.*)$")
_M620_10_L = re.compile(r"^(M620\.10 .*\bL)([-+]?(?:\d+\.?\d*|\.\d+))(.*)$")

TRAVEL_F = 30000.0
Z_F = 1200.0
HOP = 0.4  # extra margin added on top of the required clearance when lifting


def flush_matrix(cfg: dict[str, str], n_tools: int) -> np.ndarray:
    """Flush volumes [from, to] (mm^3) x flush_multiplier. Multi-nozzle profiles store one n x n
    block per nozzle; the first (only, for a single nozzle) block is used."""
    vals = [float(v) for v in cfg.get("flush_volumes_matrix", "").split(",") if v.strip()]
    mult = float(cfg.get("flush_multiplier", "1").split(",")[0])
    colours = [c for c in re.split(r"[;,]", cfg.get("filament_colour", "")) if c.strip()]
    n = len(colours) if colours and vals and len(vals) % (len(colours) ** 2) == 0 else int(round(len(vals) ** 0.5))
    if n * n > len(vals):
        n = 0
    size = max(n, n_tools)
    m = np.full((size, size), 280.0)
    np.fill_diagonal(m, 0.0)
    if n:
        m[:n, :n] = np.array(vals[:n * n]).reshape(n, n)
    return m * mult


class HeightMap:
    def __init__(self):
        self.h = np.zeros((GRID, GRID), dtype=np.float32)
        self.row_max = np.zeros(GRID, dtype=np.float32)  # tallest material per Y row (for the X rod)
        self.max = 0.0

    def add(self, cells: np.ndarray, z: float) -> None:
        ix, iy = decode(cells)
        np.maximum.at(self.h, (ix, iy), np.float32(z))
        np.maximum.at(self.row_max, iy, np.float32(z))
        self.max = max(self.max, z)

    def required_z(self, pts: np.ndarray, z: float, head: HeadModel) -> float:
        """Lowest nozzle z at which the head can pass over polyline `pts` (N,2 mm)."""
        if self.max <= z + 1e-6:
            return z
        # Far field: `beyond` (lid, or far without a Y-banded rod) bounds every band, so applying it
        # to all material is exact; the rod clearance applies to material within its Y band.
        need = max(z, self.max - head.beyond)
        if head.rod_band is not None:
            j0 = max(int(np.floor((pts[:, 1].min() - head.rod_band) / CELL)) + 1024, 0)
            j1 = int(np.floor((pts[:, 1].max() + head.rod_band) / CELL)) + 1025
            need = max(need, float(self.row_max[j0:j1].max()) - head.far)
        if not head.radii:
            return need
        m = head.reach + 1.0
        (x0, y0), (x1, y1) = pts.min(0) - m, pts.max(0) + m
        i0, i1 = int(np.floor(x0 / CELL)) + 1024, int(np.floor(x1 / CELL)) + 1025
        j0, j1 = int(np.floor(y0 / CELL)) + 1024, int(np.floor(y1 / CELL)) + 1025
        win = self.h[max(i0, 0):i1, max(j0, 0):j1]
        ix, iy = np.nonzero(win > z + 1e-6)
        if ix.size == 0:
            return need
        h = win[ix, iy].astype(float)
        c = np.stack([(ix + max(i0, 0) - 1024 + 0.5) * CELL, (iy + max(j0, 0) - 1024 + 0.5) * CELL], axis=1)
        d = np.maximum(0.0, _dist_to_polyline(c, pts) - CELL * 0.7071)
        return max(need, float((h - head_allow(head, d)).max()))


def _dist_to_polyline(c: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 1:
        return np.hypot(*(c - pts[0]).T)
    a, b = pts[:-1], pts[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    out = np.empty(len(c))
    step = max(1, 2_000_000 // len(a))  # bound the (cells x segments) temporaries
    for k in range(0, len(c), step):
        cc = c[k:k + step]
        t = np.clip(((cc[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        out[k:k + step] = np.sqrt(((cc[:, None, :] - proj) ** 2).sum(-1)).min(1)
    return out


class Emitter:
    def __init__(self, parsed: Parsed, head: HeadModel, synthesize_blocks: bool = True):
        self.p = parsed
        self.head = head
        n_tools = 1 + max([parsed.first_tool] + [s.tool for s in parsed.segments] + [b.to for b in parsed.blocks])
        self.flush = flush_matrix(parsed.config, n_tools)
        self.blocks: dict[tuple[int, int], Block] = {}
        for b in parsed.blocks:
            self.blocks.setdefault((b.frm, b.to), b)
        self.synth = synthesize_blocks
        self.stats = {"tool_changes": 0, "flush_mm3": 0.0, "synth_blocks": 0, "travel_lifts": 0}

    # -- tool change ----------------------------------------------------------------------
    def block_lines(self, a: int, b: int, zbase: float) -> tuple[list[str], float]:
        blk = self.blocks.get((a, b))
        scale = 1.0
        if blk is None:
            if not self.synth:
                raise KeyError(f"no tool change {a}->{b} in source G-code")
            same_to = [k for k in self.blocks if k[1] == b]
            if not same_to:
                raise KeyError(f"no tool change into T{b} in source G-code")
            src = max(same_to, key=lambda k: self.flush[k[0], b])
            blk = self.blocks[src]
            scale = self.flush[a, b] / self.flush[src[0], b] if self.flush[src[0], b] > 0 else 1.0
            self.stats["synth_blocks"] += 1
        out = []
        in_flush = False
        dz = zbase - blk.z_orig
        for ln in blk.lines:
            if ln.startswith("; FLUSH_START"):
                in_flush = True
            elif ln.startswith("; FLUSH_END"):
                in_flush = False
            elif scale != 1.0 and in_flush:
                m = _FLUSH_E.match(ln)
                if m and float(m.group(2)) > 0:
                    ln = f"{m.group(1)}{float(m.group(2)) * scale:.5g}{m.group(3)}"
            if scale != 1.0:
                m = _M620_10_L.match(ln)
                if m:
                    ln = f"{m.group(1)}{float(m.group(2)) * scale:.5g}{m.group(3)}"
            if is_move(ln) and "Z" in ln.split(";")[0]:
                ln = _Z_WORD.sub(lambda mm: f"Z{float(mm.group(2)) + dz:.3f}", ln, count=1)
            out.append(ln)
        self.stats["tool_changes"] += 1
        self.stats["flush_mm3"] += float(self.flush[a, b])
        return out, blk.post_r

    # -- main -----------------------------------------------------------------------------
    def emit(self, order: list[int], assign: dict[int, int] | None = None) -> str:
        """assign: unit -> filament for support units printed with another filament than Orca
        gave them (Scheduler.last_assign); their nozzle temperature is left at the loaded one's."""
        assign = assign or {}
        p, head = self.p, self.head
        out: list[str] = list(p.prelude)
        st = MachineState()
        for ln in p.prelude:
            st.feed(ln)
        hm = HeightMap()
        tool = p.first_tool
        retracted = p.prelude_r
        layer_seen: set[int] = set()
        for n, si in enumerate(order):
            s: Segment = p.segments[si]
            s_tool = assign.get(si, s.tool)
            if s_tool != tool:
                lines, retracted = self.block_lines(tool, s_tool, max(hm.max, s.z))
                for ln in lines:
                    rep = self._spiral_lift(ln, st, hm) if is_move(ln) else None
                    if rep is not None:
                        out += rep
                        continue
                    st.feed(ln)
                    out.append(ln)
                tool = s_tool
            out += ["; CHANGE_LAYER", f"; Z_HEIGHT: {s.z:g}", f"; LAYER_HEIGHT: {s.layer_height:g}"]
            if s.layer not in layer_seen:
                layer_seen.add(s.layer)
                out.extend(p.layers[s.layer].header)
            want = s.lead_state
            if s_tool != s.tool:
                want = want.copy()
                want.nozzle = None  # keep the loaded filament's temperature
            out.extend(self._restore(st, want))
            out.extend(self._transition(st, s, hm))
            if abs(retracted - s.lead_d) > 1e-4:
                out.append(f"G1 E{retracted - s.lead_d:.5f} F1800 ; colorbatch retract fix")
            if s.label:
                out += [f"; start printing object, unique label id: {s.label[0]}", f"M624 {s.label[1]}"]
            if s.feature:  # moved units must not inherit the previous unit's role/width in the preview
                out += [s.feature, f"; LINE_WIDTH: {s.width:g}"]
            for ln in s.lead:
                st.feed(ln)
                out.append(ln)
            out.append(f"G1 F{s.lead_f:.10g}")
            st.f = s.lead_f
            self._body(out, st, s, hm)
            if s.label:
                out += [f"; stop printing object, unique label id: {s.label[0]}", "M625"]
            hm.add(s.cells, s.z)
            retracted = s.tail_r
        out.extend(p.postlude)
        return "\n".join(out) + "\n"

    def _restore(self, st: MachineState, want: MachineState) -> list[str]:
        out = []
        if want.nozzle is not None and want.nozzle != st.nozzle:
            out.append(f"M104 S{want.nozzle:g}")
        if want.bed is not None and want.bed != st.bed:
            out.append(f"M140 S{want.bed:g}")
        for p_, v in sorted(want.fans.items()):
            if st.fans.get(p_) != v:
                out.append(f"M106 P{p_} S{v:g}")
        if want.accel is not None and want.accel != st.accel:
            out.append(f"M204 S{want.accel:g}")
        for ln in out:
            st.feed(ln)
        return out

    def _lift_to(self, st: MachineState, z: float) -> list[str]:
        ln = f"G1 Z{z:.3f} F{Z_F:g}"
        st.feed(ln)
        return [ln]

    def _transition(self, st: MachineState, s: Segment, hm: HeightMap) -> list[str]:
        x1, y1, z1 = s.lead_end
        pts = np.array([[st.x, st.y], [x1, y1]])
        need = hm.required_z(pts, max(st.z, z1), self.head)
        zt = max(st.z, z1 + HOP, need + HOP if need > max(st.z, z1) + 1e-6 else 0.0)
        out = []
        if zt > st.z + 1e-6:
            out += self._lift_to(st, zt)
        mv = f"G1 X{x1:.3f} Y{y1:.3f} F{TRAVEL_F:g}"
        st.feed(mv)
        out.append(mv)
        out += self._lift_to(st, z1)
        return out

    def _spiral_lift(self, ln: str, st: MachineState, hm: HeightMap) -> list[str] | None:
        """Orca's spiral z-hop (G2/G3 with I/J but no X/Y: a full circle while lifting) sweeps
        ~1.2 mm around the nozzle. If that circle passes too close to taller printed material,
        return a plain vertical lift to the same height instead; None if the arc is fine."""
        w = words(ln)
        if "X" in w or "Y" in w or not ("I" in w or "J" in w):
            return None
        pts = np.array(path_points(ln, st.x, st.y))
        if hm.required_z(pts, st.z, self.head) <= st.z + 1e-6:
            return None
        self.stats["spiral_lifts_straightened"] = self.stats.get("spiral_lifts_straightened", 0) + 1
        f = st.f
        out = self._lift_to(st, w.get("Z", st.z))
        out.append(f"G1 F{f:.10g}")
        st.f = f
        return out

    def _body(self, out: list[str], st: MachineState, s: Segment, hm: HeightMap) -> None:
        for ln in s.body:
            if is_move(ln):
                w = words(ln)
                e = w.get("E", 0.0)
                rep = self._spiral_lift(ln, st, hm) if e == 0 else None
                if rep is not None:
                    out += rep
                    continue
                if e <= 0 and ("X" in w or "Y" in w):
                    z_after = w.get("Z", st.z)
                    z_path = min(st.z, z_after)  # ramped moves start low: judge by the lower end
                    pts = np.array(path_points(ln, st.x, st.y, sample=False))
                    need = hm.required_z(pts, z_path, self.head)
                    if need > z_path + 1e-6:
                        self.stats["travel_lifts"] += 1
                        x1, y1 = w.get("X", st.x), w.get("Y", st.y)
                        f = st.f
                        out += self._lift_to(st, max(need + HOP, z_after))
                        mv = f"G1 X{x1:.3f} Y{y1:.3f} F{TRAVEL_F:g} ; colorbatch lifted travel"
                        st.feed(mv)
                        out.append(mv)
                        out += self._lift_to(st, z_after)
                        if e < 0:  # keep the wipe retraction
                            out.append(f"G1 E{e:.5f} F1800")
                        out.append(f"G1 F{f:.10g}")
                        st.f = f
                        continue
            st.feed(ln)
            out.append(ln)


def head_allow(head: HeadModel, d: np.ndarray) -> np.ndarray:
    """Clearance per horizontal distance. Beyond the profile it is `beyond` (the lid when the X rod
    is Y-banded): the rod itself is applied separately per Y band in HeightMap.required_z."""
    hs = np.array(head.heights + (head.beyond,))
    return hs[np.searchsorted(np.array(head.radii), d, side="right")]
