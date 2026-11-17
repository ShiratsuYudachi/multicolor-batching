"""Parse Orca/Bambu single-nozzle multi-material G-code into reorderable units.

The body (first `; CHANGE_LAYER` .. machine end G-code) is split into
  * Segments: everything one tool prints in one layer between two boundaries,
  * Blocks:   tool-change blocks (`; filament end gcode` .. end of filament start gcode).
Segments are later re-emitted in a different order, so each one records what it needs to be
self-contained: lead-in end position, retraction state, machine state (temps/fans/accel/feed),
object label, and its XY footprint for collision checks.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

CELL = 0.5  # footprint grid, mm
_WORD = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
_MOVE = re.compile(r"^G[0-3](?:\s|$)")


def words(line: str) -> dict[str, float]:
    code = line.split(";", 1)[0]
    return {k.upper(): float(v) for k, v in _WORD.findall(code[2:] if len(code) > 2 else "")}


def is_move(line: str) -> bool:
    return bool(_MOVE.match(line))


@dataclass
class MachineState:
    """Subset of printer state that a moved segment must restore."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    f: float = 0.0
    nozzle: float | None = None   # M104/M109 S
    bed: float | None = None      # M140/M190 S
    fans: dict = field(default_factory=dict)  # M106 P -> S
    accel: float | None = None    # M204 S

    def copy(self) -> "MachineState":
        s = MachineState(self.x, self.y, self.z, self.f, self.nozzle, self.bed, dict(self.fans), self.accel)
        return s

    def feed(self, line: str) -> tuple[float, bool]:
        """Update state; return (E delta, is_extruding_xy_move)."""
        s = line.lstrip()
        if not s or s[0] == ";":
            return 0.0, False
        head = s.split(None, 1)[0].split(";")[0].upper()
        if head in ("G0", "G1", "G2", "G3"):
            w = words(s)
            if "F" in w:
                self.f = w["F"]
            e = w.get("E", 0.0)
            xy = "X" in w or "Y" in w
            self.x = w.get("X", self.x)
            self.y = w.get("Y", self.y)
            self.z = w.get("Z", self.z)
            return e, (xy and e > 0)
        if head in ("M104", "M109"):
            w = words(s)
            if "S" in w:
                self.nozzle = w["S"]
        elif head in ("M140", "M190"):
            w = words(s)
            if "S" in w:
                self.bed = w["S"]
        elif head == "M106":
            w = words(s)
            self.fans[int(w.get("P", 1))] = w.get("S", 255.0)
        elif head == "M107":
            self.fans[int(words(s).get("P", 1))] = 0.0
        elif head == "M204":
            w = words(s)
            if "S" in w:
                self.accel = w["S"]
        return 0.0, False


@dataclass
class Segment:
    idx: int
    layer: int
    z: float
    layer_height: float
    tool: int
    lead: list[str]              # non-motion lines of the lead-in (accel, deretract, comments)
    body: list[str]              # from first extrusion to end (incl. tail retraction/wipe)
    lead_end: tuple[float, float, float]
    lead_f: float
    lead_state: MachineState     # state the body expects (after the lead-in)
    lead_d: float                # positive E of the lead-in (deretraction)
    tail_r: float = 0.0          # retraction after the last extrusion
    label: tuple[str, str] | None = None  # (unique label id, M624 payload)
    cells: np.ndarray | None = None       # encoded footprint cells
    extruded: float = 0.0
    feature: str | None = None   # "; FEATURE: ..." in effect at the first extrusion (preview tag)
    width: float = 0.45          # line width in effect at the first extrusion
    paths: list | None = None    # per extrusion line: (N, 3) centre-line points x, y and half width (mm)
    support: bool = False        # only support material: any loaded filament may print it


@dataclass
class Block:
    frm: int
    to: int
    lines: list[str]
    z_orig: float     # layer z the block was generated at (its Z moves are relative to this)
    post_r: float = 0.0  # retraction state the following segment expects


@dataclass
class Layer:
    idx: int
    z: float
    height: float
    header: list[str] = field(default_factory=list)


@dataclass
class Parsed:
    prelude: list[str]
    postlude: list[str]
    segments: list[Segment]
    blocks: list[Block]
    layers: list[Layer]
    first_tool: int
    config: dict[str, str]
    prelude_r: float  # retraction state at the end of the prelude


def read_config(lines: list[str]) -> dict[str, str]:
    cfg = {}
    inside = False
    for ln in lines:
        if ln.startswith("; CONFIG_BLOCK_START"):
            inside = True
        elif ln.startswith("; CONFIG_BLOCK_END"):
            break
        elif inside and ln.startswith("; ") and " = " in ln:
            k, v = ln[2:].split(" = ", 1)
            cfg[k.strip()] = v.strip()
    return cfg


# --- footprint rasterisation -------------------------------------------------------------

def _disk_offsets(radius_cells: float) -> np.ndarray:
    r = int(math.ceil(radius_cells))
    g = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(g, g, indexing="ij")
    m = dx * dx + dy * dy <= radius_cells * radius_cells + 1e-9
    return np.stack([dx[m], dy[m]], axis=1)


_OFFS_CACHE: dict[int, np.ndarray] = {}


def encode(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return ix.astype(np.int64) * 8192 + iy.astype(np.int64)


def decode(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return c // 8192, c % 8192


def rasterise(samples: list[tuple[float, float, float]]) -> np.ndarray:
    """samples: (x, y, half_width) points along extrusions -> unique encoded cells."""
    if not samples:
        return np.zeros(0, dtype=np.int64)
    a = np.asarray(samples)
    out = []
    for hw_key in np.unique(np.round(a[:, 2] / CELL * 4).astype(int)):
        sel = np.round(a[:, 2] / CELL * 4).astype(int) == hw_key
        offs = _OFFS_CACHE.get(hw_key)
        if offs is None:
            offs = _OFFS_CACHE[hw_key] = _disk_offsets(hw_key / 4.0)
        ix = np.floor(a[sel, 0] / CELL).astype(np.int64) + 1024
        iy = np.floor(a[sel, 1] / CELL).astype(np.int64) + 1024
        out.append(encode((ix[:, None] + offs[None, :, 0]).ravel(), (iy[:, None] + offs[None, :, 1]).ravel()))
    return np.unique(np.concatenate(out))


def cell_centres(cells: np.ndarray) -> np.ndarray:
    ix, iy = decode(cells)
    return np.stack([(ix - 1024 + 0.5) * CELL, (iy - 1024 + 0.5) * CELL], axis=1)


def _arc_points(x0, y0, w, cw):
    x1, y1 = w.get("X", x0), w.get("Y", y0)
    cx, cy = x0 + w.get("I", 0.0), y0 + w.get("J", 0.0)
    r = math.hypot(x0 - cx, y0 - cy)
    a0, a1 = math.atan2(y0 - cy, x0 - cx), math.atan2(y1 - cy, x1 - cx)
    if cw:
        while a1 >= a0:
            a1 -= 2 * math.pi
    else:
        while a1 <= a0:
            a1 += 2 * math.pi
    if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
        a1 = a0 - 2 * math.pi if cw else a0 + 2 * math.pi
    n = max(2, int(abs(a1 - a0) * r / 0.25) + 1)
    t = np.linspace(a0, a1, n)
    return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))


def path_points(line: str, x0: float, y0: float, sample: bool = True) -> list[tuple[float, float]]:
    """Polyline of a G0-G3 move starting at (x0, y0); straight moves are sampled every ~0.25 mm
    (for rasterising) or returned as their two end points (sample=False, for distance queries)."""
    w = words(line)
    head = line.split(None, 1)[0].upper()
    if head in ("G2", "G3") and ("I" in w or "J" in w):
        return _arc_points(x0, y0, w, head == "G2")
    x1, y1 = w.get("X", x0), w.get("Y", y0)
    if not sample:
        return [(x0, y0), (x1, y1)]
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.25))
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]


# --- splitting ---------------------------------------------------------------------------

_BLOCK_END = ("; printing object", "; start printing object", "; stop printing object", "M624", "M625",
              "; FEATURE", "M204", "G1 X", "G1 Y", "G0 ", "G2 ", "G3 ", "G17")
_LABEL_START = re.compile(r"^; start printing object, unique label id: (\S+)")


def _find_postlude(lines: list[str], body_start: int) -> int:
    """Index where the machine end G-code (and trailing stop-label/fan-off lines) starts."""
    end_custom = max(i for i, ln in enumerate(lines) if ln.startswith("; FEATURE: Custom"))
    st = MachineState()
    last = body_start
    for i in range(body_start, end_custom):
        e, ext = st.feed(lines[i])
        if ext or lines[i].startswith("; WIPE_END") or (is_move(lines[i]) and e < 0):
            last = i
    return last + 1


def split(text: str, island_gap: float = 3.5) -> Parsed:
    lines = text.splitlines()
    cfg = read_config(lines)
    body_start = next(i for i, ln in enumerate(lines) if ln.startswith("; CHANGE_LAYER"))
    post_start = _find_postlude(lines, body_start)
    prelude, body, postlude = lines[:body_start], lines[body_start:post_start], lines[post_start:]

    tools = [int(m.group(1)) for ln in prelude for m in [re.match(r"^M620 S(\d+)A", ln)] if m]
    if not tools:  # start G-code without AMS mapping: plain T<n> (T255/T1000 are Bambu specials)
        tools = [int(m.group(1)) for ln in prelude for m in [re.match(r"^T(\d+)\s*(?:;|$)", ln)]
                 if m and int(m.group(1)) < 255]
    first_tool = tools[-1] if tools else 0

    st = MachineState()
    for ln in prelude:
        st.feed(ln)

    raw_segs: list[tuple[int, MachineState, list[str], int, bool]] = []  # (layer, state, lines, tool, layer_start)
    blocks: list[Block] = []
    block_after_seg: dict[int, int] = {}   # raw seg index -> block index following it
    layers: list[Layer] = []
    tool = first_tool
    cur: list[str] | None = None
    blk: Block | None = None
    blk_started = False
    i = 0
    while i < len(body):
        ln = body[i]
        if blk is not None:
            if blk_started and ln.strip() and ln.startswith(_BLOCK_END):
                blocks.append(blk)
                tool = blk.to
                blk = None
                cur = []
                raw_segs.append((len(layers) - 1, st.copy(), cur, tool, False))
                continue  # re-process this line as segment content
            blk.lines.append(ln)
            st.feed(ln)
            m = re.match(r"^M620 S(\d+)A", ln)
            if m:
                blk.to = int(m.group(1))
            if ln.startswith("; filament start gcode"):
                blk_started = True
            i += 1
            continue
        if ln.startswith("; CHANGE_LAYER"):
            z = h = 0.0
            j = i + 1
            while j < len(body) and body[j].startswith(("; Z_HEIGHT", "; LAYER_HEIGHT")):
                v = float(body[j].split(":", 1)[1])
                if body[j].startswith("; Z_HEIGHT"):
                    z = v
                else:
                    h = v
                j += 1
            layers.append(Layer(len(layers), z, h))
            cur = []
            raw_segs.append((len(layers) - 1, st.copy(), cur, tool, True))
            i = j
            continue
        if ln.startswith("; filament end gcode"):
            block_after_seg[len(raw_segs) - 1] = len(blocks)
            blk = Block(frm=tool, to=-1, lines=[ln], z_orig=layers[-1].z)
            blk_started = False
            i += 1
            continue
        cur.append(ln)
        st.feed(ln)
        i += 1
    if blk is not None:
        raise ValueError("unterminated tool-change block")

    segments: list[Segment] = []
    raw_to_seg: dict[int, int] = {}  # raw segment -> index of its first unit
    label: tuple[str, str] | None = None
    for ri, (li, st0, seg_lines, seg_tool, layer_start) in enumerate(raw_segs):
        units, label = _build_units(len(segments), layers[li], seg_lines, st0, seg_tool, label,
                                    segments[-1] if segments else None, layer_start, island_gap)
        if not units:
            continue
        raw_to_seg[ri] = len(segments)
        segments.extend(units)

    # Retraction the segment after each block expects, and the prelude hand-off.
    for ri, bi in block_after_seg.items():
        nxt = next((raw_to_seg[k] for k in sorted(raw_to_seg) if k > ri), None)
        blocks[bi].post_r = segments[nxt].lead_d if nxt is not None else 0.0
    return Parsed(prelude, postlude, segments, blocks, layers, first_tool, cfg,
                  segments[0].lead_d if segments else 0.0)


def _take_layer_header(lines: list[str], layer: Layer, prev: Segment | None) -> list[str]:
    """Move the layer-change retract/wipe to the previous segment's tail, header lines to the layer."""
    i = 0
    in_wipe = False
    tail: list[str] = []
    # Leading retract + wipe happen where the nozzle physically is: the end of the previous segment.
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("; WIPE_START"):
            in_wipe = True
        if in_wipe or ln.startswith("G1 E-"):
            tail.append(ln)
            if ln.startswith("; WIPE_END"):
                in_wipe = False
            i += 1
            continue
        break
    out: list[str] = []
    if prev is not None:
        prev.body.extend(tail)
        prev.tail_r += -sum(min(words(t).get("E", 0.0), 0.0) for t in tail if is_move(t))
    else:
        out.extend(tail)
    for ln in lines[i:]:
        if ln.startswith(("M73 L", "M991 ", "; layer num/total_layer_count", "; update layer progress")):
            layer.header.append(ln)
        else:
            out.append(ln)
    return out


def _strip_timelapse(lines: list[str]) -> list[str]:
    """Drop the per-layer timelapse snapshot: it parks the head at layer-z, which is unsafe out of order."""
    out, skip = [], False
    for ln in lines:
        if ln.startswith("; SKIPTYPE:"):
            skip = ln.strip() == "; SKIPTYPE: timelapse"
            if skip:
                continue
        if ln.startswith("; SKIPPABLE_END"):
            skip = False
        if not skip:
            out.append(ln)
    return out


def _strip_labels(lines, label):
    """Object labels (skip-object) are re-generated around every emitted unit."""
    kept = []
    for ln in lines:
        m = _LABEL_START.match(ln)
        if m:
            label = (m.group(1), label[1] if label else "")
            continue
        if ln.startswith("M624 "):
            label = (label[0] if label else "0", ln[5:].strip())
            continue
        if ln.startswith(("M625", "; stop printing object, unique label id")):
            continue
        kept.append(ln)
    return kept, label


def _build_units(idx0, layer, lines, st0, tool, label, prev, layer_start, island_gap):
    """Turn one (layer, tool) run into one unit per island cluster (island_gap <= 0: a single unit).

    The run is cut at every travel move into chunks; chunks whose footprints come closer than
    `island_gap` are clustered (union-find), and each cluster becomes one unit whose body is its
    chunks in original order. Joins between non-adjacent chunks get the state the next chunk
    expected (feature/width tags, feed, accel, fans, retraction).
    """
    if layer_start:
        lines = _take_layer_header(lines, layer, prev)
    lines = _strip_timelapse(lines)
    lines, label = _strip_labels(lines, label)

    # One replay: state before each line, extrusion samples, travel candidates.
    st = st0.copy()
    states: list[MachineState] = []
    feats: list[tuple[str | None, float]] = []
    feature, width = None, 0.45
    ext_lines: list[int] = []
    samples: dict[int, list] = {}
    travels: list[int] = []
    e_of: list[float] = []
    for k, ln in enumerate(lines):
        if ln.startswith("; FEATURE:"):
            feature = ln
        elif ln.startswith("; LINE_WIDTH:"):
            width = float(ln.split(":", 1)[1])
        states.append(st.copy())
        feats.append((feature, width))
        x0, y0 = st.x, st.y
        e, ext = st.feed(ln)
        e_of.append(e if is_move(ln) else 0.0)
        if ext:
            ext_lines.append(k)
            samples[k] = [(px, py, width / 2) for px, py in path_points(ln, x0, y0)]
        elif is_move(ln):
            w = words(ln)
            if ("X" in w or "Y" in w) and "E" not in w:
                travels.append(k)
    states.append(st.copy())
    feats.append((feature, width))

    if not ext_lines:
        # Nothing printed (empty layer remainder): fold its retraction into the previous unit.
        if prev is not None:
            prev.tail_r -= sum(e for ln, e in zip(lines, e_of) if is_move(ln) and not ("X" in ln or "Y" in ln))
        return [], label

    first, last = ext_lines[0], ext_lines[-1]
    cuts = [0] + ([t for t in travels if first < t < last] if island_gap > 0 else [])
    bounds = cuts + [len(lines)]
    chunks = []  # (a, b, extrusion line indices, cells)
    for a, b in zip(bounds[:-1], bounds[1:]):
        ks = [k for k in ext_lines if a <= k < b]
        if not ks and chunks:  # pure travel/retract chunk: stays glued to its predecessor
            pa, _, pks, pc = chunks[-1]
            chunks[-1] = (pa, b, pks, pc)
            continue
        chunks.append((a, b, ks, rasterise([p for k in ks for p in samples[k]])))

    # Support-only chunks (every extrusion is Support / Support interface) become their own units,
    # so the scheduler may print them with whichever filament is loaded (see Segment.support).
    is_sup = [bool(ks) and all((feats[k][0] or "").startswith("; FEATURE: Support") for k in ks)
              for _, _, ks, _ in chunks]
    parent = list(range(len(chunks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if len(chunks) > 1:
        # Coarse grid with island_gap-sized cells: chunks in the same or neighbouring coarse cells
        # are merged (merges a little more than strictly within the gap; never less).
        k = max(1, int(round(island_gap / CELL)))
        owner: dict[tuple[int, bool], int] = {}
        for i, (_, _, _, cells) in enumerate(chunks):
            ix, iy = decode(cells)
            coarse = np.unique(encode(ix // k, iy // k)).tolist()
            for c in coarse:
                for d in (-8193, -8192, -8191, -1, 0, 1, 8191, 8192, 8193):
                    j = owner.get((c + d, is_sup[i]))
                    if j is not None and find(i) != find(j):
                        parent[find(i)] = find(j)
            for c in coarse:
                owner.setdefault((c, is_sup[i]), i)
    clusters: dict[int, list[int]] = {}
    for i in range(len(chunks)):
        clusters.setdefault(find(i), []).append(i)

    def tail_r(c):
        a, b, ks, _ = chunks[c]
        return -sum(min(e_of[k], 0.0) for k in range(ks[-1] + 1, b))

    units: list[Segment] = []
    for members in sorted(clusters.values()):
        a0, b0, ks0, _ = chunks[members[0]]
        f = ks0[0]
        lead, lead_d = [], 0.0
        for k in range(a0, f):
            ln = lines[k]
            if is_move(ln):
                w = words(ln)
                if "X" in w or "Y" in w or "Z" in w:
                    continue  # lead-in motion is regenerated by the emitter
                lead_d += e_of[k]
            lead.append(ln)
        body = list(lines[f:b0])
        for prev_c, c in zip(members, members[1:]):
            a, b, _, _ = chunks[c]
            if c != prev_c + 1:  # non-adjacent join: restore what chunk c expected
                want, have = states[a], states[chunks[prev_c][1]]
                fx = []
                if want.accel is not None and want.accel != have.accel:
                    fx.append(f"M204 S{want.accel:g}")
                fx += [f"M106 P{p_} S{v:g}" for p_, v in sorted(want.fans.items()) if have.fans.get(p_) != v]
                dr = tail_r(prev_c) - tail_r(c - 1)
                if abs(dr) > 1e-4:
                    fx.append(f"G1 E{dr:.5f} F1800 ; colorbatch retract fix")
                ft, wd = feats[a]
                if ft:
                    fx += [ft, f"; LINE_WIDTH: {wd:g}"]
                body += fx + [f"G1 F{want.f:.10g}"]
            body += lines[a:b]
        s0 = states[f]
        seg = Segment(idx0 + len(units), layer.idx, layer.z, layer.height, tool, lead, body,
                      (s0.x, s0.y, s0.z), s0.f, s0.copy(), lead_d, label=label)
        seg.feature, seg.width = feats[f]
        seg.extruded = sum(e_of[k] for c in members for k in chunks[c][2])
        seg.tail_r = tail_r(members[-1])
        seg.cells = np.unique(np.concatenate([chunks[c][3] for c in members]))
        seg.support = is_sup[members[0]]
        seg.paths = [np.array(samples[k], dtype=float) for c in members for k in chunks[c][2]]
        units.append(seg)
    return units, label
