"""Order segments so one colour prints several layers before the next tool change.

Constraints for printing unit u (layer L, tool t, top z_u):
  support:   every unit of layer L-1 whose footprint touches u's footprint is printed;
  obstacle:  every printed unit w with z_w > z_u is at least head.min_distance(z_w - z_u) away;
  no-block:  every unprinted unit v with z_v < z_u is at least head.min_distance(z_u - z_v) away,
             so printing u never makes a later unit impossible (the lowest unprinted unit is
             therefore always printable and the greedy cannot deadlock).
"""
from __future__ import annotations

import numpy as np

from .gcode import CELL, Segment, cell_centres, decode, encode
from .head import HeadModel


class Geometry:
    def __init__(self, segs: list[Segment], head: HeadModel, unit_heads: list[HeadModel] | None = None):
        """unit_heads: optional per-unit head model (the head used when the nozzle prints that
        unit), e.g. tolerance calibration zones; default `head` for every unit."""
        self.head = head
        self.n = len(segs)
        self.z = np.array([s.z for s in segs])
        self.tool = np.array([s.tool for s in segs])
        self.layer = np.array([s.layer for s in segs])
        # Deduplicate identical footprints (prismatic parts repeat them layer after layer).
        fp_index: dict[bytes, int] = {}
        self.fp = np.zeros(self.n, dtype=np.int64)
        self.fp_cells: list[np.ndarray] = []
        self.fp_pts: list[np.ndarray] = []
        bbox = []
        for i, s in enumerate(segs):
            key = s.cells.tobytes()
            k = fp_index.get(key)
            if k is None:
                k = fp_index[key] = len(self.fp_cells)
                self.fp_cells.append(s.cells)
                self.fp_pts.append(cell_centres(_boundary(s.cells)))
            self.fp[i] = k
            pts = self.fp_pts[k]
            bbox.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
        self.bbox = np.array(bbox)
        self._dist: dict[tuple[int, int], float] = {}
        heads = unit_heads or [head] * len(segs)
        self._hid = np.zeros(self.n, dtype=np.int64)
        self._tables: list[tuple[np.ndarray, np.ndarray]] = []
        seen: dict[HeadModel, int] = {}
        for i, hm in enumerate(heads):
            k = seen.get(hm)
            if k is None:
                k = seen[hm] = len(self._tables)
                r = np.array(hm.radii + (np.inf,))
                self._tables.append((np.concatenate([[0.0], r[:-1]]), np.array(hm.heights + (hm.far,))))
            self._hid[i] = k

    def min_distance(self, dz: np.ndarray, hid) -> np.ndarray:
        """Required distance for obstacles dz above the nozzle; hid = head table per element (or scalar)."""
        hid = np.broadcast_to(np.asarray(hid), dz.shape)
        out = np.zeros(dz.shape)
        for k in np.unique(hid):
            r_lo, h = self._tables[int(k)]
            m = hid == k
            idx = np.searchsorted(h, dz[m] - 1e-9, side="left")
            out[m] = np.where(idx < len(h), r_lo[np.minimum(idx, len(h) - 1)], np.inf)
        return np.where(dz <= 1e-9, 0.0, out)

    def bbox_gap(self, u: int, others: np.ndarray) -> np.ndarray:
        a, b = self.bbox[u], self.bbox[others]
        dx = np.maximum(0.0, np.maximum(b[:, 0] - a[2], a[0] - b[:, 2]))
        dy = np.maximum(0.0, np.maximum(b[:, 1] - a[3], a[1] - b[:, 3]))
        return np.maximum(0.0, np.hypot(dx, dy) - CELL * 1.4143)  # lower bound of dist()

    def dist(self, u: int, v: int) -> float:
        a, b = int(self.fp[u]), int(self.fp[v])
        key = (a, b) if a < b else (b, a)
        d = self._dist.get(key)
        if d is None:
            if np.intersect1d(self.fp_cells[a], self.fp_cells[b], assume_unique=True).size:
                d = 0.0
            else:
                pa, pb = self.fp_pts[a], self.fp_pts[b]
                best = np.inf
                for k in range(0, len(pa), 512):
                    diff = pa[k:k + 512, None, :] - pb[None, :, :]
                    best = min(best, float(np.sqrt((diff ** 2).sum(-1)).min()))
                # Nozzle axis lies somewhere in u's cells, obstacle material anywhere in v's cells:
                # subtract both half-diagonals (same convention as the emitter/validator).
                d = max(0.0, best - CELL * 1.4143)
            self._dist[key] = d
        return d

    def clear(self, u: int, others: np.ndarray, dz: np.ndarray, nozzle_at_u: bool) -> bool:
        """True if every unit in `others` (height difference dz>0) keeps its required distance.
        nozzle_at_u: the nozzle prints u (others are taller) vs. later prints others (u is taller)."""
        if others.size == 0:
            return True
        need = self.min_distance(dz, self._hid[u] if nozzle_at_u else self._hid[others])
        if np.isinf(need).any():
            return False
        gap = self.bbox_gap(u, others)
        close = gap < need
        for v, nd in zip(others[close], need[close]):
            if self.dist(u, int(v)) < nd:
                return False
        return True


def _interior(cells: np.ndarray) -> np.ndarray:
    """Mask of cells whose four neighbours are all in the footprint."""
    ix, iy = decode(cells)
    keep = np.ones(len(cells), dtype=bool)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        keep &= np.isin(encode(ix + dx, iy + dy), cells, assume_unique=False)
    return keep


def _boundary(cells: np.ndarray) -> np.ndarray:
    edge = ~_interior(cells)
    return cells[edge] if edge.any() else cells


def _core(cells: np.ndarray) -> np.ndarray:
    """Footprint eroded by one cell (the cell itself if that leaves nothing, e.g. a single wall)."""
    keep = _interior(cells)
    return cells[keep] if keep.any() else cells


FINE = 0.05     # mm, grid for the stacked-vs-side-by-side test
SHRINK = 0.1    # mm taken off each line's half width before testing for overlap


def densify(s: np.ndarray, step: float = FINE, max_gap: float = 0.3) -> np.ndarray:
    """Insert points every `step` between consecutive samples of the same path (gap <= max_gap),
    so narrowed line stamps stay continuous instead of dotted."""
    if len(s) < 2:
        return s
    d = np.hypot(*(s[1:, :2] - s[:-1, :2]).T)
    n = np.where(d <= max_gap, np.ceil(d / step).astype(int), 1)
    n = np.maximum(n, 1)
    idx = np.repeat(np.arange(len(s) - 1), n)
    frac = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    pts = s[idx, :2] + (s[idx + 1, :2] - s[idx, :2]) * frac[:, None]
    out = np.column_stack([pts, s[idx, 2]])
    return np.vstack([out, s[-1:]])


def _stacked(upper: Segment, lower: Segment) -> bool:
    """True if `upper`'s extrusions sit on `lower`'s (not just beside them).

    Both are rasterised on a 0.05 mm grid with every line narrowed by SHRINK on each side; any
    remaining overlap means upper overhangs lower by more than ~2*SHRINK, so lower must go first.
    Side-by-side colours only touch at the line edges and drop out after the narrowing.
    """
    a, b = upper.samples, lower.samples
    if a is None or b is None or not len(a) or not len(b):
        return False
    m = float(max(a[:, 2].max(), b[:, 2].max()))
    lo = np.maximum(a[:, :2].min(0), b[:, :2].min(0)) - m
    hi = np.minimum(a[:, :2].max(0), b[:, :2].max(0)) + m
    if (hi <= lo).any():
        return False
    shape = np.ceil((hi - lo) / FINE).astype(int) + 1

    def stamp(s):
        s = s[np.all((s[:, :2] >= lo - 0.5) & (s[:, :2] <= hi + 0.5), axis=1)]
        s = densify(s)
        s = s[np.all((s[:, :2] >= lo - s[:, 2:3]) & (s[:, :2] <= hi + s[:, 2:3]), axis=1)]
        grid = np.zeros(shape, dtype=bool)
        for r in np.unique(s[:, 2]):
            rr = r - SHRINK
            if rr <= 0:
                continue
            k = int(np.ceil(rr / FINE))
            g = np.arange(-k, k + 1)
            dx, dy = np.meshgrid(g, g, indexing="ij")
            disk = (dx * dx + dy * dy) * FINE * FINE <= rr * rr
            ox, oy = dx[disk], dy[disk]
            c = np.round((s[s[:, 2] == r, :2] - lo) / FINE).astype(int)
            ix = (c[:, None, 0] + ox[None]).ravel()
            iy = (c[:, None, 1] + oy[None]).ravel()
            ok = (ix >= 0) & (iy >= 0) & (ix < shape[0]) & (iy < shape[1])
            grid[ix[ok], iy[ok]] = True
        return grid

    return bool((stamp(a) & stamp(b)).any())


def support_deps(segs: list[Segment], reach_layers: int = 1) -> list[list[int]]:
    """u depends on the units below it that it lies on.

    Cores (footprints eroded by one 0.5 mm cell) that overlap mean plainly stacked units. Units
    whose footprints only touch are checked on a fine grid (_stacked): a colour that overhangs
    the one below it by a fraction of a line -- e.g. along a sloped colour interface -- still needs
    it printed first, while side-by-side colours do not (their height difference is governed by
    the head clearance rules). The check reaches `reach_layers` down: with a tip tolerance a colour
    may run that many layers ahead of a touching one, and a slow creep (0.07 mm per layer on a
    taper) adds up to a real overhang over the unprinted unit.
    """
    by_layer: dict[int, list[int]] = {}
    for i, s in enumerate(segs):
        by_layer.setdefault(s.layer, []).append(i)
    cores = [_core(s.cells) for s in segs]
    deps: list[list[int]] = [[] for _ in segs]
    for i, s in enumerate(segs):
        for k in range(1, max(1, reach_layers) + 1):
            for j in by_layer.get(s.layer - k, []):
                if k == 1 and np.intersect1d(cores[i], cores[j], assume_unique=True).size:
                    deps[i].append(j)
                elif np.intersect1d(s.cells, segs[j].cells, assume_unique=True).size and _stacked(s, segs[j]):
                    deps[i].append(j)
    return deps


class Scheduler:
    def __init__(self, segs: list[Segment], head: HeadModel, flush: np.ndarray, first_tool: int,
                 unit_heads: list[HeadModel] | None = None):
        self.segs = segs
        self.g = Geometry(segs, head, unit_heads)
        # How many layers a colour can run ahead of a touching one: the head's allowance right at
        # the nozzle (tip tolerance, or the fixed height) over the thinnest layer.
        tip = max(h.allow(0.0) for h in (unit_heads or [head]))
        min_lh = min((s.layer_height for s in segs if s.layer_height > 0), default=0.2)
        self.deps = support_deps(segs, 1 + int(np.ceil(tip / min_lh - 1e-9)))
        self.flush = flush
        self.first_tool = first_tool
        self.extr = np.array([s.extruded for s in segs])
        self.dependents: list[list[int]] = [[] for _ in segs]
        for u, ds in enumerate(self.deps):
            for d in ds:
                self.dependents[d].append(u)
        self.lead_xy = np.array([s.lead_end[:2] for s in segs])

    # -- state ----------------------------------------------------------------------------
    def _init_state(self):
        n = len(self.segs)
        remaining = np.array([len(d) for d in self.deps])
        ready: dict[int, set[int]] = {}
        for u in np.nonzero(remaining == 0)[0]:
            ready.setdefault(int(self.g.tool[u]), set()).add(int(u))
        return np.zeros(n, dtype=bool), remaining, ready

    @staticmethod
    def _copy(state):
        printed, remaining, ready = state
        return printed.copy(), remaining.copy(), {t: set(s) for t, s in ready.items()}

    def _mark(self, state, u: int) -> None:
        printed, remaining, ready = state
        printed[u] = True
        ready[int(self.g.tool[u])].discard(u)
        for v in self.dependents[u]:
            remaining[v] -= 1
            if remaining[v] == 0:
                ready.setdefault(int(self.g.tool[v]), set()).add(v)

    def feasible(self, u: int, printed: np.ndarray) -> bool:
        """Clearance rules only (support is guaranteed by the ready sets)."""
        g = self.g
        zu = g.z[u]
        higher = np.nonzero(printed & (g.z > zu + 1e-9))[0]
        if not g.clear(u, higher, g.z[higher] - zu, nozzle_at_u=True):
            return False
        lower = np.nonzero(~printed & (g.z < zu - 1e-9))[0]
        return g.clear(u, lower, zu - g.z[lower], nozzle_at_u=False)

    def _pick(self, state, tool: int, pos) -> int | None:
        cands = state[2].get(tool)
        if not cands:
            return None
        c = np.fromiter(cands, dtype=np.int64)
        d = np.hypot(self.lead_xy[c, 0] - pos[0], self.lead_xy[c, 1] - pos[1])
        for u in c[np.lexsort((d, self.g.z[c]))]:  # lowest first, then nearest
            if self.feasible(int(u), state[0]):
                return int(u)
        return None

    def batch(self, state, tool: int, pos) -> list[int]:
        """Greedy run of `tool` on `state` (mutated): repeatedly print its lowest feasible unit."""
        seq = []
        while True:
            u = self._pick(state, tool, pos)
            if u is None:
                return seq
            self._mark(state, u)
            seq.append(u)
            pos = self.lead_xy[u]

    def cost(self, order: list[int]) -> tuple[int, float]:
        """(tool changes, flush volume) of an order, starting from the first tool."""
        changes, flush, tool = 0, 0.0, self.first_tool
        for u in order:
            t = int(self.g.tool[u])
            if t != tool:
                changes += 1
                flush += float(self.flush[tool, t])
                tool = t
        return changes, flush

    def run(self, strategy: str = "snake") -> list[int]:
        """snake:  rounds over all tools, direction alternating so the last tool continues;
                   each turn prints everything the tool can (a full lowest layer always fits,
                   so rounds <= layers).
        greedy: after each batch, switch to the tool whose look-ahead batch extrudes most
                   per unit of flush."""
        n = len(self.segs)
        state = self._init_state()
        order: list[int] = []
        tool = self.first_tool
        pos = (0.0, 0.0)
        tools = sorted({int(t) for t in self.g.tool} | {tool})
        k = tools.index(tool)
        cycle = tools[k:] + tools[:k]
        rnd = 0
        while len(order) < n:
            got = self.batch(state, tool, pos)
            order.extend(got)
            if got:
                pos = self.lead_xy[got[-1]]
            if len(order) == n:
                break
            avail = [t for t, s in state[2].items() if s and t != tool]
            if strategy == "snake":
                # Next tool in the current sweep direction that has something ready.
                seq = cycle if rnd % 2 == 0 else cycle[::-1]
                i = seq.index(tool)
                nxt = next((t for t in seq[i + 1:] if t in avail), None)
                if nxt is None:
                    rnd += 1
                    seq = cycle if rnd % 2 == 0 else cycle[::-1]
                    nxt = next((t for t in seq[seq.index(tool) + 1:] + seq if t in avail), None)
                tool = nxt
            else:
                best, best_score = None, -1.0
                for t in sorted(avail):
                    run = self.batch(self._copy(state), t, pos)
                    gain = float(self.extr[run].sum()) if run else 0.0
                    score = gain / (self.flush[tool, t] + 1.0)
                    if gain > 0 and score > best_score:
                        best, best_score = t, score
                tool = best
            if tool is None:
                raise RuntimeError("no printable unit: scheduler invariant broken")
        return order
