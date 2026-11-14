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
    def __init__(self, segs: list[Segment], head: HeadModel):
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
        r = np.array(head.radii + (np.inf,))
        h = np.array(head.heights + (head.far,))
        self._r_lo = np.concatenate([[0.0], r[:-1]])
        self._h = h

    def min_distance(self, dz: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self._h, dz - 1e-9, side="left")
        out = np.where(idx < len(self._h), self._r_lo[np.minimum(idx, len(self._h) - 1)], np.inf)
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

    def clear(self, u: int, others: np.ndarray, dz: np.ndarray) -> bool:
        """True if every unit in `others` (height difference dz>0 to u) keeps its required distance."""
        if others.size == 0:
            return True
        need = self.min_distance(dz)
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


def support_deps(segs: list[Segment]) -> list[list[int]]:
    """u depends on the units one layer down that actually lie beneath it.

    Cores (eroded footprints) are compared so that side-by-side colours whose rasterised
    perimeters merely touch do not become supports of each other; their height difference
    is governed by the head clearance rules instead.
    """
    by_layer: dict[int, list[int]] = {}
    for i, s in enumerate(segs):
        by_layer.setdefault(s.layer, []).append(i)
    cores = [_core(s.cells) for s in segs]
    deps: list[list[int]] = [[] for _ in segs]
    for i, s in enumerate(segs):
        for j in by_layer.get(s.layer - 1, []):
            if np.intersect1d(cores[i], cores[j], assume_unique=True).size:
                deps[i].append(j)
    return deps


class Scheduler:
    def __init__(self, segs: list[Segment], head: HeadModel, flush: np.ndarray, first_tool: int):
        self.segs = segs
        self.g = Geometry(segs, head)
        self.deps = support_deps(segs)
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
        if not g.clear(u, higher, g.z[higher] - zu):
            return False
        lower = np.nonzero(~printed & (g.z < zu - 1e-9))[0]
        return g.clear(u, lower, zu - g.z[lower])

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
