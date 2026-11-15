"""Cross-layer colour batching for single-nozzle multi-material G-code (Bambu A1 / AMS lite)."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .emit import Emitter
from .gcode import cell_centres, split
from .head import HeadModel
from .schedule import Scheduler

__all__ = ["Options", "batch_gcode", "HeadModel"]


@dataclass
class Options:
    mode: str = "geometry"         # "off" | "fixed" | "geometry"
    fixed_height: float = 0.4      # mm, fixed mode
    head_profile: list | None = None  # [(radius, height)], geometry mode; None = A1 estimate
    head_far: float | None = None
    tip_tolerance: float = 0.0     # geometry mode: empirical floor added to every band (mm)
    island_gap: float = 3.5        # mm; islands of one colour farther apart are scheduled separately (0 = off)
    strategies: tuple = ("snake", "greedy")
    # Calibration: [[x_min, x_max, tolerance], ...] in bed X (mm). Units whose footprint centre lies
    # in a zone use that tip_tolerance instead of the global one (geometry mode, island units only).
    tolerance_zones: list | None = None

    def head(self) -> HeadModel:
        if self.mode == "fixed":
            return HeadModel.fixed(self.fixed_height)
        kw = {"min_clearance": self.tip_tolerance}
        if self.head_profile:
            kw["profile"] = [tuple(x) for x in self.head_profile]
        if self.head_far is not None:
            kw["far"] = self.head_far
        return HeadModel.geometry(**kw)


def batch_gcode(text: str, opts: Options) -> tuple[str, dict]:
    """Return (new G-code, stats).

    Each unit granularity (islands, and whole layer-per-colour runs) is scheduled with every
    strategy in opts.strategies; the order with the least flush volume wins. If none beats
    Orca's own layer order, the input is returned unchanged.
    """
    if opts.mode == "off":
        return text, {"mode": "off"}
    head = opts.head()
    zones = [tuple(map(float, z)) for z in (opts.tolerance_zones or [])] if opts.mode == "geometry" else []
    gaps = [opts.island_gap, 0.0] if opts.mode == "geometry" and opts.island_gap > 0 else [0.0]
    if zones:  # zones are per island: whole-layer units would straddle them
        gaps = [opts.island_gap if opts.island_gap > 0 else 3.5]
    best = None  # (flush, changes, label, parsed, emitter, order)
    candidates = {}
    base = None
    for gap in gaps:
        parsed = split(text, island_gap=gap)
        em = Emitter(parsed, head)
        unit_heads = None
        if zones:
            zone_head = {t: replace(opts, tip_tolerance=t).head() for _, _, t in zones}
            unit_heads = []
            for s in parsed.segments:
                x = float(cell_centres(s.cells)[:, 0].mean())
                t = next((t for lo, hi, t in zones if lo <= x <= hi), None)
                unit_heads.append(zone_head[t] if t is not None else head)
        sched = Scheduler(parsed.segments, head, em.flush, parsed.first_tool, unit_heads)
        if base is None:
            base = sched.cost(list(range(len(parsed.segments))))
        for s in opts.strategies:
            order = sched.run(s)
            changes, flush = sched.cost(order)
            label = f"{s}/{'islands' if gap > 0 else 'layer-units'}"
            candidates[label] = changes
            if best is None or (flush, changes) < best[:2]:
                best = (flush, changes, label, parsed, em, order)
    flush, changes, label, parsed, em, order = best
    stats = dict(mode=opts.mode, head=head.name + (f" + {len(zones)} tolerance zones" if zones else ""),
                 segments=len(parsed.segments),
                 source_tool_changes=len(parsed.blocks), strategy=label, candidates=candidates)
    # Calibration plates exist to show the per-zone height leads, even when the strictest zone
    # keeps the tool-change count at layer order.
    if flush >= base[1] and not zones:
        return text, dict(stats, strategy="unchanged (no improvement over layer order)",
                          tool_changes=len(parsed.blocks), synth_blocks=0, travel_lifts=0)
    out = em.emit(order)
    return out, dict(stats, **em.stats)
