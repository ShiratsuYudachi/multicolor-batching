"""Cross-layer colour batching for single-nozzle multi-material G-code (Bambu A1 / AMS lite)."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .emit import Emitter
from .gcode import cell_centres, read_config, split
from .head import HeadModel
from .progress import retime
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
    # Print support-only parts with whichever filament is loaded (supports are removed anyway).
    # Only applied when every filament has the same filament_type (e.g. all PLA).
    support_any_filament: bool = True

    def head(self, printer_cfg: dict | None = None) -> HeadModel:
        """printer_cfg: the G-code's config block. In geometry mode its X-rod/lid clearances
        (extruder_clearance_height_to_rod / _dist_to_rod / _height_to_lid) model the gantry:
        the rod limit only applies to parts within dist_to_rod in Y of the nozzle."""
        if self.mode == "fixed":
            return HeadModel.fixed(self.fixed_height)
        kw = {"min_clearance": self.tip_tolerance}
        if self.head_profile:
            kw["profile"] = [tuple(x) for x in self.head_profile]
        cfg = printer_cfg or {}

        def num(key):
            try:
                return float(str(cfg[key]).split(",")[0])
            except (KeyError, ValueError):
                return None

        rod, band, lid = (num("extruder_clearance_height_to_rod"), num("extruder_clearance_dist_to_rod"),
                          num("extruder_clearance_height_to_lid"))
        if self.head_far is not None:
            kw["far"] = self.head_far
        elif rod:
            kw["far"] = rod
        if band and lid:
            kw.update(rod_band=band, lid=lid)
        return HeadModel.geometry(**kw)


def unsupported_reason(cfg: dict[str, str]) -> str | None:
    """Why this G-code cannot be batched (from its own config block), or None."""
    def truthy(key):
        return str(cfg.get(key, "")).strip().lower() in ("1", "true")
    if truthy("enable_prime_tower"):
        return "prime tower is enabled; batching needs it off"
    seq = str(cfg.get("print_sequence", "by layer")).strip().lower()
    if seq and seq != "by layer":
        return f"print sequence is '{seq}', batching needs 'by layer'"
    nozzles = [v for v in str(cfg.get("nozzle_diameter", "")).split(",") if v.strip()]
    if len(nozzles) > 1:
        return f"{len(nozzles)}-nozzle printer; batching supports single-nozzle multi-material only"
    if "M620 S" not in "".join(cfg.get("change_filament_gcode", "M620 S")):
        return "no AMS tool-change G-code"
    return None


def batch_gcode(text: str, opts: Options) -> tuple[str, dict]:
    """Return (new G-code, stats).

    Each unit granularity (islands, and whole layer-per-colour runs) is scheduled with every
    strategy in opts.strategies; the order with the least flush volume wins. If none beats
    Orca's own layer order, the input is returned unchanged. Unsupported files (see
    unsupported_reason) are returned unchanged with stats["skipped"] set.
    """
    if opts.mode == "off":
        return text, {"mode": "off"}
    cfg = read_config(text.splitlines())
    reason = unsupported_reason(cfg)
    if reason:
        return text, {"mode": opts.mode, "skipped": reason}
    head = opts.head(cfg)
    zones = [tuple(map(float, z)) for z in (opts.tolerance_zones or [])] if opts.mode == "geometry" else []
    gaps = [opts.island_gap, 0.0] if opts.mode == "geometry" and opts.island_gap > 0 else [0.0]
    if zones:  # zones are per island: whole-layer units would straddle them
        gaps = [opts.island_gap if opts.island_gap > 0 else 3.5]
    types = {t.strip() for t in str(cfg.get("filament_type", "")).split(";") if t.strip()}
    support_any = opts.support_any_filament and len(types) <= 1
    best = None  # (flush, changes, label, parsed, emitter, order, assign)
    candidates = {}
    base = None
    for gap in gaps:
        parsed = split(text, island_gap=gap)
        em = Emitter(parsed, head)
        unit_heads = None
        if zones:
            zone_head = {t: replace(opts, tip_tolerance=t).head(parsed.config) for _, _, t in zones}
            unit_heads = []
            for s in parsed.segments:
                x = float(cell_centres(s.cells)[:, 0].mean())
                t = next((t for lo, hi, t in zones if lo <= x <= hi), None)
                unit_heads.append(zone_head[t] if t is not None else head)
        sched = Scheduler(parsed.segments, head, em.flush, parsed.first_tool, unit_heads, support_any)
        if base is None:
            base = sched.cost(list(range(len(parsed.segments))))
        for s in opts.strategies:
            order = sched.run(s)
            assign = dict(sched.last_assign)
            changes, flush = sched.cost(order, assign)
            label = f"{s}/{'islands' if gap > 0 else 'layer-units'}"
            candidates[label] = changes
            if best is None or (flush, changes) < best[:2]:
                best = (flush, changes, label, parsed, em, order, assign)
    flush, changes, label, parsed, em, order, assign = best
    stats = dict(mode=opts.mode, head=head.name + (f" + {len(zones)} tolerance zones" if zones else ""),
                 segments=len(parsed.segments),
                 source_tool_changes=len(parsed.blocks), strategy=label, candidates=candidates)
    # Calibration plates exist to show the per-zone height leads, even when the strictest zone
    # keeps the tool-change count at layer order.
    if flush >= base[1] and not zones:
        return text, dict(stats, strategy="unchanged (no improvement over layer order)",
                          tool_changes=len(parsed.blocks), synth_blocks=0, travel_lifts=0)
    out, timing = retime(text, em.emit(order, assign))
    reassigned = sum(1 for u, t in assign.items() if t != parsed.segments[u].tool)
    return out, dict(stats, **em.stats, **timing, support_units_recoloured=reassigned)
