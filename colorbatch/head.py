"""Print-head clearance models.

A model answers: an already-printed obstacle whose nearest point is `d` mm (horizontal) from the
nozzle axis may stand at most `allow(d)` mm above the nozzle tip.

* fixed:    allow(d) = H everywhere (ITC "safe batch height").
* geometry: piecewise-constant profile [(r_i, h_i)]: for d < r_i (first matching) allow h_i;
            beyond the last radius allow `far` (gantry / X-rod clearance).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

# Bambu Lab A1 / A1 mini 0.4 mm hotend (same hotend on both). Derived from the community-measured
# hotend CAD "Bambulab A1 Hotend" (Printables 879230, A1_Hotend.step): for each band start r the
# value is the lowest point of the hotend at horizontal distance >= r from the nozzle axis
# (axisymmetric, i.e. worst direction). Tip land radius 0.52 mm; cone up to the silicone sock,
# whose flat bottom sits 1.9 mm above the tip out to r = 10.5 mm; sock chamfer to r = 12 mm.
# Beyond 12 mm the CAD has only the hotend, not the toolhead shell / part-cooling duct: 4.9 mm out
# to 30 mm is an ASSUMPTION (duct not lower than the sock edge) until measured on the printer.
# Past that, Orca's extruder_clearance_height_to_rod = 25 mm.
A1_PROFILE: list[tuple[float, float]] = [
    (0.6, 0.0),
    (0.75, 0.21),
    (1.0, 0.5),
    (1.25, 0.98),
    (1.5, 1.5),
    (10.5, 1.9),
    (11.0, 2.1),
    (11.5, 3.1),
    (12.0, 4.1),
    (30.0, 4.9),
]
A1_FAR = 25.0


@dataclass(frozen=True)
class HeadModel:
    radii: tuple[float, ...]
    heights: tuple[float, ...]
    far: float
    name: str = "custom"

    @staticmethod
    def fixed(h: float) -> "HeadModel":
        return HeadModel((), (), h, f"fixed {h:g} mm")

    @staticmethod
    def geometry(profile=A1_PROFILE, far=A1_FAR, name="A1 geometry", min_clearance: float = 0.0) -> "HeadModel":
        """min_clearance > 0 adds an empirical floor (ITC-style squish tolerance) to every band."""
        prof = sorted(profile)
        hs: list[float] = []
        for _, h in prof:  # clearance can never shrink further out
            h = max(h, min_clearance)
            hs.append(max(h, hs[-1]) if hs else h)
        if min_clearance > 0:
            name = f"{name} + {min_clearance:g} mm floor"
        return HeadModel(tuple(r for r, _ in prof), tuple(hs), max(far, hs[-1] if hs else far), name)

    def allow(self, d: float) -> float:
        i = bisect.bisect_right(self.radii, d)
        return self.heights[i] if i < len(self.radii) else self.far

    def min_distance(self, dz: float) -> float:
        """Smallest horizontal distance at which an obstacle dz above the tip is tolerated."""
        if dz <= 1e-9:
            return 0.0
        prev_r = 0.0
        for r, h in zip(self.radii, self.heights):
            if h + 1e-9 >= dz:
                return prev_r
            prev_r = r
        return prev_r if self.far + 1e-9 >= dz else math.inf

    @property
    def reach(self) -> float:
        """Beyond this distance only `far` matters."""
        return self.radii[-1] if self.radii else 0.0
