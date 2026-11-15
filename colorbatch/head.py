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

# Bambu Lab A1 / A1 mini 0.4 mm hotend + toolhead. Each band value is the lowest point of the head
# at horizontal distance >= band start from the nozzle axis (axisymmetric = worst direction), taking
# the lower of two independently measured community CAD models:
#  * hotend:   Printables 879230 "Bambulab A1 Hotend" -- tip land r 0.5 mm, cone, silicone sock
#              flat at 1.9 mm out to r 10.5 mm, chamfer to r 12 mm;
#  * printer:  MakerWorld 770920 "Bambulab A1 CAD step model" (measured on a 5/2024 A1, +-0.2 mm) --
#              heater block at 2.7 mm, toolhead shell / duct bottom flat at 5.4 mm reaching r 72 mm
#              (in one direction; 18-28 mm in the others), X gantry beam 27.4 mm above the tip.
# Beyond the shell, Orca's extruder_clearance_height_to_rod = 25 mm (below the CAD's 27.4 mm).
A1_PROFILE: list[tuple[float, float]] = [
    (0.6, 0.0),
    (0.75, 0.17),
    (1.0, 0.5),
    (1.25, 0.98),
    (1.5, 1.5),
    (10.5, 1.9),
    (11.0, 2.1),
    (11.5, 3.1),
    (12.0, 4.1),
    (12.25, 4.9),
    (72.0, 5.4),
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
