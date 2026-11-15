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
    """`far` is the clearance beyond the profile. With `rod_band` set it is the X-gantry clearance
    (the rod spans the whole X axis): it only applies to obstacles whose Y distance from the nozzle
    is within `rod_band`; farther in Y only the lid/frame clearance `lid` applies (Orca's sequential
    printing model: extruder_clearance_height_to_rod / _dist_to_rod / _height_to_lid)."""
    radii: tuple[float, ...]
    heights: tuple[float, ...]
    far: float
    name: str = "custom"
    rod_band: float | None = None
    lid: float | None = None

    @staticmethod
    def fixed(h: float) -> "HeadModel":
        return HeadModel((), (), h, f"fixed {h:g} mm")

    @staticmethod
    def geometry(profile=A1_PROFILE, far=A1_FAR, name="A1 geometry", min_clearance: float = 0.0,
                 rod_band: float | None = None, lid: float | None = None) -> "HeadModel":
        """min_clearance > 0 adds an empirical floor (ITC-style squish tolerance) to every band."""
        prof = sorted(profile)
        hs: list[float] = []
        for _, h in prof:  # clearance can never shrink further out
            h = max(h, min_clearance)
            hs.append(max(h, hs[-1]) if hs else h)
        if min_clearance > 0:
            name = f"{name} + {min_clearance:g} mm floor"
        far = max(far, hs[-1] if hs else far)
        if rod_band is not None:
            name += f", X rod {far:g} mm within ±{rod_band:g} mm in Y"
            lid = max(lid if lid is not None else far, far)
        return HeadModel(tuple(r for r, _ in prof), tuple(hs), far, name, rod_band, lid)

    @property
    def beyond(self) -> float:
        """Largest clearance anywhere beyond the profile (lid if the rod is Y-banded, else far)."""
        return self.lid if self.rod_band is not None and self.lid is not None else self.far

    def allow(self, d: float) -> float:
        """Clearance at horizontal distance d, assuming the obstacle may lie under the X rod."""
        i = bisect.bisect_right(self.radii, d)
        return self.heights[i] if i < len(self.radii) else self.far

    def min_distance(self, dz: float) -> float:
        """Smallest horizontal distance at which an obstacle dz above the tip is tolerated
        (ignoring the Y-band rule: under the rod)."""
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
