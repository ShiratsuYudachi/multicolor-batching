"""Synthetic test-suite: small, tall, many-colours-per-layer prints where flush dominates."""
from __future__ import annotations

import trimesh

from make3mf import write_3mf, ROOT


def box(x, y, z, cx=0.0, cy=0.0, z0=0.0):
    m = trimesh.creation.box((x, y, z))
    m.apply_translation((cx, cy, z0 + z / 2))
    return m


def cyl(d, h, cx=0.0, cy=0.0, z0=0.0):
    m = trimesh.creation.cylinder(radius=d / 2, height=h, sections=48)
    m.apply_translation((cx, cy, z0 + h / 2))
    return m


def pillars():
    """4 separate 8 mm pillars, 20 mm tall, 16 mm pitch, 4 colours: best case for batching."""
    return [(cyl(8, 20, cx=16 * i), i + 1) for i in range(4)]


def keychain():
    """Base plate in colour 1, 1 mm raised bars in colours 2/3 on top: interleaved only on top layers."""
    parts = [(box(50, 20, 2), 1)]
    for i in range(6):
        parts.append((box(3, 14, 1.0, cx=-20 + 8 * i, z0=2.0), 2 + i % 2))
    return parts


def figurine():
    """Tall figure: body column colour 1 with two side 'arms' (colour 2) 5 mm away and a 'badge' touching the body (colour 3)."""
    return [
        (cyl(14, 30), 1),
        (box(4, 4, 20, cx=-12, z0=0), 2),
        (box(4, 4, 20, cx=12, z0=0), 2),
        (box(2, 6, 12, cx=0, cy=-8, z0=8), 3),  # touches the body: tip-clearance bound
    ]


def split_cube():
    """Two halves of a 20 mm cube in two colours, touching: only ITC-style tiny batches possible."""
    return [(box(10, 20, 20, cx=-5), 1), (box(10, 20, 20, cx=5), 2)]


def square_pillars(gap: float, grid: bool = False, size: float = 8.0, height: float = 20.0):
    """4 square pillars (one colour each) with `gap` mm between faces, in a row or a 2x2 grid."""
    pitch = size + gap
    pos = [((i % 2) * pitch, (i // 2) * pitch) for i in range(4)] if grid else [(i * pitch, 0.0) for i in range(4)]
    return [(box(size, size, height, cx=x, cy=y), i + 1) for i, (x, y) in enumerate(pos)]


def mixed_gaps():
    """One row, three different gaps between neighbours (0.5, 3, 12 mm): all regimes on one plate."""
    xs, x = [], 0.0
    for g in (0.0, 0.5, 3.0, 12.0):
        x += g + (8.0 if xs else 0.0)
        xs.append(x)
    return [(box(8, 8, 20, cx=cx), i + 1) for i, cx in enumerate(xs)]


def _flower(r0: float, h: float, top_scale: float, petals=6, amp=0.15, n=180):
    import numpy as np
    from shapely.geometry import Polygon
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 * (1 + amp * np.sin(petals * t))
    m = trimesh.creation.extrude_polygon(Polygon(np.c_[r * np.cos(t), r * np.sin(t)]), h)
    v = m.vertices.copy()
    v[:, :2] *= (1 - (1 - top_scale) * v[:, 2:3] / h)  # taper towards the top
    m.vertices = v
    return m


def cb_demo():
    """Approximation of the Color Batching README test: a 20x20x10 box split diagonally into two
    colours, and a tapered flower-outline mound with a colour-2 skin over a colour-1 core."""
    import numpy as np
    blk = box(20, 20, 10, cx=-14)
    n = np.array([-10.0, 0.0, 20.0])  # plane through (x=-24,z=0)..(x=-4,z=10): wedge below it
    o = np.array([-24.0, 0.0, 0.0])
    upper = blk.slice_plane(o, n, cap=True)
    lower = blk.slice_plane(o, -n, cap=True)
    outer = _flower(9.0, 10.0, 0.6)
    core = _flower(7.2, 10.0, 0.6)
    skin = outer.difference(core)
    for m in (skin, core):
        m.apply_translation((14.0, 0.0, 0.0))
    return [(upper, 1), (lower, 2), (core, 1), (skin, 2)]


MODELS = {"pillars": pillars, "keychain": keychain, "figurine": figurine, "split_cube": split_cube,
          "cb_demo": cb_demo, "sq_mixed_gaps": mixed_gaps, "sq_grid_gap3": lambda: square_pillars(3, grid=True),
          "sq_grid_gap12": lambda: square_pillars(12, grid=True)}
MODELS.update({f"sq_row_gap{g:g}": (lambda g=g: square_pillars(g)) for g in (0, 1, 3, 6, 12, 25)})


TOL_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8)
TOL_PITCH = 44.0  # 24 mm pair + 20 mm gap


def tol_test():
    """Tolerance calibration plate: five pairs of touching 12x12x10 mm blocks (colour 1 | colour 2),
    left to right for tolerances TOL_VALUES, one tolerance zone per pair (see tol_zones)."""
    parts = []
    for i in range(len(TOL_VALUES)):
        x = i * TOL_PITCH
        parts += [(box(12, 12, 10, cx=x - 6), 1), (box(12, 12, 10, cx=x + 6), 2)]
    return parts


def tol_zones(bed_centre_x: float = 128.0) -> list[list[float]]:
    """Bed-X zones for tol_test as placed by write_3mf (assembly centred on the bed)."""
    shift = bed_centre_x - (len(TOL_VALUES) - 1) * TOL_PITCH / 2
    return [[round(shift + i * TOL_PITCH - TOL_PITCH / 2, 1), round(shift + i * TOL_PITCH + TOL_PITCH / 2, 1), t]
            for i, t in enumerate(TOL_VALUES)]


MODELS["tol_test"] = tol_test


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or MODELS:
        out = ROOT / "work" / name / "in.3mf"
        write_3mf(out, MODELS[name]())
        print(out)
