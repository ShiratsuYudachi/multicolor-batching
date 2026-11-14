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


MODELS = {"pillars": pillars, "keychain": keychain, "figurine": figurine, "split_cube": split_cube}


if __name__ == "__main__":
    for name, fn in MODELS.items():
        out = ROOT / "work" / name / "in.3mf"
        write_3mf(out, fn())
        print(out)
