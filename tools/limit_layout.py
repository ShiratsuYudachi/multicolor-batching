"""Squeeze the "准备打印的2" plate to the tightest layout the clearance rules still allow with
the minimum number of tool changes, to test the rules on the printer.

Rows in Y: white (T1) front, yellow (T2) middle, red (T3) back. Yellow is printed first and is
taller than the X-rod clearance, so the binding rules are the rod's Y band (yellow vs both rows)
and the toolhead-shell radius (yellow vs the nearest white/red part). Both gaps are binary-searched
for the smallest value that still yields the minimum change count.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colorbatch import Options, batch_gcode  # noqa: E402
from colorbatch.gcode import read_config  # noqa: E402
from colorbatch.report import check_collisions, check_underprint  # noqa: E402

SRC = Path(os.path.expanduser("~/Downloads/准备打印的2.3mf"))
# Footprints of the source plate (bed mm, from its sliced G-code) and the item each belongs to.
ORIG = {  # item id: (x0, x1, y0, y1)
    "8": (147, 227, 18, 44),   # white, large
    "4": (73, 108, 20, 39),    # white, small
    "6": (38, 62, 94, 114),    # yellow
    "2": (164, 219, 147, 202),  # red, large
    "10": (42, 69, 170, 196),  # red, small
}


def box_dist(a, b) -> float:
    dx = max(0.0, b[0] - a[1], a[0] - b[1])
    dy = max(0.0, b[2] - a[3], a[2] - b[3])
    return math.hypot(dx, dy)


def layout(gy: float, r: float) -> dict[str, tuple[float, float, float, float]]:
    """Target boxes: white row at y=2, yellow at the left, gy mm Y gap between rows, the small
    white/red parts slid towards yellow until their box distance to it is r."""
    def place(k, x0, y0):
        b = ORIG[k]
        return (x0, x0 + b[1] - b[0], y0, y0 + b[3] - b[2])
    out = {"8": place("8", 147, 2), "4": None, "6": None, "2": None, "10": None}
    white_y1 = max(2 + ORIG[k][3] - ORIG[k][2] for k in ("8", "4"))
    out["6"] = place("6", 2, white_y1 + gy)
    yel = out["6"]
    red_y0 = yel[3] + gy
    out["2"] = place("2", 164, red_y0)
    # Small parts: smallest x (closest to yellow) with box distance >= r.
    for k, y0 in (("4", 2), ("10", red_y0)):
        w = ORIG[k][1] - ORIG[k][0]
        dy = max(0.0, y0 - yel[3], yel[2] - (y0 + ORIG[k][3] - ORIG[k][2]))
        dx = math.sqrt(max(r * r - dy * dy, 0.0))
        out[k] = place(k, yel[1] + dx, y0)
        assert out[k][0] + w <= 256
    return out


def write_plate(boxes, path: Path) -> None:
    z = zipfile.ZipFile(SRC)
    model = z.read("3D/3dmodel.model").decode()

    def fix(m):
        oid, v = m.group(1), m.group(2).split()
        b, o = boxes[oid], ORIG[oid]
        v[9] = f"{float(v[9]) + b[0] - o[0]:.6f}"
        v[10] = f"{float(v[10]) + b[2] - o[2]:.6f}"
        return m.group(0).replace(m.group(2), " ".join(v))

    model = re.sub(r'<item objectid="(\d+)"[^>]*transform="([^"]+)"', fix, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for n in z.namelist():
            out.writestr(n, model if n == "3D/3dmodel.model" else z.read(n))


def evaluate(gy: float, r: float, name: str) -> tuple[int, str]:
    d = ROOT / "work" / name
    write_plate(layout(gy, r), d / "in.3mf")
    subprocess.run([sys.executable, str(ROOT / "tools" / "slice.py"), name], check=True, capture_output=True)
    text = (d / "out" / "plate_1.gcode").read_text(encoding="utf-8")
    out, stats = batch_gcode(text, Options())
    (d / "out" / "batched.gcode").write_text(out, encoding="utf-8", newline="\n")
    return stats["tool_changes"], out


def search(lo, hi, f, target, step=0.25):
    """Smallest x in [lo, hi] with f(x) <= target (f monotone non-increasing in x)."""
    while hi - lo > step:
        mid = round((lo + hi) / 2, 3)
        if f(mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    target, _ = evaluate(70, 90, "limit_probe")
    print("changes with generous gaps:", target, flush=True)
    gy = search(54, 70, lambda g: evaluate(g, 90, "limit_probe")[0], target)
    print("min Y gap:", gy, flush=True)
    r = search(60, 90, lambda rr: evaluate(gy, rr, "limit_probe")[0], target)
    print("min radial gap:", r, flush=True)
    changes, out = evaluate(gy, r, "limit_plate")
    text = (ROOT / "work" / "limit_plate" / "out" / "plate_1.gcode").read_text(encoding="utf-8")
    cfg = read_config(text.splitlines())
    print(json.dumps({"y_gap": gy, "radial_gap": r, "changes": changes,
                      "collisions": len(check_collisions(out, Options().head(cfg))),
                      "underprints": len(check_underprint(out)),
                      "boxes": layout(gy, r)}, indent=1))


if __name__ == "__main__":
    main()
