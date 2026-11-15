"""CLI: python -m colorbatch in.gcode -o out.gcode [--mode geometry|fixed] [--height 0.4] [--check]"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import Options, batch_gcode
from .report import check_collisions, metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="colorbatch")
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--mode", choices=["off", "fixed", "geometry"], default="geometry")
    ap.add_argument("--height", type=float, default=0.4, help="fixed mode batch height (mm)")
    ap.add_argument("--profile", help='geometry mode head profile JSON, e.g. "[[0.5,0],[2,0.6],[14,3.5]]"')
    ap.add_argument("--far", type=float, help="clearance beyond the profile (mm)")
    ap.add_argument("--tip-tolerance", type=float, default=0.0,
                    help="geometry mode: empirical height floor added to every band (mm)")
    ap.add_argument("--no-islands", action="store_true", help="one unit per layer and colour")
    ap.add_argument("--island-gap", type=float, default=3.5, help="geometry mode: island clustering distance (mm)")
    ap.add_argument("--tolerance-zones", help='calibration: JSON [[x_min, x_max, tolerance], ...] in bed X (mm)')
    ap.add_argument("--check", action="store_true", help="replay output and report head collisions")
    a = ap.parse_args(argv)
    text = open(a.input, encoding="utf-8").read()
    opts = Options(mode=a.mode, fixed_height=a.height,
                   head_profile=json.loads(a.profile) if a.profile else None, head_far=a.far,
                   tip_tolerance=a.tip_tolerance, island_gap=0.0 if a.no_islands else a.island_gap,
                   tolerance_zones=json.loads(a.tolerance_zones) if a.tolerance_zones else None)
    t0 = time.time()
    out, stats = batch_gcode(text, opts)
    stats["seconds"] = round(time.time() - t0, 2)
    before, after = metrics(text), metrics(out)
    report = {"stats": stats, "before": before, "after": after}
    if a.check:
        report["collisions"] = check_collisions(out, opts.head())
    if a.output:
        with open(a.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
    json.dump(report, sys.stdout, indent=1, default=float)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
