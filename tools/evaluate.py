"""Batch every sliced test model in each mode; print a comparison table and safety checks."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colorbatch import Options, batch_gcode  # noqa: E402
from colorbatch.report import check_collisions, check_underprint, metrics  # noqa: E402

MODES = {
    "fixed0.4": Options(mode="fixed", fixed_height=0.4),
    "geometry": Options(mode="geometry"),
    "geometry+0.4floor": Options(mode="geometry", tip_tolerance=0.4),
}


def main(names: list[str]) -> None:
    rows = []
    for name in names:
        src = ROOT / "work" / name / "out" / "plate_1.gcode"
        text = src.read_text(encoding="utf-8")
        base = metrics(text)
        rows.append((name, "layer (Orca)", base, "-", 0.0, ""))
        for mname, opts in MODES.items():
            t0 = time.time()
            out, stats = batch_gcode(text, opts)
            dt = time.time() - t0
            (src.parent / f"{mname}.gcode").write_text(out, encoding="utf-8", newline="\n")
            m = metrics(out)
            coll = check_collisions(out, opts.head()) + check_underprint(out)
            same = m["model_mm3_by_tool"] == base["model_mm3_by_tool"]
            note = ("" if same else "MODEL VOLUME MISMATCH ") + f"{stats['strategy']} {stats.get('candidates')} " \
                + f"units={stats['segments']} " \
                + (f"synth={stats['synth_blocks']} " if stats["synth_blocks"] else "") \
                + (f"lifts={stats['travel_lifts']}" if stats["travel_lifts"] else "")
            rows.append((name, mname, m, str(len(coll)), dt, note))
    print("| model | order | tool changes | purge cm³ | model cm³ | purge/model | est. min | collisions+underprints | run s | notes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, mode, m, coll, dt, note in rows:
        print(f"| {name} | {mode} | {m['tool_changes']} | {m['purge_mm3'] / 1000:.1f} | {m['model_mm3'] / 1000:.2f} | "
              f"{m['purge_mm3'] / max(m['model_mm3'], 1):.1f} | {m['est_time_min']:.0f} | {coll} | {dt:.1f} | {note} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pillars", "keychain", "figurine", "split_cube"])
