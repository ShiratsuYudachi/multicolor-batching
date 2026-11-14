"""Headless Orca slice of work/<model>/in.3mf -> work/<model>/out/plate_1.gcode."""
import subprocess
import sys
from pathlib import Path

ORCA = r"C:\Program Files\OrcaSlicer\orca-slicer.exe"
WORK = Path(__file__).resolve().parents[1] / "work"


def slice_model(name: str) -> Path:
    d = WORK / name
    r = subprocess.run([ORCA, "--slice", "0", "--outputdir", "out", "in.3mf"], cwd=d, capture_output=True, text=True)
    out = d / "out" / "plate_1.gcode"
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"{name}: orca exit {r.returncode & 0xFFFFFFFF:#x} {r.stdout[-300:]}")
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or [p.name for p in WORK.iterdir() if (p / "in.3mf").exists()]:
        try:
            print(n, slice_model(n))
        except RuntimeError as e:
            print(e)
