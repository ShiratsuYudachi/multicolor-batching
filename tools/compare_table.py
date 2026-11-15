"""Tool changes and print time for a fixed set of plates: Orca layer order vs batching modes.

Time is Orca's own estimate for the source and the Orca-calibrated M73 estimate (progress.py)
for batched outputs.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from colorbatch import Options, batch_gcode  # noqa: E402
from colorbatch.gcode import read_config  # noqa: E402
from colorbatch.report import check_collisions, check_underprint, metrics  # noqa: E402
from make3mf import project_settings, write_3mf  # noqa: E402
from models import slope_box, square_pillars  # noqa: E402
from slice import slice_model  # noqa: E402

MARIO = Path.home() / "Downloads" / "mario+bros+3mf+MegaLab3D" / "mario figure 3d mega tha.3mf"
_TIME = re.compile(r"total estimated time: ((?:\d+[dhms] ?)+)")


def minutes(text: str) -> float:
    m = _TIME.search(text[:20000])
    mult = {"d": 1440, "h": 60, "m": 1, "s": 1 / 60}
    return sum(int(v) * mult[u] for v, u in re.findall(r"(\d+)([dhms])", m.group(1))) if m else float("nan")


def a1_copy(src: Path, name: str) -> None:
    """Re-target a downloaded project at the A1 (prime tower off), keeping its filament colours,
    with plate 1's object(s) moved to the A1 bed centre; slices plate 1 only."""
    z = zipfile.ZipFile(src)
    old = json.loads(z.read("Metadata/project_settings.config"))
    cols = old.get("filament_colour") or []
    cfg = project_settings("A1", len(cols))
    cfg["filament_colour"] = cols
    settings = z.read("Metadata/model_settings.config").decode()
    plate1 = next(p for p in re.findall(r"<plate>(.*?)</plate>", settings, re.S) if 'plater_id" value="1"' in p)
    ids = set(re.findall(r'key="object_id" value="(\d+)"', plate1))

    def centre(m):
        if m.group(1) not in ids:
            return m.group(0)
        v = m.group(2).split()
        v[9], v[10] = "128", "128"
        return m.group(0).replace(m.group(2), " ".join(v))

    model = re.sub(r'<item objectid="(\d+)"[^>]*transform="([^"]+)"', centre, z.read("3D/3dmodel.model").decode())
    dst = ROOT / "work" / name / "in.3mf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as out:
        for n in z.namelist():
            if n not in ("Metadata/project_settings.config", "Metadata/slice_info.config"):
                out.writestr(n, model if n == "3D/3dmodel.model" else z.read(n))
        out.writestr("Metadata/project_settings.config", json.dumps(cfg, indent=1))
    import subprocess
    from slice import ORCA
    subprocess.run([ORCA, "--slice", "1", "--outputdir", "out", "in.3mf"], cwd=dst.parent, check=True,
                   capture_output=True)


PLATES = {
    "4 pillars, touching": "sq_row_gap0",
    "4 pillars, 30 mm apart": "sq_row_gap30",
    "4 pillars, plate corners": "sq_corners",
    "sloped box (Color Batching demo)": "slope_box",
    "Mario (7 colours, A1)": "mario_a1",
    "Ki-43 (4 colours, painted)": "ki43",
}
MODES = {"geometry": 0.0, "+0.2": 0.2, "+0.4": 0.4, "+0.8": 0.8}


def prepare() -> None:
    write_3mf(ROOT / "work" / "sq_row_gap30" / "in.3mf", square_pillars(30))
    write_3mf(ROOT / "work" / "sq_corners" / "in.3mf", square_pillars(256 - 8 - 2 * 6 - 8, grid=True))
    write_3mf(ROOT / "work" / "slope_box" / "in.3mf", slope_box())
    a1_copy(MARIO, "mario_a1")
    for name in ("sq_row_gap30", "sq_corners", "slope_box"):
        slice_model(name)


def main() -> None:
    prepare()
    print("| plate | Orca layer order | geometry | +0.2 | +0.4 | +0.8 |")
    print("|---|---|---|---|---|---|")
    for label, name in PLATES.items():
        text = (ROOT / "work" / name / "out" / "plate_1.gcode").read_text(encoding="utf-8")
        cfg = read_config(text.splitlines())
        cells = [f"{metrics(text)['tool_changes']} / {minutes(text):.0f} min"]
        for tol in MODES.values():
            out, stats = batch_gcode(text, Options(tip_tolerance=tol))
            bad = len(check_collisions(out, Options(tip_tolerance=tol).head(cfg))) + len(check_underprint(out))
            cells.append(f"{metrics(out)['tool_changes']} / {minutes(out):.0f} min" + (f" (!{bad})" if bad else ""))
        print(f"| {label} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
