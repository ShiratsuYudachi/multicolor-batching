"""Supports off vs on (tree, support filament 0 = Orca's default): tool changes and time for the
real models, Orca layer order vs batching. Plates are auto-arranged by Orca (--arrange 1) where it
can, so support trees do not run into neighbouring parts; if arranging fails the plate is sliced as
placed. Ki-43 is the airframe alone (work/ki43plane): with its six small parts the supports run into
them."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from colorbatch import Options, batch_gcode  # noqa: E402
from colorbatch.report import check_collisions, check_underprint, metrics  # noqa: E402
from colorbatch.gcode import read_config  # noqa: E402
from compare_table import minutes  # noqa: E402
from slice import ORCA  # noqa: E402


def variant(name: str, support: bool) -> Path:
    src = ROOT / "work" / name / "in.3mf"
    d = ROOT / "work" / f"{name}_sup{int(support)}"
    d.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(src)
    cfg = json.loads(z.read("Metadata/project_settings.config"))
    cfg["enable_support"] = "1" if support else "0"
    diff = cfg.get("different_settings_to_system") or [""]
    keys = {k for k in diff[0].split(";") if k} | {"enable_support"}
    diff[0] = ";".join(sorted(keys))
    cfg["different_settings_to_system"] = diff
    with zipfile.ZipFile(d / "in.3mf", "w", zipfile.ZIP_DEFLATED) as out:
        for n in z.namelist():
            out.writestr(n, json.dumps(cfg, indent=1) if n == "Metadata/project_settings.config" else z.read(n))
    shutil.rmtree(d / "out", ignore_errors=True)
    out = d / "out" / "plate_1.gcode"
    for extra in (["--arrange", "1"], []):  # Orca's arrange can fail once supports widen the footprint
        r = subprocess.run([ORCA, *extra, "--slice", "1", "--outputdir", "out", "in.3mf"], cwd=d,
                           capture_output=True, text=True)
        if out.exists():
            break
    if not out.exists():
        raise RuntimeError(f"{d.name}: orca exit {r.returncode & 0xFFFFFFFF:#x}")
    return out


def main():
    print("| model | supports | support lines | Orca layer order | geometry | +0.2 | +0.4 | +0.8 |")
    print("|---|---|---|---|---|---|---|---|")
    for name in sys.argv[1:] or ("mario_a1", "ki43plane"):
        for support in (False, True):
            text = variant(name, support).read_text(encoding="utf-8")
            cfg = read_config(text.splitlines())
            cells = [f"{metrics(text)['tool_changes']} / {minutes(text):.0f} min"]
            for tol in (0.0, 0.2, 0.4, 0.8):
                out, _ = batch_gcode(text, Options(tip_tolerance=tol))
                bad = len(check_collisions(out, Options(tip_tolerance=tol).head(cfg))) + len(check_underprint(out))
                cells.append(f"{metrics(out)['tool_changes']} / {minutes(out):.0f} min" + (f" (!{bad})" if bad else ""))
            print(f"| {name} | {'on' if support else 'off'} | {text.count('; FEATURE: Support')} | "
                  + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
