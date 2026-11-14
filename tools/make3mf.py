"""Build Bambu/Orca project 3MFs for the batching test-suite.

Each test model is a list of (trimesh mesh, filament index 1-based). Printer/process/filament
settings are resolved from the OrcaSlicer profile tree (inherits chain) and embedded as
Metadata/project_settings.config, so `orca-slicer --slice 0` works headless.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT.parent / "OrcaSlicer-batching" / "resources" / "profiles" / "BBL"
BASE_3MF = ROOT / "work" / "bullet" / "in.3mf"  # supplies a complete key set for project_settings

PRINTERS = {
    "A1": ("Bambu Lab A1 0.4 nozzle", "0.20mm Standard @BBL A1", "Bambu PLA Basic @BBL A1"),
    "A1M": ("Bambu Lab A1 mini 0.4 nozzle", "0.20mm Standard @BBL A1M", "Bambu PLA Basic @BBL A1M"),
}
COLOURS = ["#FFFFFF", "#000000", "#E02020", "#2060E0", "#20A040", "#F0C000", "#A040C0", "#808080"]


def _index(kind: str) -> dict[str, dict]:
    out = {}
    for p in (PROFILES / kind).glob("*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "name" in j:
            out[j["name"]] = j
    return out


def resolve(kind: str, name: str) -> dict:
    idx = _index(kind)
    chain = []
    while name:
        j = idx[name]
        chain.append(j)
        name = j.get("inherits", "")
    merged: dict = {}
    for j in reversed(chain):
        merged.update(j)
    for k in ("inherits", "instantiation", "type", "from", "setting_id", "filament_id"):
        merged.pop(k, None)
    return merged


def project_settings(printer: str, n_filaments: int, overrides: dict | None = None) -> dict:
    mname, pname, fname = PRINTERS[printer]
    base = json.loads(zipfile.ZipFile(BASE_3MF).read("Metadata/project_settings.config"))
    machine, process, filament = resolve("machine", mname), resolve("process", pname), resolve("filament", fname)
    cfg = dict(base)
    cfg.update({k: v for k, v in machine.items() if k != "name"})
    cfg.update({k: v for k, v in process.items() if k != "name"})
    # Filament keys are per-filament vectors: replicate the single resolved preset.
    for k, v in filament.items():
        if k == "name":
            continue
        vals = v if isinstance(v, list) else [v]
        cfg[k] = [vals[0]] * n_filaments
    cfg["print_compatible_printers"] = [mname]
    cfg.pop("compatible_printers", None)
    cfg.pop("compatible_printers_condition", None)
    cfg["printer_settings_id"] = mname
    cfg["print_settings_id"] = pname
    cfg["filament_settings_id"] = [fname] * n_filaments
    cfg["filament_colour"] = COLOURS[:n_filaments]
    # Per-filament vectors in the base project sized for its own filament count: resize them.
    base_n = len(base.get("filament_colour", [])) or 1
    for k, v in list(cfg.items()):
        if isinstance(v, list) and len(v) == base_n and k in base and k not in machine and k not in process:
            cfg[k] = [v[0]] * n_filaments
    # Flush matrix: n*n, symmetric default from the flush_volumes_vector heuristic (Orca default 280 mm3 off-diagonal).
    cfg["flush_volumes_matrix"] = [("0" if i == j else "280") for i in range(n_filaments) for j in range(n_filaments)]
    cfg["flush_volumes_vector"] = ["140"] * (2 * n_filaments)
    cfg["enable_prime_tower"] = "0"
    cfg["print_sequence"] = "by layer"
    if overrides:
        cfg.update(overrides)
    return cfg


def _mesh_xml(mesh: trimesh.Trimesh) -> str:
    v = "\n".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in mesh.vertices)
    t = "\n".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    return f"<mesh><vertices>{v}</vertices><triangles>{t}</triangles></mesh>"


def write_3mf(path: Path, parts: list[tuple[trimesh.Trimesh, int]], printer: str = "A1",
              n_filaments: int | None = None, overrides: dict | None = None, centre=(128.0, 128.0)) -> None:
    n_filaments = n_filaments or max(e for _, e in parts)
    # Centre the whole assembly on the bed.
    allb = np.vstack([m.bounds for m, _ in parts])
    off = np.array([centre[0] - (allb[:, 0].min() + allb[:, 0].max()) / 2,
                    centre[1] - (allb[:, 1].min() + allb[:, 1].max()) / 2, -allb[:, 2].min()])
    # One object whose parts (volumes) carry their own filament, so parts may stack and touch.
    objs, comps, parts_cfg = [], [], []
    for i, (m, ext) in enumerate(parts):
        oid = i + 1
        m = m.copy()
        m.apply_translation(off)
        objs.append(f'<object id="{oid}" type="model">{_mesh_xml(m)}</object>')
        comps.append(f'<component objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>')
        parts_cfg.append(
            f'<part id="{oid}" subtype="normal_part"><metadata key="name" value="part{oid}"/>'
            f'<metadata key="extruder" value="{ext}"/>'
            f'<metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/></part>')
    aid = len(parts) + 1
    objs.append(f'<object id="{aid}" type="model"><components>{"".join(comps)}</components></object>')
    items = [f'<item objectid="{aid}" transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/>']
    settings = [f'<object id="{aid}"><metadata key="name" value="{path.parent.name}"/>'
                f'<metadata key="extruder" value="{parts[0][1]}"/>{"".join(parts_cfg)}</object>']
    inst = [f'<model_instance><metadata key="object_id" value="{aid}"/>'
            f'<metadata key="instance_id" value="0"/><metadata key="identify_id" value="100"/></model_instance>']
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">'
             '<metadata name="Application">BambuStudio-01.10.00.00</metadata>'
             '<metadata name="BambuStudio:3mfVersion">1</metadata>'
             f'<resources>{"".join(objs)}</resources><build>{"".join(items)}</build></model>')
    model_settings = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>' + "".join(settings) +
                      '<plate><metadata key="plater_id" value="1"/><metadata key="plater_name" value=""/>'
                      '<metadata key="locked" value="false"/>' + "".join(inst) + '</plate></config>')
    ctypes = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
              '<Default Extension="png" ContentType="image/png"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ctypes)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/model_settings.config", model_settings)
        z.writestr("Metadata/project_settings.config",
                   json.dumps(project_settings(printer, n_filaments, overrides), indent=1))
