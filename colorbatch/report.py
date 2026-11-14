"""Metrics and an independent safety check on any (original or batched) G-code file."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .emit import HeightMap
from .gcode import CELL, MachineState, is_move, path_points, rasterise, read_config, words
from .head import HeadModel

_TC = re.compile(r"^M620 S(\d+)A")


def metrics(text: str) -> dict:
    lines = text.splitlines()
    cfg = read_config(lines)
    dia = float(cfg.get("filament_diameter", "1.75").split(",")[0])
    area = math.pi * dia * dia / 4
    body = next(i for i, ln in enumerate(lines) if ln.startswith("; CHANGE_LAYER"))
    st = MachineState()
    in_tc = False
    model_e: Counter = Counter()
    purge_e = 0.0
    changes = 0
    tool = None
    t_sec = 0.0
    for i, ln in enumerate(lines):
        m = _TC.match(ln)
        if m:
            tool = int(m.group(1))
            if i > body:
                changes += 1
        if ln.startswith("; filament end gcode") and i > body:
            in_tc = True
        elif ln.startswith("; filament start gcode"):
            in_tc = False
        x0, y0, z0 = st.x, st.y, st.z
        e, ext = st.feed(ln)
        if is_move(ln):
            dist = math.dist((x0, y0, z0), (st.x, st.y, st.z))
            if st.f > 0:
                t_sec += max(dist, abs(e)) / (st.f / 60.0)
            if in_tc and e > 0:
                purge_e += e
            elif ext and tool is not None:
                model_e[tool] += e
        elif ln.startswith("M400 S"):
            t_sec += words(ln).get("S", 0.0)
        elif ln.startswith("M400 P"):
            t_sec += words(ln).get("P", 0.0) / 1000.0
    return {
        "tool_changes": changes,
        "purge_mm3": purge_e * area,
        "model_mm3": sum(model_e.values()) * area,
        "model_mm3_by_tool": {k: round(v * area, 1) for k, v in sorted(model_e.items())},
        "est_time_min": t_sec / 60.0,
    }


def check_collisions(text: str, head: HeadModel, tol: float = 0.05, limit: int = 20) -> list[str]:
    """Replay the file; report moves where printed material is closer to the head than `head` allows."""
    lines = text.splitlines()
    body = next(i for i, ln in enumerate(lines) if ln.startswith("; CHANGE_LAYER"))
    end = max(i for i, ln in enumerate(lines) if ln.startswith("; FEATURE: Custom"))
    st = MachineState()
    for ln in lines[:body]:
        st.feed(ln)
    hm = HeightMap()
    issues = []
    width = 0.45
    pending: list[tuple[float, float, float]] = []
    pending_z = None

    def flush():
        nonlocal pending, pending_z
        if pending:
            hm.add(rasterise(pending), pending_z)
        pending, pending_z = [], None

    for i in range(body, end):
        ln = lines[i]
        if ln.startswith("; LINE_WIDTH:"):
            width = float(ln.split(":", 1)[1])
        if not is_move(ln):
            continue
        x0, y0, z0 = st.x, st.y, st.z
        e, ext = st.feed(ln)
        w = words(ln)
        if not ("X" in w or "Y" in w):
            continue
        pts = np.array(path_points(ln, x0, y0, sample=False))
        zmove = min(z0, st.z)
        if pending_z is not None and abs(pending_z - st.z) > 1e-6:
            flush()
        need = hm.required_z(pts, zmove, head)
        if need > zmove + tol:
            issues.append(f"line {i + 1}: z={zmove:.2f} needs >= {need:.2f}: {ln.strip()}")
            if len(issues) >= limit:
                break
        if ext:
            pending.extend((px, py, width / 2) for px, py in path_points(ln, x0, y0))
            pending_z = st.z
            if len(pending) > 4000:
                flush()
    return issues
