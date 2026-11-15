"""Metrics and an independent safety check on any (original or batched) G-code file."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .emit import HeightMap
from .gcode import CELL, MachineState, is_move, path_points, rasterise, read_config, words
from .head import HeadModel
from .schedule import densify

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


def check_underprint(text: str, res: float = 0.05, shrink: float = 0.1, limit: int = 20) -> list[str]:
    """Replay the file on a fine grid; report extrusions whose nozzle path runs under material that
    is already printed above it (e.g. a colour printed ahead over an unprinted one beneath).
    Lines are narrowed by `shrink` per side so side-by-side neighbours do not count."""
    lines = text.splitlines()
    body = next(i for i, ln in enumerate(lines) if ln.startswith("; CHANGE_LAYER"))
    end = max(i for i, ln in enumerate(lines) if ln.startswith("; FEATURE: Custom"))
    st = MachineState()
    for ln in lines[:body]:
        st.feed(ln)
    moves = []
    width = 0.45
    for i in range(body, end):
        ln = lines[i]
        if ln.startswith("; LINE_WIDTH:"):
            width = float(ln.split(":", 1)[1])
        x0, y0 = st.x, st.y
        _, ext = st.feed(ln)
        if ext:
            p = np.array(path_points(ln, x0, y0))
            moves.append((i, densify(np.column_stack([p, np.zeros(len(p))]), step=res)[:, :2], st.z, width / 2))
    if not moves:
        return []
    allp = np.vstack([m[1] for m in moves])
    lo = allp.min(0) - 1.0
    shape = np.ceil((allp.max(0) + 1.0 - lo) / res).astype(int) + 1
    top = np.zeros(shape, dtype=np.float32)
    disks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    issues = []
    for i, pts, z, hw in moves:
        c = np.round((pts - lo) / res).astype(int)
        above = top[c[:, 0], c[:, 1]] - z
        if (above > 1e-3).any():
            issues.append(f"line {i + 1}: z={z:.2f} under material at {z + float(above.max()):.2f}: {lines[i].strip()}")
            if len(issues) >= limit:
                break
        k = int(np.ceil(max(hw - shrink, 0.0) / res))
        if k not in disks:
            g = np.arange(-k, k + 1)
            dx, dy = np.meshgrid(g, g, indexing="ij")
            m = (dx * dx + dy * dy) * res * res <= max(hw - shrink, 0.0) ** 2
            disks[k] = (dx[m], dy[m])
        ox, oy = disks[k]
        np.maximum.at(top, ((c[:, None, 0] + ox).ravel(), (c[:, None, 1] + oy).ravel()), np.float32(z))
    return issues


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
