"""Before/after GIF of the print order: Orca's G-code (left) and the batched G-code (right) build up
side by side on a common clock (the printer's M73 time estimate), drawn in an isometric view in the filament colours.

    python tools/make_gif.py work/sq_corners/out/plate_1.gcode docs/corners.gif --title "Four pillars"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colorbatch import Options, batch_gcode  # noqa: E402
from colorbatch.gcode import MachineState, is_move, path_points, read_config  # noqa: E402

_TOOL = re.compile(r"^M620 S(\d+)A")
_M73 = re.compile(r"^M73 P(\d+) R(\d+)")


def trace(text: str):
    """Extrusion segments in print order: (N, 2, 3) xyz, filament per segment, elapsed minutes
    (from the printer's M73 progress), tool changes so far per segment."""
    st = MachineState()
    tool, changes = None, 0
    segs, tools, vol, chg = [], [], [], []
    elapsed, total_min = 0.0, None
    for ln in text.splitlines():
        m = _M73.match(ln)
        if m:
            p, r = int(m.group(1)), int(m.group(2))
            if total_min is None:
                total_min = float(r)
            elapsed = total_min - r
            continue
        m = _TOOL.match(ln)
        if m:
            t = int(m.group(1))
            if tool is not None and t != tool:
                changes += 1
            tool = t
            continue
        if not is_move(ln):
            st.feed(ln)
            continue
        x0, y0, z = st.x, st.y, st.z
        e, ext = st.feed(ln)
        if not ext:
            continue
        pts = path_points(ln, x0, y0)
        if not all(0 <= px <= 256 and 0 <= py <= 256 for px, py in pts):
            continue  # purge into the chute / wipe off the bed
        prev = (x0, y0)
        for p in pts:
            segs.append(((prev[0], prev[1], z), (p[0], p[1], z)))
            prev = p
            tools.append(tool)
            chg.append(changes)
        vol.extend([elapsed] * len(pts))
    return np.array(segs), np.array(tools), np.array(vol), np.array(chg)


def project(xyz, zscale):
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([(x - y) * c, (x + y) * s + z * zscale], axis=-1)


def render(before: str, after: str, out: Path, title: str, frames: int, zscale: float):
    cfg = read_config(before.splitlines())
    colours = [c.strip()[:7] for c in str(cfg.get("filament_colour", "")).split(";")]
    used = sorted({int(t) for t in trace(before)[1] if t is not None})
    if len({colours[t].upper() for t in used if t < len(colours)}) < len(used):  # test models: all white
        palette = ["#e8a33d", "#3b6fd8", "#d64545", "#4caf50", "#8e5bd0", "#222222", "#7fd3e8", "#f2d13a"]
        colours = palette * 2
    data = [trace(before), trace(after)]
    allp = np.concatenate([project(d[0], zscale).reshape(-1, 2) for d in data])
    lo, hi = allp.min(0), allp.max(0)
    pad = (hi - lo) * 0.04
    total = max(d[2][-1] for d in data)
    images = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), dpi=80)
    steps = list(np.linspace(0, 1, frames)) + [1.0] * (frames // 6)  # hold the last frame
    labels = ["Orca (layer by layer)", "colour batching"]
    for f in steps:
        for ax, (segs, tools, vol, chg), lab in zip(axes, data, labels):
            ax.clear()
            n = int(np.searchsorted(vol, f * total, side="right"))
            if n:
                cols = [colours[t] if t is not None and t < len(colours) else "#888888" for t in tools[:n]]
                ax.add_collection(LineCollection(project(segs[:n], zscale), colors=cols, linewidths=0.6))
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{lab}\nfilament changes: {chg[n - 1] if n else 0}   "
                         f"{int(min(f * total, vol[-1]) // 60)}h{int(min(f * total, vol[-1]) % 60):02d}m"
                         + ("  done" if n == len(vol) else ""), fontsize=12)
        fig.suptitle(title, fontsize=13)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert(
            "P", palette=Image.ADAPTIVE, colors=64))
    plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=90, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gcode")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--zscale", type=float, default=1.0, help="vertical exaggeration")
    a = ap.parse_args()
    before = Path(a.gcode).read_text(encoding="utf-8")
    after, stats = batch_gcode(before, Options())
    print(stats.get("source_tool_changes"), "->", stats.get("tool_changes"))
    render(before, after, Path(a.out), a.title, a.frames, a.zscale)


if __name__ == "__main__":
    main()
