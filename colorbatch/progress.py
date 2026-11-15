"""Printer progress (M73 P/R) for re-ordered G-code.

Orca writes `M73 P<percent> R<minutes left>` for the original order; after batching those values
jump around. Here every line gets a duration from a simple feed-rate model, calibrated against
Orca's own estimate of the source file with two factors -- one for tool-change blocks, one for
everything else -- read from the source's M73 R markers. The output's M73 P/R lines are then
regenerated from those durations, and the header time comment is updated.
"""
from __future__ import annotations

import math
import re

from .gcode import MachineState, is_move, words

_M73 = re.compile(r"^M73 P(\d+) R(\d+)")
_HEADER = re.compile(r"^; model printing time: .*; total estimated time: .*$")


def line_seconds(lines: list[str]) -> tuple[list[float], list[bool]]:
    """Per-line duration (feed-rate model, dwell commands) and whether the line is in a tool change."""
    body = next((i for i, ln in enumerate(lines) if ln.startswith("; CHANGE_LAYER")), len(lines))
    st = MachineState()
    secs, in_block = [], []
    blk = False
    for i, ln in enumerate(lines):
        if ln.startswith("; filament end gcode") and i > body:
            blk = True
        t = 0.0
        x0, y0, z0 = st.x, st.y, st.z
        e, _ = st.feed(ln)
        if is_move(ln):
            if st.f > 0:
                t = max(math.dist((x0, y0, z0), (st.x, st.y, st.z)), abs(e)) / (st.f / 60.0)
        elif ln.startswith("M400 S"):
            t = words(ln).get("S", 0.0)
        elif ln.startswith("M400 P"):
            t = words(ln).get("P", 0.0) / 1000.0
        secs.append(t)
        in_block.append(blk)
        if ln.startswith("; filament start gcode"):
            blk = False
    return secs, in_block


def _calibration(lines: list[str]) -> tuple[float, float, float] | None:
    """(total Orca seconds, block factor, print factor) from the source's M73 R markers."""
    secs, in_block = line_seconds(lines)
    total_min = None
    orca_block = orca_other = own_block = own_other = 0.0
    last_r = None
    for ln, s, b in zip(lines, secs, in_block):
        m = _M73.match(ln)
        if m:
            r = int(m.group(2))
            if total_min is None:
                total_min = r
            if last_r is not None and r < last_r:
                # The minute that just elapsed is credited to the kind of line that closed it.
                if b:
                    orca_block += (last_r - r) * 60.0
                else:
                    orca_other += (last_r - r) * 60.0
            last_r = r
        if b:
            own_block += s
        else:
            own_other += s
    if total_min is None or own_other <= 0:
        return None
    if last_r:
        orca_other += last_r * 60.0  # minutes left after the last marker (end G-code)
    kb = orca_block / own_block if own_block > 0 else 1.0
    ko = orca_other / own_other
    return total_min * 60.0, kb, ko


def _fmt(sec: float) -> str:
    sec = int(round(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def retime(source: str, output: str) -> tuple[str, dict]:
    """Regenerate M73 P/R in `output` (a re-ordering of `source`) and update the header time."""
    src_lines = source.splitlines()
    cal = _calibration(src_lines)
    if cal is None:
        return output, {}
    orca_total, kb, ko = cal
    lines = [ln for ln in output.splitlines() if not _M73.match(ln)]
    secs, in_block = line_seconds(lines)
    dur = [s * (kb if b else ko) for s, b in zip(secs, in_block)]
    total = sum(dur)
    out, elapsed, last = [], 0.0, None
    for ln, d in zip(lines, dur):
        mark = (min(99, int(100 * elapsed / total)), int(math.ceil((total - elapsed) / 60.0)))
        if mark != last and (is_move(ln) or last is None):
            out.append(f"M73 P{mark[0]} R{mark[1]}")
            last = mark
        out.append(ln)
        elapsed += d
    for i, ln in enumerate(out[:200]):
        if _HEADER.match(ln):
            src_hdr = next((s for s in src_lines[:200] if _HEADER.match(s)), None)
            prep = 0.0
            if src_hdr:  # keep Orca's prepare time (total - model printing) as is
                prep = _hdr_seconds(src_hdr, "total estimated time") - _hdr_seconds(src_hdr, "model printing time")
            out[i] = f"; model printing time: {_fmt(total - prep)}; total estimated time: {_fmt(total)}"
            break
    return "\n".join(out) + "\n", {"orca_source_min": orca_total / 60.0, "estimated_min": total / 60.0,
                                    "block_factor": kb, "print_factor": ko}


def _hdr_seconds(hdr: str, key: str) -> float:
    m = re.search(re.escape(key) + r": ((?:\d+[dhms] ?)+)", hdr)
    if not m:
        return 0.0
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return float(sum(int(v) * mult[u] for v, u in re.findall(r"(\d+)([dhms])", m.group(1))))
