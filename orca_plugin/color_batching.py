# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "shapely"]
#
# [tool.orcaslicer.plugin]
# name = "Color Batching"
# description = "跨层换色批处理：在喷头不碰撞已打部分的前提下，让同一颜色连续打多层再换料，减少换料次数和冲刷废料。适用于拓竹 A1 / A1 mini（单喷头 + AMS lite），需关闭擦料塔、逐层打印。"
# author = "Nico"
# version = "0.1.0"
# ///
"""Cross-layer colour batching as an Orca slicing-pipeline plugin (G-code stage).

Enable per process preset: Process > Others > Slicing Pipeline Plugin -> "Color Batching".
Config (Plugins > Config tab, or per-preset override):
  {"mode": "geometry" | "fixed" | "off",
   "fixed_height": 0.4,                     # fixed mode: allowed height difference (mm)
   "head_profile": [[0.5, 0.0], [1.0, 0.2], ...] | null,   # geometry mode: (radius, height) steps
   "head_far": 25.0 | null,                 # clearance beyond the profile (X-rod)
   "tip_tolerance": 0.0,                    # geometry mode: height floor for every band (mm)
   "island_gap": 3.5,                       # mm; colour islands farther apart are scheduled separately (0 = off)
   "tolerance_zones": null}                 # calibration: [[x_min, x_max, tolerance], ...] in bed X (mm)
"""
import json
import os
import sys

import orca

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colorbatch import Options, batch_gcode  # noqa: E402

DEFAULTS = {"mode": "geometry", "fixed_height": 0.4, "head_profile": None, "head_far": None,
            "tip_tolerance": 0.0, "island_gap": 3.5, "tolerance_zones": None}


class ColorBatching(orca.slicing.SlicingPipelineCapabilityBase):
    def get_name(self):
        return "Color Batching"

    def get_default_config(self):
        return dict(DEFAULTS)

    def execute(self, ctx):
        if ctx.step != orca.slicing.Step.psGCodePostProcess:
            return orca.ExecutionResult.success()
        try:
            conf = dict(DEFAULTS, **json.loads(self.get_config() or "{}"))
        except ValueError as exc:
            return orca.ExecutionResult.failure(orca.PluginResult.RecoverableError, f"bad plugin config: {exc}")
        if conf["mode"] == "off":
            return orca.ExecutionResult.skipped("mode = off")
        try:
            with open(ctx.gcode_path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            return orca.ExecutionResult.failure(orca.PluginResult.RecoverableError, f"cannot read G-code: {exc}")
        if "M620 S" not in text:
            return orca.ExecutionResult.skipped("no AMS tool changes in this G-code")
        opts = Options(mode=conf["mode"], fixed_height=float(conf["fixed_height"]),
                       head_profile=conf["head_profile"], head_far=conf["head_far"],
                       tip_tolerance=float(conf["tip_tolerance"]), island_gap=float(conf["island_gap"]),
                       tolerance_zones=conf["tolerance_zones"])
        try:
            out, stats = batch_gcode(text, opts)
        except Exception as exc:  # noqa: BLE001 - leave the file untouched on any failure
            return orca.ExecutionResult.failure(orca.PluginResult.RecoverableError, f"batching failed: {exc!r}")
        if stats.get("skipped"):
            return orca.ExecutionResult.skipped(stats["skipped"])
        # Rewrite in place (same path): Orca re-reads this file for the preview afterwards.
        try:
            with open(ctx.gcode_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(out)
        except OSError as exc:
            return orca.ExecutionResult.failure(orca.PluginResult.RecoverableError, f"cannot write G-code: {exc}")
        return orca.ExecutionResult.success(
            f"{stats['source_tool_changes']} -> {stats['tool_changes']} tool changes "
            f"({stats['head']}, {stats['segments']} units, {stats['travel_lifts']} lifted travels)")


@orca.plugin
class ColorBatchingPlugin(orca.base):
    def register_capabilities(self):
        orca.register_capability(ColorBatching)
