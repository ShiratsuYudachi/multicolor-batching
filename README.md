# multicolor-batching

Cross-layer colour batching for single-nozzle multi-material printers (Bambu A1 / A1 mini + AMS lite).
Instead of changing filament on every layer, one colour prints several layers in a row wherever the
print head cannot hit already-printed parts, which cuts tool changes and flush waste.

Stage A of plan C: an OrcaSlicer **slicing-pipeline plugin** that rewrites the exported G-code
(`psGCodePostProcess`). No Orca rebuild is needed. Stage B (moving this into the Orca fork
`Dev/OrcaSlicer-batching`, branch `feature/cross-layer-batching`) is still to do.

## Use in OrcaSlicer (≥ 2.5.0)

1. `python tools/install_plugin.py` copies the plugin to `%APPDATA%\OrcaSlicer\orca_plugins\color_batching`.
   Restart Orca. On first load Orca installs `numpy` for the plugin with uv.
2. Process settings › Others › **Slicing pipeline plugins** › Add plugin › `Color Batching`.
3. Prime tower **off**, print sequence **by layer**. Otherwise the plugin skips the file.
4. Slice. On the first run Orca asks for the Python `open` permission; allow it.

On Bambu printers Orca re-reads the rewritten file for the preview, so the layer slider shows the new
print order: each "layer" in the slider is one batched unit. **The statistics panel (filament change
count, flush grams, time) is not recalculated by Orca 2.5.0 and still shows the original numbers.**
To get correct numbers, export the G-code and drag it back into Orca, or use the CLI below.

Plugin config (Plugins › Config, JSON):

| key | default | meaning |
|---|---|---|
| `mode` | `"geometry"` | `"off"`, `"fixed"` (ITC-style single height), `"geometry"` (print-head profile) |
| `fixed_height` | `0.4` | fixed mode: allowed height difference to printed parts, mm |
| `head_profile` | A1 hotend CAD | geometry mode: `[[radius, height], …]`; within `radius` of the nozzle axis, obstacles may stand `height` above the tip |
| `head_far` | `25` | allowed height beyond the profile (X-rod clearance) |
| `tip_tolerance` | `0` | geometry mode: height floor added to every band (empirical squish tolerance, e.g. 0.4) |
| `island_gap` | `3.5` | geometry mode: islands of one colour farther apart than this are scheduled separately |
| `tolerance_zones` | `null` | calibration: `[[x_min, x_max, tolerance], …]` in bed X; islands centred in a zone use that `tip_tolerance`. `tools/models.py tol_test` builds the matching five-pair plate (0 / 0.2 / 0.4 / 0.6 / 0.8 mm, left to right) and `tol_zones()` its zones |

The default A1 head profile in `colorbatch/head.py` takes the lower of two community-measured CAD models:
[Bambulab A1 Hotend (Printables 879230)](https://www.printables.com/model/879230-bambulab-a1-hotend)
(tip land r 0.5 mm, cone, silicone-sock bottom 1.9 mm above the tip out to r 10.5 mm) and the full
[Bambulab A1 CAD step model (MakerWorld 770920)](https://makerworld.com/en/models/770920-bambulab-a1-cad-step-model)
(heater block 2.7 mm, no sock; toolhead shell/duct bottom 5.4 mm, reaching r 72 mm on one side; X gantry
27.4 mm). Beyond the shell the profile uses Orca's X-rod clearance of 25 mm. Both models are
hand-measured (±0.2 mm claimed for the full printer); the two disagree on the sock (1.9 vs 2.7 mm).

## CLI

```
python -m colorbatch in.gcode -o out.gcode [--mode geometry|fixed] [--height 0.4]
                     [--tip-tolerance 0.4] [--island-gap 3.5] [--profile JSON] [--check]
```

`--check` replays the output with the head model and lists every move that comes closer to printed
material than the model allows.

## How it works

* `colorbatch/gcode.py` splits the body into units (one colour × one layer, optionally per island
  cluster) and tool-change blocks; each unit records its lead-in position, retraction state, machine
  state (temperatures, fans, acceleration, feed), object label and footprint (0.5 mm grid).
* `colorbatch/schedule.py` orders the units. A unit u is printable when the units under it are
  printed (support) and when (a) every printed taller unit is far enough away for the head profile
  and (b) every unprinted lower unit stays printable afterwards, so the scheduler never deadlocks.
  Two strategies run, `snake` (alternating tool sweeps) and `greedy` (look-ahead per switch), on both
  island units and whole-layer units. The order with the least flush volume is kept. If none beats
  Orca's own order, the file is left unchanged.
* `colorbatch/emit.py` writes the new order. Tool-change blocks are reused from the source G-code with
  Z rebased onto the tallest printed point. Missing (from, to) pairs are synthesised by scaling the
  flush of a block with the same target. Each unit gets a lift/travel/lower move, a state restore,
  a retraction fix and skip-object labels. Travel moves that pass too close to taller parts are lifted.
  Per-layer timelapse snapshots are dropped, because they park the head at layer height.
* `colorbatch/progress.py` regenerates the printer's progress (`M73 P… R…`) and the header time for
  the new order. Each line gets a feed-rate-model duration, scaled by two factors (tool-change blocks,
  everything else) fitted to Orca's own M73 timeline of the source file. Retiming the unchanged source
  reproduces Orca's total exactly; for the new order it is an estimate in Orca's terms.

## Evaluation (`python tools/evaluate.py`)

Models are generated by `tools/models.py` (and `work/ki43`, a painted 4-colour 1/48 aircraft) and sliced
headless with Orca 2.5.0 for the A1, 0.20 mm Standard, PLA, no prime tower. Purge counts the extrusion
inside tool-change blocks. Time is a rough feed-rate estimate, comparable between rows only.
Collisions come from `report.check_collisions` with the same head model.

| model | order | tool changes | purge cm³ | est. min | collisions |
|---|---|---|---|---|---|
| pillars (4 × Ø8 mm, 16 mm pitch) | Orca layer order | 300 | 91.7 | 168 | – |
| | fixed 0.4 mm | 102 | 31.2 | 59 | 0 |
| | geometry | 25 | 7.6 | 18 | 0 |
| figurine (body + arms 3 mm away + touching badge) | Orca | 160 | 48.9 | 99 | – |
| | fixed 0.4 mm | 55 | 16.8 | 38 | 0 |
| | geometry | 90 | 27.5 | 57 | 0 |
| | geometry + 0.4 floor | 35 | 10.7 | 27 | 0 |
| split cube (two touching halves) | Orca | 100 | 30.6 | 62 | – |
| | fixed 0.4 mm | 34 | 10.4 | 24 | 0 |
| | geometry | 100 (unchanged) | 30.6 | 62 | 0 |
| Ki-43 painted 4-colour | Orca | 202 | 61.7 | 202 | – |
| | fixed 0.4 mm | 93 | 28.4 | 135 | 0 |
| | geometry | 172 | 52.6 | 178 | 0 |
| | geometry + 0.4 floor | 84 | 25.7 | 130 | 0 |

In every run the model extrusion per filament is identical to the source.

Reading: separated colour regions gain the most (pillars −92 % purge). Where colours touch, pure
geometry mode allows no height step, because the tip land sits over the neighbour. Touching colours
therefore only batch with a tolerance floor (fixed mode or `tip_tolerance`). ITC reports 0.35–0.4 mm
working on its printers; this is unverified on the A1.

## Known limits

* Prime tower must be off. "Flush into infill/objects" needs the tower in Orca, so it is unused here.
* Orca's statistics panel is stale after the rewrite (see above). The `.gcode.3mf` slice info (shown by
  Bambu Handy / cloud) carries the same stale numbers. The printer screen uses the regenerated M73.
* Synthesised tool-change blocks (pairs absent from the source) scale the flush of another block with
  the same target. Filament-specific start/end G-code of the *old* filament comes from that block.
* Bed/nozzle temperature and fan changes between first layer and the rest are restored per unit,
  so the first-layer settings can switch back and forth while early layers are still being batched.
* Toolhead/part-cooling-duct geometry beyond the hotend, interface strength between batches, and
  stringing are unmeasured.
