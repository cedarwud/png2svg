# Phase 18 — 3GPP 3-panel Extractor v1

Purpose
- Extract a geometry skeleton from PNG without looking at expected answers.
- Produce normalized params for `t_3gpp_events_3panel` with debug artifacts.

Inputs
- PNG (RGBA) image.

Outputs (minimum)
- `canvas` width/height (and `canvas_w`/`canvas_h` for convenience).
- Three panel bounds (x0/y0/x1/y1), ordered left-to-right.
- Per-panel axes: y-axis x, x-axis y, arrow direction, stroke width.
- Vertical dashed lines: t_start/t_trigger x positions (per panel or shared).
- TTT shaded region rectangles derived from t_start/t_trigger.
- Debug artifacts: preprocessed.png, overlays.png, extracted.json.

CV Detection Strategy
1) Preprocessing
   - Convert to RGBA and compute an ink mask via luminance/alpha thresholding.
   - Store a binary preprocessed image for debugging.

2) Panel Segmentation
   - Compute column projection (sum of ink per column).
   - Find two low-ink separators near 1/3 and 2/3 width.
   - If separators are unreliable, fall back to a default 3-column layout.

3) Axes Detection
   - For each panel, run long-line detection on the ink mask.
   - Identify the longest vertical line as y-axis and longest horizontal line as x-axis.
   - Snap axis positions to panel edges if within ±3 px.

4) Dashed Line (t_start/t_trigger) Detection
   - Scan panel columns for repeated short segments (dash pattern).
   - Cluster co-linear segments into dashed-line candidates.
   - Choose leftmost/rightmost dashed lines as t_start/t_trigger.
   - If missing, fall back to default ratios (0.2/0.6).

5) TTT Region
   - Derive shaded rectangles from t_start/t_trigger and panel bounds.

Determinism
- No randomness; all thresholds are deterministic and derived from image size.

Debug Artifacts
- `preprocessed.png`: binarized/thresholded view used for detection.
- `overlays.png`: panels/axes/dashed lines drawn over the input.
- `extracted.json`: extracted params (including v1 metadata).
