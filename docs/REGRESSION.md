# Regression Case Diversity Checklist

Use this checklist to avoid near-duplicate cases. Each case should introduce a
meaningful variation in layout, geometry, or constraints. Tag each case in
`datasets/regression_v0/manifest.yaml` using relevant tags:

`long_text`, `dashed`, `small_canvas`, `large_canvas`, `dense_ticks`,
`multi_series`, `lane_layout`, `many_nodes`

Optional quality gate overrides (convert pipeline only) can be added per case:
```
gates:
  rmse_max: 0.2
  bad_pixel_ratio_max: 0.05
  pixel_tolerance: 10
```

## t_3gpp_events_3panel
- Panel geometry changes (width/height/spacing) that affect axes and shading.
- Ratio changes (`t_start_ratio`, `t_trigger_ratio`) that move guides/TTT regions.
- Curve shapes that cross or diverge (serving vs neighbor) within the panels.
- Title length variations to cover `long_text`.

## t_procedure_flow
- Presence/absence of `lanes[]` (`lane_layout`).
- Vary node count and arrangement (`many_nodes`).
- Include dashed and solid edges (`dashed`).
- Use polyline edges with bends vs straight line edges.
- Include multiline node text (`long_text`) with tspans.

## t_performance_lineplot
- Vary series count (`multi_series`).
- Include dashed series (`dashed`).
- Dense vs sparse ticks (`dense_ticks`).
- Small vs large canvas sizes to stress layout (`small_canvas`, `large_canvas`).
- Longer titles/axis labels to cover `long_text`.

## t_project_architecture_v1
- Vary panel titles and bullet counts (`long_text`).
- Vary work package goal/output lengths (`long_text`).
- Include tighter canvas margins (`small_canvas`) or larger spacing (`large_canvas`).

## Real regression (external PNGs)
For real PNG figures that cannot be committed, set `REAL_PNG_DIR` to a local
folder and edit `datasets/real_regression_v1/manifest.yaml` with relative paths.

Example manifest entry:
```
- id: real_case_001
  relative_path: example/figure_001.png
  expected_templates:
    - t_performance_lineplot
    - t_3gpp_events_3panel
  allow_force_template: true
  gates:
    must_pass_validator: true
    max_bad_pixel_ratio: 0.05
    max_rmse: 0.05
    pixel_tolerance: 10
```

Run:
```
REAL_PNG_DIR=/path/to/real/pngs python tools/regress.py --real-manifest datasets/real_regression_v1/manifest.yaml
```
Real regression writes `output/regress_real/summary.json` and `summary.md`.

Triage failures:
```
python tools/triage_real_failures.py output/regress_real --out output/regress_real/triage.json
```

## Case Input Rebuild
Regenerate deterministic FAST/HARD inputs from `expected.svg`:
```
python tools/rebuild_case_inputs.py datasets/regression_v0 --variants fast,hard --overwrite
```

## Dataset Sanity Check
Validate that case inputs are not near-blank placeholders:
```
python tools/check_dataset_sanity.py datasets/regression_v0
```

## Convert Regression Variants
FAST (full run):
```
python tools/regress.py datasets/regression_v0 --pipeline convert --tier fast --input-variant fast
```

HARD (sampled):
```
python tools/regress.py datasets/regression_v0 --pipeline convert --tier fast --input-variant hard --limit 3
```

Input naming:
- `input.png` (fast)
- `input_hard.png` (hard)

## Hard Regression Tier
Hard cases live under `datasets/regression_hard_v1/` and use `input.png` as the
fig1-like input (expected assets are generated from params).

Rebuild expected/input assets:
```
python tools/build_hard_case_assets.py datasets/regression_hard_v1 --overwrite
```

Run hard tier:
```
python tools/regress.py datasets/regression_hard_v1 --pipeline convert --tier hard
```
Artifacts are written under `output/regress_hard/`.
If OCR (tesseract) is unavailable, hard-tier text guarantees may be skipped; install
`tesseract-ocr` to enable full coverage.

## Makefile Shortcuts
```
make regress-fast-render
make regress-fast-convert
make regress-fast-hard-sample
make regress-hard
REAL_PNG_DIR=/path/to/pngs make regress-real
make rebuild-inputs
make dataset-sanity
```

Run both tiers:
```
python tools/regress.py datasets/regression_v0 --pipeline convert --tier all
```
