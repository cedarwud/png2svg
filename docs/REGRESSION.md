# Regression Case Diversity Checklist

Use this checklist to avoid near-duplicate cases. Each case should introduce a
meaningful variation in layout, geometry, or constraints. Tag each case in
`datasets/regression_v0/manifest.yaml` using relevant tags:

`long_text`, `dashed`, `small_canvas`, `large_canvas`, `dense_ticks`,
`multi_series`, `lane_layout`, `many_nodes`

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

## Real regression (external PNGs)
For real PNG figures that cannot be committed, set `REAL_PNG_DIR` to a local
folder and edit `datasets/real_regression_v1/manifest.yaml` with relative paths.

Example manifest entry:
```
- id: real_case_001
  relative_path: example/figure_001.png
  expected_template: t_performance_lineplot
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
