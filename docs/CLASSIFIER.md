# Template Classifier (PNG → Template ID)

The classifier returns a stable JSON payload for downstream pipelines.
When `--debug-dir` is provided, it writes `overlay.png` and `features.json`.

Schema (informal):
```
{
  "template_id": "t_3gpp_events_3panel" | "t_procedure_flow" | "t_performance_lineplot",
  "confidence": 0.0-1.0,
  "candidate_templates": [
    {"template_id": "...", "score": number}
  ],
  "image_meta": {"width": int, "height": int},
  "features_summary": {
    "ink_ratio": number,
    "saturated_ratio": number,
    "axis_aligned_ratio": number,
    "long_vertical_lines": int,
    "long_horizontal_lines": int,
    "short_vertical_segments": int,
    "short_horizontal_segments": int
  }
}
```

Example:
```
{
  "template_id": "t_performance_lineplot",
  "confidence": 0.82,
  "candidate_templates": [
    {"template_id": "t_performance_lineplot", "score": 1.8},
    {"template_id": "t_procedure_flow", "score": 0.6},
    {"template_id": "t_3gpp_events_3panel", "score": 0.2}
  ],
  "image_meta": {"width": 800, "height": 300},
  "features_summary": {
    "ink_ratio": 0.08,
    "saturated_ratio": 0.05,
    "axis_aligned_ratio": 0.72,
    "long_vertical_lines": 1,
    "long_horizontal_lines": 1,
    "short_vertical_segments": 6,
    "short_horizontal_segments": 6
  }
}
```
