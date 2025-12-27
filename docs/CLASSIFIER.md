# Template Classifier (PNG → Template ID)

The classifier returns a stable JSON payload for downstream pipelines.
When `--debug-dir` is provided, it writes `overlay.png` and `features.json`.

Schema (informal):
```
{
  "template_id": "t_3gpp_events_3panel" | "t_procedure_flow" | "t_performance_lineplot" | "t_project_architecture_v1" | "t_rl_agent_loop_v1" | "t_performance_grid_v1" | "unknown",
  "decision": "known" | "unknown",
  "reason_codes": ["LOW_CONFIDENCE" | "AMBIGUOUS_MARGIN"],
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
    "short_horizontal_segments": int,
    "color_count": int,
    "ocr_token_count": int
  }
}
```

Example:
```
{
  "template_id": "t_performance_lineplot",
  "decision": "known",
  "reason_codes": [],
  "confidence": 0.82,
  "candidate_templates": [
    {"template_id": "t_performance_lineplot", "score": 1.8},
    {"template_id": "t_procedure_flow", "score": 0.6},
    {"template_id": "t_3gpp_events_3panel", "score": 0.2},
    {"template_id": "t_performance_grid_v1", "score": 0.1}
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

Thresholds for unknown gating live in `config/classifier_thresholds.v1.yaml`.

Heuristics (high-level):
- `t_rl_agent_loop_v1`: OCR keyword hits (agent/environment/action/reward/policy/etc) plus axis-aligned layout.
- `t_performance_grid_v1`: multi-panel separators (long vertical/horizontal lines) plus grid-oriented OCR hints.
