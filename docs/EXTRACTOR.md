# Extract Skeleton (PNG → params.json)

The extractor returns a fixed, best-effort params schema that is accepted by the
template renderers. Fields may be `null` or placeholder values when detection is
uncertain. Debug artifacts are written to `--debug-dir`:

- `preprocessed.png` (binary/gray image used for detection)
- `overlay.png` (detected boxes/axes/panels drawn on the input)
- `ocr.json` (raw OCR box list, text may be null)
- `extract_report.json` (warnings/errors with stable codes)

Common fields:
```
{
  "template": "t_3gpp_events_3panel" | "t_procedure_flow" | "t_performance_lineplot",
  "canvas": {"width": int, "height": int},
  "title": string | null,
  "extracted": { ... }   // debug-only fields
}
```

Text extraction fields (debug):
```
{
  "extracted": {
    "texts_detected": int,
    "text_items": [
      {
        "content": string,
        "x": number,
        "y": number,
        "role": string,
        "anchor": "start" | "middle" | "end",
        "baseline_group": string | null
      }
    ]
  }
}
```

Template-specific fields (minimum):

1) t_3gpp_events_3panel
```
{
  "panels": [{"id","label","x","y","width","height"} x3],
  "t_start_ratio": number,
  "t_trigger_ratio": number,
  "curves": {"serving": [{"x","y"}...], "neighbor": [{"x","y"}...]}
}
```

2) t_performance_lineplot
```
{
  "axes": {
    "plot": {"x","y","width","height"},
    "x": {"label","min","max","ticks"},
    "y": {"label","min","max","ticks"}
  },
  "series": [{"id","label","color","dashed","dasharray","stroke_width","points"}...]
}
```

3) t_procedure_flow
```
{
  "lanes": [],
  "nodes": [{"id","x","y","width","height","rx","ry","text"}...],
  "edges": [{"from","to","label","dashed","dasharray","points"?}...]
}
```
