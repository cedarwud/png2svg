# Extract Skeleton (PNG → params.json)

The extractor returns a fixed, best-effort params schema that is accepted by the
template renderers. Fields may be `null` or placeholder values when detection is
uncertain. Debug artifacts are written to `--debug-dir`:

- `01_preprocessed.png` (binary/gray image used for detection)
- `02_overlay.png` (detected boxes/axes/panels/lines drawn on the input)
- `03_ocr_raw.json` (raw OCR box list)
- `04_params.json` (normalized params.json output)
- `05_snap_preview.svg` (geometry preview after snapping)
- `extract_report.json` (warnings/errors with stable codes)
- `effective_config.json` (adaptive parameters used for this image)

Common fields:
```
{
  "template": "t_3gpp_events_3panel" | "t_procedure_flow" | "t_performance_lineplot" | "t_project_architecture_v1" | "t_rl_agent_loop_v1" | "t_performance_grid_v1",
  "canvas": {"width": int, "height": int},
  "title": string | null,
  "texts": [ ... ],
  "geometry": { ... },
  "extracted": { ... }   // debug-only fields
}
```

Text extraction fields (debug):
```
{
  "extracted": {
    "texts_detected": int,
    "ocr_backend": "auto" | "tesseract" | "none",
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

Text items (params):
```
{
  "texts": [
    {
      "content": string,
      "x": number,
      "y": number,
      "bbox": {"x","y","width","height"},
      "role": string,
      "anchor": "start" | "middle" | "end",
      "baseline_group": string | null,
      "conf": number,
      "font_size": number,
      "roi_id": string | null
    }
  ]
}
```

Geometry primitives (params):
```
{
  "geometry": {
    "lines": [{"x1","y1","x2","y2","stroke","stroke_width","dashed","dasharray","role"}],
    "rects": [{"x","y","width","height","stroke","stroke_width","fill","role"}],
    "markers": [{"x","y","radius","fill","role"}]
  }
}
```

Optional OCR (tesseract):
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- RHEL/CentOS: `sudo yum install tesseract`
- Optional Python wrapper: `pip install pytesseract`
- Backend selection: `PNG2SVG_OCR_BACKEND=auto|pytesseract|tesseract|none`
- OCR runs on explicit ROIs only (full-image OCR is disabled).
- Per-ROI timeout and max OCR dimension are controlled by `ocr.timeout_sec` and `ocr.max_dim`
  in `config/extract_adaptive.v1.yaml`.
- When `--debug-dir` is used, OCR results are cached in `ocr_cache.json` to avoid re-running OCR
  across candidate templates.

Adaptive extractor settings are defined in `config/extract_adaptive.v1.yaml`.
OCR cleanup knobs include `max_bbox_height_ratio`, `max_line_height_ratio`,
and `max_line_aspect_ratio` for filtering noisy or vertical text lines.

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

4) t_rl_agent_loop_v1
```
{
  "title": string,
  "boxes": [{"id","role","label","enabled"}...],
  "signals": {"action": string, "feedback": string}
}
```

5) t_performance_grid_v1
```
{
  "layout": "2x2" | "1x3",
  "panels": [{"id","title","series":[{"id","points","stroke","dashed","dasharray"}...]}...],
  "legend": [{"id","label","stroke","dashed"}...]
}
```
