# SDD: Project Architecture Template v1 (t_project_architecture_v1)

## 1) Background and Problem Statement
This project targets high-quality, editable SVG output for a fixed set of templates. For the Fig.6 Project Architecture diagram, the most reliable path is template-based reconstruction rather than generic vectorization. The diagram has a stable layout (top row of panels A/B/C and a bottom Work Packages container with WP1-WP4) and a stable visual style (rectangles, straight lines, minimal colors). That makes a deterministic, parameter-driven renderer the safest approach for quality and editability.

## 2) Scope and Non-goals
Scope:
- Add a new template renderer: `t_project_architecture_v1`.
- Support user-provided params for text content while keeping geometry fixed.
- Output contract-compliant SVG (no raster, no filters/gradients, limited colors, allowed stroke widths).

Non-goals:
- No OCR or generic tracing.
- No classifier changes are required for this route (use `--force-template`).
- No auto-layout optimization beyond simple word wrapping and truncation.

## 3) Input PNG Assumptions (Fixed Layout)
The input PNG is assumed to be:
- 1920x1080
- White background
- No gradients, filters, or blur
- Geometry uses only rounded rectangles, straight lines, and right-angle arrowheads
- Single sans-serif font, readable size, small line counts per box

## 4) System Architecture (Existing Pipeline)
- `tools/png2svg.py render`: reads params.json -> renderer -> SVG
- `src/templates/*`: template-specific renderers
- `src/common/svg_builder.py`: group creation and helper methods
- `config/figure_contract.v1.yaml`: SVG contract enforcement
- `tools/regress.py`: regression cases against expected.svg/png

This template plugs into the existing renderer and regression pipeline without new dependencies.

## 5) Data Model (params schema)
Template params use a fixed schema with text overrides only:

```
{
  "template": "t_project_architecture_v1",
  "canvas": {"width": 1920, "height": 1080},
  "title": "Project Architecture",
  "subtitle": "Work Packages (WP1-WP4)",
  "panels": [
    {"id": "A", "title": "Panel A: Core Platform", "bullets": ["item 1", "item 2"]},
    {"id": "B", "title": "Panel B: Data & Analytics", "bullets": ["item 1", "item 2"]},
    {"id": "C", "title": "Panel C: Integration", "bullets": ["item 1", "item 2"]}
  ],
  "work_packages": [
    {"id": "WP1", "title": "WP1", "goal": "short goal", "output": "short output"},
    {"id": "WP2", "title": "WP2", "goal": "short goal", "output": "short output"},
    {"id": "WP3", "title": "WP3", "goal": "short goal", "output": "short output"},
    {"id": "WP4", "title": "WP4", "goal": "short goal", "output": "short output"}
  ]
}
```

Rules:
- Panels must be exactly A/B/C.
- Work packages must be exactly WP1-WP4.
- Missing text uses defaults.
- Overlong text is wrapped and truncated (with "...").

## 6) Rendering Specification
Groups (per contract):
- `figure_root`
- `g_axes` (empty)
- `g_curves` (empty)
- `g_annotations` (rectangles, lines)
- `g_text` (all text)
- `g_markers` (arrowheads)

Geometry layout (fixed, relative to canvas size):
- Header area: title/subtitle
- Top row: 3 equal-width panels (A/B/C)
- Bottom row: a Work Packages container with 4 inner boxes
- Connectors: vertical lines from each panel down to the container, with arrowheads

Style rules:
- Stroke widths: 1 or 2
- Colors: black stroke/text (#000000), light gray fills (#f2f2f2), optional white fill (#ffffff)
- Font family: `Arial, sans-serif`
- Text anchors: start (left aligned)
- Multiline text uses `<tspan id="...">` per line

## 7) Optional Extraction (Minimal)
- No OCR. No template inference.
- Extractor only validates canvas size and returns default params for conversion.

## 8) Error Handling
All validation errors use `Png2SvgError` with stable codes:
- Canvas size mismatch (expects 1920x1080)
- Invalid panels/work_packages list structure
- Invalid panel or WP ids

Long text is wrapped and truncated (no hard error unless structure is invalid).

## 9) Testing Plan
- Regression case: `datasets/regression_v0/cases/case_018_project_architecture_v1`
- Unit test: render + validate using sample params
- Commands:
  - `python tools/png2svg.py render samples/input.png samples/t_project_architecture_v1.json output/project_arch.svg`
  - `python tools/regress.py datasets/regression_v0 --pipeline render`
  - `python -m pytest`

## 10) Acceptance Criteria
- SVG output is editable in Illustrator/Inkscape (text is `<text>`, no `<image>`)
- Contract validation passes (groups/colors/strokes/path limits)
- Regression case passes
- Deterministic output for the same params
