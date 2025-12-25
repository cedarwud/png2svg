# Project Status (MVP v0)

## Current Capabilities
- Template-based SVG rendering via `tools/png2svg.py` using `params.json` + `input.png` canvas size (or explicit canvas in params).
- Supported templates: `t_3gpp_events_3panel`, `t_procedure_flow`, `t_performance_lineplot`.
- Contract validation via `tools/validate_svg.py` with stable error codes, messages, and hints; optional visual diff when `expected.png` is provided.
- Regression runner via `tools/regress.py` using `datasets/regression_v0/manifest.yaml`, with per-case artifacts in `output/regress/<case_id>/`.
- Sample params available under `samples/` for all templates.

## What You Can Do Now
- Render contract-compliant SVGs for the three supported templates.
- Validate output structure, typography, colors, stroke widths, and path complexity.
- Run end-to-end regression (render -> validate -> rasterize -> diff) on the dataset cases.
- Inspect generated artifacts (SVG/PNG/report/diff) per case under `output/regress/`.

## Pending Work
- Expand regression dataset to 10-15 cases and continue tightening thresholds safely.
- Add more templates beyond MVP v0 scope or automate params extraction from PNGs.

## Testing
- Unit tests: `python3 -m pytest`
- Regression suite: `python3 tools/regress.py datasets/regression_v0/manifest.yaml`
- Manual validation: `python3 tools/validate_svg.py path/to/output.svg --expected path/to/expected.png --report report.json`
