# png2svg (MVP v0 scaffold)

Template-based PNG-to-editable-SVG pipeline (MVP v0).

## Setup
- Python 3.10+
- Install deps: `pip install -r requirements.txt`

## Quickstart
Minimal path: render → validate → regress.
1) Render: `python tools/png2svg.py path/to/input.png path/to/params.json output/output.svg`
2) Validate: `python tools/validate_svg.py output/output.svg --expected path/to/expected.png --report output/report.json`
3) Regress: `python tools/regress.py datasets/regression_v0/manifest.yaml --report output/regress_report.json`

## CLI
Top-level commands:
- Render: `python tools/png2svg.py path/to/input.png path/to/params.json path/to/output.svg`
- Validate: `python tools/validate_svg.py path/to/output.svg --expected path/to/expected.png --report report.json`
- Regress: `python tools/regress.py datasets/regression_v0/manifest.yaml --report report.json`

Template examples (using `samples/`):
- 3GPP events: `python tools/png2svg.py samples/input.png samples/t_3gpp_events_3panel.json output/3gpp.svg`
- Procedure flow: `python tools/png2svg.py samples/input.png samples/t_procedure_flow.json output/flow.svg`
- Performance line plot: `python tools/png2svg.py samples/input.png samples/t_performance_lineplot.json output/lineplot.svg`

## Tests
- `pytest`

## CHANGELOG
MVP v0:
- Template-based SVG renderers for 3GPP events, procedure flow, and performance line plots.
- Contract validator with visual diff and regression runner with per-case artifacts.
- Sample params for all templates and manifest-driven regression dataset.

Known limitations:
- No automatic params extraction from PNGs (manual params.json required).
- Output quality is guaranteed only within the supported templates.
