# png2svg (MVP v0 scaffold)

Template-based PNG-to-editable-SVG pipeline (MVP v0).

## Setup
- Python 3.10+
- **System Dependencies**:
  - **OCR**: `tesseract-ocr` (required for text recognition)
  - **Cairo**: `libcairo2` or `cairo` (required for `cairosvg` validation)
  - *Ubuntu/Debian*: `sudo apt-get install tesseract-ocr libcairo2`
  - *macOS*: `brew install tesseract cairo`
- **Python Deps**: `pip install -r requirements.txt`

## Quickstart
Minimal path: convert (auto) → inspect.
1) **Using Makefile (Recommended)**:
   - `make convert IN=samples/realistic/fig1.png` (Auto-detect template)
   - `make convert-auto IN=samples/input.png` (Force Generalized V1 engine)
2) **Using CLI**:
   - `python tools/png2svg.py convert samples/realistic/fig1.png -o output/output.svg --quality-gate off`

**Output**: All generated SVGs are saved to the `output/` directory by default when using `make`.

## CLI
Top-level commands:
- Render: `python tools/png2svg.py render path/to/input.png path/to/params.json path/to/output.svg`
- Classify: `python tools/png2svg.py classify path/to/input.png --out output/classification.json --debug-dir output/debug`
- Extract: `python tools/png2svg.py extract path/to/input.png --template auto --out output/params.json --debug-dir output/extract_debug`
- Convert: `python tools/png2svg.py convert path/to/input.png -o output/output.svg --debug-dir output/convert_debug --topk 2 --text-mode hybrid`
- Validate: `python tools/validate_svg.py path/to/output.svg --expected path/to/expected.png --report report.json`
- Regress (fast tier): `python tools/regress.py datasets/regression_v0 --tier fast --report report.json`
- Rebuild case inputs: `python tools/rebuild_case_inputs.py datasets/regression_v0 --variants fast,hard --overwrite`
- Build hard case assets: `python tools/build_hard_case_assets.py datasets/regression_hard_v1 --overwrite`
- Dataset sanity check: `python tools/check_dataset_sanity.py datasets/regression_v0`

Recommended commands (Makefile shortcuts):
- `make convert IN=path/to/image.png`: Convert image using classifier (output to `output/`).
- `make convert-auto IN=path/to/image.png`: Convert image using Generalized Layout V1 (output to `output/`).
- `make pytest`: Run unit tests.
- `make regress-fast-render`
- `make regress-fast-convert`
- `make regress-fast-hard-sample`
- `make regress-hard`
- `REAL_PNG_DIR=/path/to/pngs make regress-real`
- `make rebuild-inputs`
- `make build-hard-assets`
- `make dataset-sanity`
- `make check-png`
Note: set `PYTHON=python` when you are not using the `venv/` virtualenv.

Debug artifacts:
- `--debug-dir` stores classification/extraction/validation outputs for troubleshooting.
- `convert` debug dirs include `snap_preview.svg/png` for quick inspection of snapped geometry.
- `convert` runs a quality gate against the input PNG; override with `--gate-*` or disable with `--quality-gate off`.
- `--allow-failed-gate` writes an SVG even when the quality gate fails (report still marks failure).
- `--emit-report-json` writes a convert_report.json summary to the debug directory.

Regression assets:
- `expected.svg` is the golden source; regress rasterizes it to compare with generated output.
- `expected.png` is optional and only used when `expected.svg` is missing.
- `input.png` is the FAST convert input (rasterized from `expected.svg`).
- `input_hard.png` is the HARD convert input (deterministically degraded from `input.png`).
- Hard-tier inputs live under `datasets/regression_hard_v1/` and use `input.png` as a fig1-like degraded input.

Classifier schema: see `docs/CLASSIFIER.md`.
Extractor schema: see `docs/EXTRACTOR.md`.

Template examples (using `samples/`):
- 3GPP events: `python tools/png2svg.py render samples/input.png samples/t_3gpp_events_3panel.json output/3gpp.svg`
- Procedure flow: `python tools/png2svg.py render samples/input.png samples/t_procedure_flow.json output/flow.svg`
- Performance line plot: `python tools/png2svg.py render samples/input.png samples/t_performance_lineplot.json output/lineplot.svg`
- Project architecture: `python tools/png2svg.py render samples/input.png samples/t_project_architecture_v1.json output/project_architecture.svg`
- RL agent loop: `python tools/png2svg.py render samples/input.png samples/t_rl_agent_loop_v1.json output/rl_loop.svg`
- Performance grid: `python tools/png2svg.py render samples/input.png samples/t_performance_grid_v1.json output/perf_grid.svg`

## Tests
- `pytest`

## CHANGELOG
Phase 18 (Generalized Extraction):
- Integrated **Generalized Layout Analysis** (V1).
- Added `t_auto_layout` pipeline: automatically detects panels and curves without predefined templates.
- Enhanced curve quality using Bezier fitting and morphological cleanup.
- Improved text extraction with font-weight detection and vocabulary filtering.

Known limitations:
- Automatic extraction is highly effective for charts/diagrams; complex artistic illustrations may still require manual tuning.
- Visual diff quality gate may require `--quality-gate off` for highly creative layouts.
