from __future__ import annotations

from pathlib import Path

import yaml

from png2svg import convert_png
from validators.validate import validate_svg
from validators.visual_diff import rasterize_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "regression_v0" / "manifest.yaml"
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"
THRESHOLDS = ROOT / "config" / "validator_thresholds.v1.yaml"


def _load_manifest() -> list[dict[str, str]]:
    data = yaml.safe_load(MANIFEST.read_text())
    return data["cases"]


def _case_dir(entry: dict[str, str]) -> Path:
    return ROOT / "datasets" / "regression_v0" / entry["dir"]


def _expected_png(case_dir: Path, tmp_path: Path) -> Path:
    png_path = case_dir / "expected.png"
    if png_path.exists():
        return png_path
    svg_path = case_dir / "expected.svg"
    out_path = tmp_path / f"{case_dir.name}_expected.png"
    rasterize_svg_to_png(svg_path, out_path)
    return out_path


def test_convert_e2e_regression_cases(tmp_path: Path) -> None:
    for entry in _load_manifest():
        case_dir = _case_dir(entry)
        input_png = _expected_png(case_dir, tmp_path)
        output_svg = tmp_path / f"{case_dir.name}.svg"
        result = convert_png(
            input_png,
            output_svg,
            topk=2,
            contract_path=CONTRACT,
            thresholds_path=THRESHOLDS,
            enable_visual_diff=False,
        )
        assert result["status"] == "pass"
        report = validate_svg(output_svg, CONTRACT, THRESHOLDS)
        assert report.status == "pass"


def test_convert_debug_artifacts(tmp_path: Path) -> None:
    entry = _load_manifest()[0]
    case_dir = _case_dir(entry)
    input_png = _expected_png(case_dir, tmp_path)
    output_svg = tmp_path / "out.svg"
    debug_dir = tmp_path / "debug"
    result = convert_png(
        input_png,
        output_svg,
        debug_dir=debug_dir,
        topk=1,
        contract_path=CONTRACT,
        thresholds_path=THRESHOLDS,
        enable_visual_diff=False,
    )
    assert result["status"] == "pass"
    assert (debug_dir / "classification.json").exists()
    assert (debug_dir / "classify" / "overlay.png").exists()
    assert (debug_dir / "classify" / "features.json").exists()
    assert (debug_dir / "convert_report.json").exists()
    candidate_dir = next(debug_dir.glob("candidate_01_*"))
    assert (candidate_dir / "params.json").exists()
    assert (candidate_dir / "generated.svg").exists()
    assert (candidate_dir / "validate_report.json").exists()
    assert (candidate_dir / "extract" / "extract_report.json").exists()
    assert (candidate_dir / "snap_preview.svg").exists()
    assert (candidate_dir / "snap_preview.png").exists()
