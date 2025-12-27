from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from png2svg import convert_png
from png2svg.ocr import has_pytesseract, has_tesseract
from validators.validate import validate_svg
from validators.config import load_thresholds


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "regression_v0" / "manifest.yaml"
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"
THRESHOLDS = ROOT / "config" / "validator_thresholds.v1.yaml"


def _load_manifest() -> list[dict[str, str]]:
    data = yaml.safe_load(MANIFEST.read_text())
    return data["cases"]


def _case_dir(entry: dict[str, str]) -> Path:
    return ROOT / "datasets" / "regression_v0" / entry["dir"]


def _input_png(case_dir: Path) -> Path:
    png_path = case_dir / "input.png"
    if not png_path.exists():
        raise AssertionError(f"missing input.png: {png_path}")
    return png_path


def _input_hard_png(case_dir: Path) -> Path:
    png_path = case_dir / "input_hard.png"
    if not png_path.exists():
        raise AssertionError(f"missing input_hard.png: {png_path}")
    return png_path


def _force_template(entry: dict[str, str]) -> str | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("force_template")
    if value:
        return str(value)
    return None


def _gate_thresholds(variant: str) -> dict[str, float | int]:
    thresholds = load_thresholds(THRESHOLDS)
    if variant == "hard":
        gate = thresholds.get("quality_gate_hard") or thresholds.get("quality_gate") or {}
    else:
        gate = thresholds.get("quality_gate") or {}
    return {
        "rmse_max": float(gate.get("rmse_max", 1.0)),
        "bad_pixel_ratio_max": float(gate.get("bad_pixel_ratio_max", 1.0)),
        "pixel_tolerance": int(gate.get("pixel_tolerance", 10)),
    }


def test_convert_e2e_regression_cases(tmp_path: Path) -> None:
    for entry in _load_manifest():
        case_dir = _case_dir(entry)
        input_png = _input_png(case_dir)
        output_svg = tmp_path / f"{case_dir.name}.svg"
        force_template = _force_template(entry)
        result = convert_png(
            input_png,
            output_svg,
            topk=1 if force_template else 2,
            force_template=force_template,
            contract_path=CONTRACT,
            thresholds_path=THRESHOLDS,
        )
        assert result["status"] == "pass"
        report = validate_svg(output_svg, CONTRACT, THRESHOLDS)
        assert report.status == "pass"


def test_convert_debug_artifacts(tmp_path: Path) -> None:
    entry = _load_manifest()[0]
    case_dir = _case_dir(entry)
    input_png = _input_png(case_dir)
    output_svg = tmp_path / "out.svg"
    debug_dir = tmp_path / "debug"
    result = convert_png(
        input_png,
        output_svg,
        debug_dir=debug_dir,
        topk=1,
        contract_path=CONTRACT,
        thresholds_path=THRESHOLDS,
    )
    assert result["status"] == "pass"
    assert (debug_dir / "classification.json").exists()
    assert (debug_dir / "classify" / "overlay.png").exists()
    assert (debug_dir / "classify" / "features.json").exists()
    assert (debug_dir / "convert_report.json").exists()
    candidate_dir = debug_dir / "candidates" / result["selected_template"]
    assert (candidate_dir / "params.json").exists()
    assert (candidate_dir / "out.svg").exists()
    assert (candidate_dir / "validate_report.json").exists()
    assert (candidate_dir / "gate_report.json").exists()
    assert (candidate_dir / "rendered.png").exists()
    assert (candidate_dir / "extract" / "extract_report.json").exists()
    assert (candidate_dir / "snap_preview.svg").exists()
    assert (candidate_dir / "snap_preview.png").exists()
    assert (debug_dir / "final" / "out.svg").exists()
    assert (debug_dir / "final" / "gate_report.json").exists()


def test_convert_hard_inputs_sampled(tmp_path: Path) -> None:
    sample_ids = {
        "t_3gpp_events_3panel": "case_013_3gpp_realistic",
        "t_procedure_flow": "case_014_flow_realistic",
        "t_performance_lineplot": "case_015_lineplot_realistic",
    }
    thresholds = _gate_thresholds("hard")
    for template_id, case_id in sample_ids.items():
        case_dir = ROOT / "datasets" / "regression_v0" / "cases" / case_id
        input_png = _input_hard_png(case_dir)
        output_svg = tmp_path / f"{case_id}.svg"
        result = convert_png(
            input_png,
            output_svg,
            debug_dir=tmp_path / f"debug_{case_id}",
            topk=2,
            force_template=template_id,
            gate_rmse_max=thresholds["rmse_max"],
            gate_bad_pixel_max=thresholds["bad_pixel_ratio_max"],
            gate_pixel_tolerance=thresholds["pixel_tolerance"],
        )
        assert result["status"] == "pass"
        selected = result.get("selected", {})
        gate = selected.get("quality_gate", {})
        assert gate.get("status") == "pass"
        assert float(gate.get("rmse", 1.0)) <= float(gate.get("rmse_max", 1.0))
        assert float(gate.get("bad_pixel_ratio", 1.0)) <= float(
            gate.get("bad_pixel_ratio_max", 1.0)
        )


def test_convert_project_architecture_auto(tmp_path: Path) -> None:
    if not (has_pytesseract() or has_tesseract()):
        pytest.skip("OCR backend not available")
    case_dir = ROOT / "datasets" / "regression_v0" / "cases" / "case_018_project_architecture_v1"
    input_png = _input_png(case_dir)
    output_svg = tmp_path / "case_018_auto.svg"
    result = convert_png(
        input_png,
        output_svg,
        topk=2,
        contract_path=CONTRACT,
        thresholds_path=THRESHOLDS,
    )
    assert result["status"] == "pass"
    assert result.get("selected_template") == "t_project_architecture_v1"
    report = validate_svg(output_svg, CONTRACT, THRESHOLDS)
    assert report.status == "pass"
