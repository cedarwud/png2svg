from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from png2svg import convert_png
from png2svg.ocr import has_pytesseract, has_tesseract
from validators.config import load_thresholds
from validators.validate import (
    E2008_TEXT_MISSING,
    E2009_TEXT_ID_MISSING,
    E2010_TEXT_AS_PATH,
    E2011_TSPAN_ID_MISSING,
    E2012_MULTILINE_TSPAN_MISSING,
    E2014_TEXT_COUNT_LOW,
    E2015_TEXT_FALLBACK_MISSING,
    E2016_TEXT_ANCHOR_INVALID,
    E2017_TEXT_OUTLINE_DETECTED,
    W2106_TEXT_BASELINE_NOT_ALIGNED,
)

ROOT = Path(__file__).resolve().parents[1]
HARD_CASE = ROOT / "datasets" / "regression_hard_v1" / "cases" / "case_3gpp_fig1_like"
INPUT_PNG = HARD_CASE / "input.png"
THRESHOLDS = ROOT / "config" / "validator_thresholds.v1.yaml"
TEMPLATE_ID = "t_3gpp_events_3panel"

TEXT_ERROR_CODES = {
    E2008_TEXT_MISSING,
    E2009_TEXT_ID_MISSING,
    E2010_TEXT_AS_PATH,
    E2011_TSPAN_ID_MISSING,
    E2012_MULTILINE_TSPAN_MISSING,
    E2014_TEXT_COUNT_LOW,
    E2015_TEXT_FALLBACK_MISSING,
    E2016_TEXT_ANCHOR_INVALID,
    E2017_TEXT_OUTLINE_DETECTED,
}


def _has_rasterizer() -> bool:
    if shutil.which("resvg"):
        return True
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_available() -> bool:
    backend = os.environ.get("PNG2SVG_OCR_BACKEND", "auto").lower()
    if backend == "none":
        return False
    if backend == "pytesseract":
        return has_pytesseract()
    if backend == "tesseract":
        return has_tesseract()
    return has_pytesseract() or has_tesseract()


def _gate_thresholds() -> dict[str, float | int]:
    thresholds = load_thresholds(THRESHOLDS)
    gate = thresholds.get("quality_gate_hard") or thresholds.get("quality_gate") or {}
    return {
        "rmse_max": float(gate.get("rmse_max", 1.0)),
        "bad_pixel_ratio_max": float(gate.get("bad_pixel_ratio_max", 1.0)),
        "pixel_tolerance": int(gate.get("pixel_tolerance", 10)),
    }


@pytest.mark.skipif(not _ocr_available(), reason="OCR backend unavailable")
def test_hard_case_text_editability(tmp_path: Path) -> None:
    assert INPUT_PNG.exists()
    if not _has_rasterizer():
        pytest.skip("Rasterizer unavailable (resvg/cairosvg required).")
    thresholds = _gate_thresholds()
    output_svg = tmp_path / "hard.svg"
    result = convert_png(
        INPUT_PNG,
        output_svg,
        debug_dir=tmp_path / "debug",
        topk=2,
        force_template=TEMPLATE_ID,
        gate_rmse_max=thresholds["rmse_max"],
        gate_bad_pixel_max=thresholds["bad_pixel_ratio_max"],
        gate_pixel_tolerance=thresholds["pixel_tolerance"],
    )
    assert result["status"] == "pass"
    validation = result.get("selected", {}).get("validation", {})
    error_codes = {issue.get("code") for issue in validation.get("errors", [])}
    assert not (error_codes & TEXT_ERROR_CODES)
    warning_codes = {issue.get("code") for issue in validation.get("warnings", [])}
    assert W2106_TEXT_BASELINE_NOT_ALIGNED not in warning_codes
    stats = validation.get("stats", {})
    assert int(stats.get("texts_detected", 0) or 0) > 0
