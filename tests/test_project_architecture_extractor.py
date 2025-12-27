from __future__ import annotations

from pathlib import Path

import pytest

from png2svg import extract_skeleton
from png2svg.ocr import has_pytesseract, has_tesseract


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "datasets" / "regression_v0" / "cases" / "case_018_project_architecture_v1"


def test_project_architecture_ocr_extracts_text() -> None:
    if not (has_pytesseract() or has_tesseract()):
        pytest.skip("OCR backend not available")
    input_png = CASE_DIR / "input.png"
    params = extract_skeleton(input_png, "t_project_architecture_v1")
    extracted = params.get("extracted", {})
    assert isinstance(extracted, dict)
    meta = extracted.get("project_architecture", {})
    assert isinstance(meta, dict)
    assert meta.get("ocr_used") is True
    fields = meta.get("fields_from_ocr")
    assert isinstance(fields, list) and fields
