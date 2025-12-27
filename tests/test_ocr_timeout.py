from __future__ import annotations

import json
from pathlib import Path

from png2svg import convert_png
from png2svg.ocr import OcrTimeoutError
import png2svg.extractor as extractor

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"
THRESHOLDS = ROOT / "config" / "validator_thresholds.v1.yaml"


def test_convert_handles_ocr_timeout(monkeypatch, tmp_path: Path) -> None:
    def _timeout(*_args, **_kwargs):
        raise OcrTimeoutError("timeout", ["roi"], [])

    monkeypatch.setattr(extractor, "ocr_image", _timeout)
    monkeypatch.setattr(extractor, "has_tesseract", lambda: True)
    monkeypatch.setattr(extractor, "has_pytesseract", lambda: True)

    case_dir = ROOT / "datasets" / "regression_v0" / "cases" / "case_004_flow_simple"
    input_png = case_dir / "input.png"
    output_svg = tmp_path / "timeout.svg"
    debug_dir = tmp_path / "debug"

    result = convert_png(
        input_png,
        output_svg,
        debug_dir=debug_dir,
        topk=1,
        force_template="t_procedure_flow",
        contract_path=CONTRACT,
        thresholds_path=THRESHOLDS,
        quality_gate=False,
    )
    assert result["status"] in {"pass", "fail"}

    extract_report = (
        debug_dir / "candidates" / "t_procedure_flow" / "extract" / "extract_report.json"
    )
    payload = json.loads(extract_report.read_text())
    codes = [item.get("code") for item in payload.get("warnings", [])]
    assert "W4014_OCR_TIMEOUT" in codes
