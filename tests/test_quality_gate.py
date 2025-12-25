from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from png2svg import Png2SvgError, convert_png


ROOT = Path(__file__).resolve().parents[1]


def test_quality_gate_passes_regression_sample(tmp_path: Path) -> None:
    case_dir = ROOT / "datasets" / "regression_v0" / "cases" / "case_007_lineplot_basic"
    input_png = case_dir / "expected.png"
    params = json.loads((case_dir / "params.json").read_text())
    template = str(params["template"])
    output_svg = tmp_path / "out.svg"
    debug_dir = tmp_path / "debug"

    result = convert_png(
        input_png,
        output_svg,
        debug_dir=debug_dir,
        topk=1,
        force_template=template,
        gate_rmse_max=0.2,
        gate_bad_pixel_max=0.07,
    )
    selected = result["selected"]
    gate = selected["quality_gate"]
    assert gate["status"] == "pass"
    assert gate["rmse"] <= gate["rmse_max"]
    assert gate["bad_pixel_ratio"] <= gate["bad_pixel_ratio_max"]
    assert (debug_dir / "candidates" / template / "gate_report.json").exists()


def test_quality_gate_fails_on_ood(tmp_path: Path) -> None:
    ood_dir = ROOT / "samples" / "ood"
    image_path = sorted(ood_dir.glob("*.png"))[0]
    output_svg = tmp_path / "out.svg"
    debug_dir = tmp_path / "debug"

    with pytest.raises(Png2SvgError) as exc_info:
        convert_png(
            image_path,
            output_svg,
            debug_dir=debug_dir,
            topk=1,
            force_template="t_performance_lineplot",
            gate_rmse_max=0.1,
            gate_bad_pixel_max=0.1,
        )
    assert exc_info.value.code == "E5108_QUALITY_GATE_FAILED"
    gate_report = debug_dir / "candidates" / "t_performance_lineplot" / "gate_report.json"
    assert gate_report.exists()
    gate_payload = json.loads(gate_report.read_text())
    assert gate_payload["status"] == "fail"


def test_quality_gate_selects_best_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_png = tmp_path / "input.png"
    Image.new("RGBA", (10, 10), (255, 255, 255, 255)).save(input_png)
    output_svg = tmp_path / "out.svg"

    import png2svg.convert as convert_mod

    def fake_classify(_: Path, debug_dir: Path | None = None, thresholds_path: Path | None = None) -> dict:
        return {
            "template_id": "t_procedure_flow",
            "decision": "known",
            "reason_codes": [],
            "confidence": 0.9,
            "candidate_templates": [
                {"template_id": "t_performance_lineplot", "score": 0.1},
                {"template_id": "t_procedure_flow", "score": 2.0},
            ],
            "image_meta": {"width": 10, "height": 10},
            "features_summary": {},
        }

    def fake_extract(_: Path, template_id: str, debug_dir: Path | None = None) -> dict:
        return {
            "template": template_id,
            "canvas": {"width": 10, "height": 10},
            "texts": [],
            "geometry": {"lines": [], "rects": [], "markers": []},
        }

    def fake_render(_: Path, params_path: Path, svg_path: Path) -> None:
        data = json.loads(params_path.read_text())
        template_id = data.get("template")
        fill = "white" if template_id == "t_performance_lineplot" else "black"
        svg_path.write_text(
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">"
            f"<rect x=\"0\" y=\"0\" width=\"10\" height=\"10\" fill=\"{fill}\" /></svg>"
        )

    class DummyReport:
        def __init__(self) -> None:
            self.errors: list[dict] = []
            self.warnings: list[dict] = []
            self.status = "pass"

        def to_dict(self) -> dict:
            return {"status": "pass", "errors": [], "warnings": [], "stats": {}}

    def fake_validate(*args: object, **kwargs: object) -> DummyReport:
        return DummyReport()

    monkeypatch.setattr(convert_mod, "classify_png", fake_classify)
    monkeypatch.setattr(convert_mod, "extract_skeleton", fake_extract)
    monkeypatch.setattr(convert_mod, "render_svg", fake_render)
    monkeypatch.setattr(convert_mod, "validate_svg", fake_validate)

    result = convert_png(
        input_png,
        output_svg,
        debug_dir=tmp_path / "debug",
        topk=2,
        gate_rmse_max=0.01,
        gate_bad_pixel_max=0.01,
        gate_pixel_tolerance=0,
    )
    assert result["selected_template"] == "t_performance_lineplot"
    assert output_svg.exists()
