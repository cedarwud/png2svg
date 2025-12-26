from __future__ import annotations

from pathlib import Path

from png2svg import render_svg
from validators.validate import validate_svg


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"


def test_sample_params_render_and_validate(tmp_path: Path) -> None:
    input_png = SAMPLES / "input.png"
    assert input_png.exists()
    sample_files = [
        SAMPLES / "t_3gpp_events_3panel.json",
        SAMPLES / "t_procedure_flow.json",
        SAMPLES / "t_performance_lineplot.json",
        SAMPLES / "t_project_architecture_v1.json",
    ]
    for sample in sample_files:
        output_svg = tmp_path / f"{sample.stem}.svg"
        render_svg(input_png, sample, output_svg)
        report = validate_svg(output_svg, CONTRACT)
        assert report.status == "pass"
