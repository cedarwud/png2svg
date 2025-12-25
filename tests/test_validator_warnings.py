from __future__ import annotations

from pathlib import Path

from validators.validate import (
    W2101_LINE_NOT_SNAPPED,
    W2102_DASHED_SIMULATED,
    validate_svg,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"


def _base_svg(curves_extra: str = "") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120">
  <g id="figure_root">
    <g id="g_axes" />
    <g id="g_curves">
      {curves_extra}
    </g>
    <g id="g_annotations" />
    <g id="g_text">
      <text id="t1" font-family="Arial">Hello</text>
    </g>
    <g id="g_markers" />
  </g>
</svg>
"""


def _write_svg(tmp_path: Path, svg_text: str) -> Path:
    svg_path = tmp_path / "input.svg"
    svg_path.write_text(svg_text)
    return svg_path


def test_warns_on_near_horizontal_line(tmp_path: Path) -> None:
    curves_extra = (
        "<line x1=\"0\" y1=\"10\" x2=\"50\" y2=\"10.3\" stroke=\"#000\" stroke-width=\"1\" />"
    )
    report = validate_svg(_write_svg(tmp_path, _base_svg(curves_extra=curves_extra)), CONTRACT)
    codes = {issue.code for issue in report.warnings}
    assert W2101_LINE_NOT_SNAPPED in codes


def test_warns_on_dashed_simulation(tmp_path: Path) -> None:
    segments = []
    for idx in range(6):
        x1 = idx * 8
        x2 = x1 + 4
        segments.append(
            f'<line x1="{x1}" y1="40" x2="{x2}" y2="40" stroke="#000" stroke-width="1" />'
        )
    curves_extra = "".join(segments)
    report = validate_svg(_write_svg(tmp_path, _base_svg(curves_extra=curves_extra)), CONTRACT)
    codes = {issue.code for issue in report.warnings}
    assert W2102_DASHED_SIMULATED in codes
