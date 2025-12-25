from __future__ import annotations

from pathlib import Path

from validators.validate import (
    E2010_TEXT_AS_PATH,
    E2011_TSPAN_ID_MISSING,
    E2012_MULTILINE_TSPAN_MISSING,
    E2013_DASHED_MISSING_DASHARRAY,
    validate_svg,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"


def _base_svg(curves_extra: str = "", text_extra: str = "") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="figure_root">
    <g id="g_axes" />
    <g id="g_curves">
      {curves_extra}
    </g>
    <g id="g_annotations" />
    <g id="g_text">
      <text id="t1" font-family="Arial">Hello</text>
      {text_extra}
    </g>
    <g id="g_markers" />
  </g>
</svg>
"""


def _write_svg(tmp_path: Path, svg_text: str) -> Path:
    svg_path = tmp_path / "input.svg"
    svg_path.write_text(svg_text)
    return svg_path


def test_tspan_id_missing(tmp_path: Path) -> None:
    text_extra = (
        "<text id=\"t2\" font-family=\"Arial\">"
        "<tspan>Line 1</tspan><tspan>Line 2</tspan>"
        "</text>"
    )
    report = validate_svg(_write_svg(tmp_path, _base_svg(text_extra=text_extra)), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2011_TSPAN_ID_MISSING in codes


def test_multiline_text_requires_tspan(tmp_path: Path) -> None:
    text_extra = "<text id=\"t3\" font-family=\"Arial\">Line 1\nLine 2</text>"
    report = validate_svg(_write_svg(tmp_path, _base_svg(text_extra=text_extra)), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2012_MULTILINE_TSPAN_MISSING in codes


def test_text_as_path_detected(tmp_path: Path) -> None:
    text_extra = "<path id=\"txt_label\" d=\"M 0 0 L 10 10\" />"
    report = validate_svg(_write_svg(tmp_path, _base_svg(text_extra=text_extra)), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2010_TEXT_AS_PATH in codes


def test_dashed_marker_requires_dasharray(tmp_path: Path) -> None:
    curves_extra = (
        "<line class=\"dashed\" x1=\"0\" y1=\"10\" x2=\"40\" y2=\"10\" "
        "stroke=\"#000\" stroke-width=\"1\" />"
    )
    report = validate_svg(_write_svg(tmp_path, _base_svg(curves_extra=curves_extra)), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2013_DASHED_MISSING_DASHARRAY in codes
