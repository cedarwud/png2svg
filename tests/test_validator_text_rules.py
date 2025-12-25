from __future__ import annotations

from pathlib import Path

from validators.validate import (
    E2010_TEXT_AS_PATH,
    E2011_TSPAN_ID_MISSING,
    E2012_MULTILINE_TSPAN_MISSING,
    E2013_DASHED_MISSING_DASHARRAY,
    E2014_TEXT_COUNT_LOW,
    E2015_TEXT_FALLBACK_MISSING,
    E2016_TEXT_ANCHOR_INVALID,
    E2017_TEXT_OUTLINE_DETECTED,
    W2106_TEXT_BASELINE_NOT_ALIGNED,
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
      <text id="t1" font-family="Arial, sans-serif" text-anchor="start">Hello</text>
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
        "<text id=\"t2\" font-family=\"Arial, sans-serif\" text-anchor=\"start\">"
        "<tspan>Line 1</tspan><tspan>Line 2</tspan>"
        "</text>"
    )
    report = validate_svg(_write_svg(tmp_path, _base_svg(text_extra=text_extra)), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2011_TSPAN_ID_MISSING in codes


def test_multiline_text_requires_tspan(tmp_path: Path) -> None:
    text_extra = "<text id=\"t3\" font-family=\"Arial, sans-serif\" text-anchor=\"start\">Line 1\nLine 2</text>"
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


def test_font_family_requires_fallback(tmp_path: Path) -> None:
    text_extra = "<text id=\"t2\" font-family=\"Arial\" text-anchor=\"start\">Label</text>"
    report = validate_svg(_write_svg(tmp_path, _base_svg(text_extra=text_extra)), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2015_TEXT_FALLBACK_MISSING in codes


def test_text_anchor_required(tmp_path: Path) -> None:
    text_extra = "<text id=\"t2\" font-family=\"Arial, sans-serif\">Label</text>"
    report = validate_svg(_write_svg(tmp_path, _base_svg(text_extra=text_extra)), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2016_TEXT_ANCHOR_INVALID in codes


def test_text_count_minimum(tmp_path: Path) -> None:
    report = validate_svg(
        _write_svg(tmp_path, _base_svg()),
        CONTRACT,
        text_expectations={"texts_detected": 4},
    )
    codes = {issue.code for issue in report.errors}
    assert E2014_TEXT_COUNT_LOW in codes


def test_text_outline_detected(tmp_path: Path) -> None:
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="figure_root">
    <g id="g_axes" />
    <g id="g_curves">
      <path d="M 0 0 L 10 10" />
      <path d="M 0 10 L 10 20" />
      <path d="M 0 20 L 10 30" />
      <path d="M 0 30 L 10 40" />
      <path d="M 0 40 L 10 50" />
      <path d="M 0 50 L 10 60" />
      <path d="M 0 60 L 10 70" />
      <path d="M 0 70 L 10 80" />
    </g>
    <g id="g_annotations" />
    <g id="g_text" />
    <g id="g_markers" />
  </g>
</svg>
"""
    report = validate_svg(
        _write_svg(tmp_path, svg_text),
        CONTRACT,
        text_expectations={"texts_detected": 3},
    )
    codes = {issue.code for issue in report.errors}
    assert E2017_TEXT_OUTLINE_DETECTED in codes


def test_text_baseline_warning(tmp_path: Path) -> None:
    text_extra = (
        "<text id=\"t2\" font-family=\"Arial, sans-serif\" text-anchor=\"start\" y=\"20\" x=\"10\">A</text>"
        "<text id=\"t3\" font-family=\"Arial, sans-serif\" text-anchor=\"start\" y=\"20.6\" x=\"40\">B</text>"
    )
    report = validate_svg(_write_svg(tmp_path, _base_svg(text_extra=text_extra)), CONTRACT)
    codes = {issue.code for issue in report.warnings}
    assert W2106_TEXT_BASELINE_NOT_ALIGNED in codes
