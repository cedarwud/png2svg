from __future__ import annotations

from pathlib import Path

from validators.validate import (
    E2003_MISSING_GROUP,
    E2005_TOO_MANY_COLORS,
    E2006_BAD_STROKE_WIDTH,
    E2007_PATH_TOO_COMPLEX,
    E2009_TEXT_ID_MISSING,
    validate_svg,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"


def _base_svg(extra: str = "") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="figure_root">
    <g id="g_axes" />
    <g id="g_curves" />
    <g id="g_annotations" />
    <g id="g_text">
      <text id="t1" font-family="Arial" fill="#000000">Label</text>
    </g>
    <g id="g_markers" />
  </g>
  {extra}
</svg>
"""


def _write_svg(tmp_path: Path, svg_text: str) -> Path:
    svg_path = tmp_path / "input.svg"
    svg_path.write_text(svg_text)
    return svg_path


def test_missing_figure_root_group(tmp_path: Path) -> None:
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="g_axes" />
</svg>
"""
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2003_MISSING_GROUP in codes


def test_missing_text_id(tmp_path: Path) -> None:
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="figure_root">
    <g id="g_axes" />
    <g id="g_curves" />
    <g id="g_annotations" />
    <g id="g_text">
      <text font-family="Arial">NoId</text>
    </g>
    <g id="g_markers" />
  </g>
</svg>
"""
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2009_TEXT_ID_MISSING in codes


def test_stroke_width_tolerance(tmp_path: Path) -> None:
    svg_text = _base_svg(
        '<line x1="0" y1="0" x2="10" y2="0" stroke="#000000" stroke-width="2.1" />'
    )
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    assert report.status == "pass"

    svg_text = _base_svg(
        '<line x1="0" y1="0" x2="10" y2="0" stroke="#000000" stroke-width="2.3" />'
    )
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2006_BAD_STROKE_WIDTH in codes


def test_color_count(tmp_path: Path) -> None:
    colors = ["#000000", "#111111", "#222222", "#333333", "#444444", "#555555", "#666666"]
    lines = [
        f'<line x1="{i}" y1="0" x2="{i}" y2="10" stroke="{color}" stroke-width="1" />'
        for i, color in enumerate(colors)
    ]
    svg_text = _base_svg("".join(lines))
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2005_TOO_MANY_COLORS in codes


def test_path_complexity(tmp_path: Path) -> None:
    commands = ["M 0 0"] + [f"L {i} {i}" for i in range(70)]
    path_d = " ".join(commands)
    svg_text = _base_svg(f'<path id="p1" d="{path_d}" stroke="#000" fill="none" />')
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2007_PATH_TOO_COMPLEX in codes
