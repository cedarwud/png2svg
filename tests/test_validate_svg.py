from __future__ import annotations

from pathlib import Path

from validators.validate import (
    E2001_FORBIDDEN_ELEMENT,
    E2003_MISSING_GROUP,
    E2004_BAD_FONT_FAMILY,
    validate_svg,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"


def _build_svg(font_family: str = "Arial", extra: str = "", missing_group: str | None = None) -> str:
    groups = ["g_axes", "g_curves", "g_annotations", "g_text", "g_markers"]
    if missing_group and missing_group in groups:
        groups.remove(missing_group)
    group_markup = "\n".join(f'      <g id="{group}" />' for group in groups if group != "g_text")
    text_markup = ""
    if "g_text" in groups:
        text_markup = (
            f'      <g id="g_text"><text id="t1" font-family="{font_family}" '
            f'fill="#000">Hello</text></g>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="figure_root">
{group_markup}
{text_markup}
  </g>
  {extra}
</svg>
"""


def _write_svg(tmp_path: Path, svg_text: str) -> Path:
    svg_path = tmp_path / "input.svg"
    svg_path.write_text(svg_text)
    return svg_path


def test_forbids_image_element(tmp_path: Path) -> None:
    svg_text = _build_svg(
        extra='<image href="x.png" x="0" y="0" width="10" height="10" />'
    )
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2001_FORBIDDEN_ELEMENT in codes


def test_missing_required_group(tmp_path: Path) -> None:
    svg_text = _build_svg(missing_group="g_markers")
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2003_MISSING_GROUP in codes


def test_forbids_gradient_element(tmp_path: Path) -> None:
    svg_text = _build_svg(extra="<defs><linearGradient id=\"g1\"/></defs>")
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2001_FORBIDDEN_ELEMENT in codes


def test_bad_font_family(tmp_path: Path) -> None:
    svg_text = _build_svg(font_family="Comic Sans MS")
    report = validate_svg(_write_svg(tmp_path, svg_text), CONTRACT)
    codes = {issue.code for issue in report.errors}
    assert E2004_BAD_FONT_FAMILY in codes
