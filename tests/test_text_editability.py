from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from png2svg import render_svg
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
    validate_svg,
)
from validators.svg_checks import local_name


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"
THRESHOLDS = ROOT / "config" / "validator_thresholds.v1.yaml"
DATASET = ROOT / "datasets" / "regression_v0" / "cases"


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


CASES = [
    ("case_006_flow_multiline", 6),
    ("case_009_lineplot_wide", 12),
    ("case_012_lineplot_dense", 15),
]


def _count_text_elements(svg_path: Path) -> int:
    root = ET.parse(svg_path).getroot()
    return sum(1 for node in root.iter() if local_name(node.tag) == "text")


@pytest.mark.parametrize("case_id,min_count", CASES)
def test_text_editability_rules(case_id: str, min_count: int, tmp_path: Path) -> None:
    case_dir = DATASET / case_id
    params_path = case_dir / "params.json"
    input_png = case_dir / "input.png"
    assert params_path.exists()
    output_svg = tmp_path / f"{case_id}.svg"
    render_svg(input_png, params_path, output_svg)
    report = validate_svg(output_svg, CONTRACT, THRESHOLDS)
    error_codes = {issue.code for issue in report.errors}
    assert not (error_codes & TEXT_ERROR_CODES)
    assert _count_text_elements(output_svg) >= min_count
