from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from png2svg import extract_skeleton, render_svg
from validators.svg_checks import count_path_commands, local_name, parse_style

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "datasets" / "regression_hard_v1" / "cases" / "case_3gpp_fig1_like"
INPUT_PNG = CASE_DIR / "input.png"


def _find_group(root: ET.Element, group_id: str) -> ET.Element | None:
    for node in root.iter():
        if local_name(node.tag) == "g" and node.get("id") == group_id:
            return node
    return None


def _stroke_dash(node: ET.Element) -> tuple[str | None, str | None]:
    style = parse_style(node.get("style", ""))
    stroke = node.get("stroke") or style.get("stroke")
    dash = node.get("stroke-dasharray") or style.get("stroke-dasharray")
    return stroke, dash


@pytest.mark.skipif(not INPUT_PNG.exists(), reason="hard case input.png missing")
def test_3gpp_curve_extraction_paths(tmp_path: Path) -> None:
    params = extract_skeleton(INPUT_PNG, "t_3gpp_events_3panel")
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True))
    output_svg = tmp_path / "out.svg"
    render_svg(INPUT_PNG, params_path, output_svg)

    root = ET.parse(output_svg).getroot()
    curves_group = _find_group(root, "g_curves")
    assert curves_group is not None
    paths = [node for node in curves_group if local_name(node.tag) == "path"]
    assert len(paths) >= 2

    max_commands = 0
    has_dashed_orange = False
    for path in paths:
        d = path.get("d") or ""
        max_commands = max(max_commands, count_path_commands(d))
        stroke, dash = _stroke_dash(path)
        if stroke and stroke.lower() == "#dd6b20" and dash:
            has_dashed_orange = True
    assert has_dashed_orange
    assert max_commands <= 20
