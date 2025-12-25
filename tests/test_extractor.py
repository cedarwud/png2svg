from __future__ import annotations

import json
from pathlib import Path

import yaml

from png2svg import extract_skeleton, render_svg
from validators.validate import validate_svg
from validators.visual_diff import rasterize_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "regression_v0" / "manifest.yaml"
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"


def _load_cases() -> list[dict[str, str]]:
    data = yaml.safe_load(MANIFEST.read_text())
    return data["cases"]


def _case_dir(entry: dict[str, str]) -> Path:
    return ROOT / "datasets" / "regression_v0" / entry["dir"]


def _case_template(case_dir: Path) -> str:
    params = json.loads((case_dir / "params.json").read_text())
    return str(params["template"])


def _expected_png(case_dir: Path, tmp_path: Path) -> Path:
    png_path = case_dir / "expected.png"
    if png_path.exists():
        return png_path
    svg_path = case_dir / "expected.svg"
    out_path = tmp_path / f"{case_dir.name}_expected.png"
    rasterize_svg_to_png(svg_path, out_path)
    return out_path


def _assert_schema(template_id: str, params: dict[str, object]) -> None:
    assert params.get("template") == template_id
    canvas = params.get("canvas")
    assert isinstance(canvas, dict)
    assert "width" in canvas and "height" in canvas
    if template_id == "t_3gpp_events_3panel":
        panels = params.get("panels")
        assert isinstance(panels, list) and len(panels) == 3
        assert "t_start_ratio" in params and "t_trigger_ratio" in params
        curves = params.get("curves")
        assert isinstance(curves, dict)
    elif template_id == "t_performance_lineplot":
        axes = params.get("axes")
        assert isinstance(axes, dict)
        series = params.get("series")
        assert isinstance(series, list) and len(series) >= 1
    elif template_id == "t_procedure_flow":
        nodes = params.get("nodes")
        assert isinstance(nodes, list) and len(nodes) >= 1
    else:
        raise AssertionError(f"Unknown template: {template_id}")


def _write_params(tmp_path: Path, name: str, params: dict[str, object]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(params, indent=2, sort_keys=True))
    return path


def _cases_for_template(template_id: str) -> list[Path]:
    entries = _load_cases()
    case_dirs = []
    for entry in entries:
        case_dir = _case_dir(entry)
        if _case_template(case_dir) == template_id:
            case_dirs.append(case_dir)
    return case_dirs


def test_extract_skeleton_renders_and_validates(tmp_path: Path) -> None:
    templates = [
        "t_3gpp_events_3panel",
        "t_procedure_flow",
        "t_performance_lineplot",
    ]
    for template_id in templates:
        case_dirs = _cases_for_template(template_id)[:2]
        for case_dir in case_dirs:
            image_path = _expected_png(case_dir, tmp_path)
            params = extract_skeleton(image_path, template_id)
            _assert_schema(template_id, params)
            params_path = _write_params(tmp_path, case_dir.name, params)
            output_svg = tmp_path / f"{case_dir.name}.svg"
            render_svg(image_path, params_path, output_svg)
            report = validate_svg(output_svg, CONTRACT)
            assert report.status == "pass"


def test_extract_deterministic(tmp_path: Path) -> None:
    case_dir = _cases_for_template("t_3gpp_events_3panel")[0]
    image_path = _expected_png(case_dir, tmp_path)
    params_a = extract_skeleton(image_path, "t_3gpp_events_3panel")
    params_b = extract_skeleton(image_path, "t_3gpp_events_3panel")
    assert params_a == params_b


def test_extract_auto_template(tmp_path: Path) -> None:
    case_dir = _cases_for_template("t_performance_lineplot")[0]
    image_path = _expected_png(case_dir, tmp_path)
    params = extract_skeleton(image_path, "auto")
    assert params["template"] == "t_performance_lineplot"
