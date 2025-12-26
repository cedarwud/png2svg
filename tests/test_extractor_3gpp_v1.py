from __future__ import annotations

import json
from pathlib import Path

import pytest

from png2svg import extract_skeleton, render_svg

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "datasets" / "regression_hard_v1" / "cases" / "case_3gpp_fig1_like"
INPUT_PNG = CASE_DIR / "input.png"


def _panel_map(panel_bounds: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = {}
    for idx, panel in enumerate(panel_bounds):
        panel_id = str(panel.get("id") or f"panel_{idx}")
        mapping[panel_id] = {
            "x0": float(panel["x0"]),
            "x1": float(panel["x1"]),
            "y0": float(panel["y0"]),
            "y1": float(panel["y1"]),
        }
    return mapping


@pytest.mark.skipif(not INPUT_PNG.exists(), reason="hard case input.png missing")
def test_extract_3gpp_v1_outputs(tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug"
    params = extract_skeleton(INPUT_PNG, "t_3gpp_events_3panel", debug_dir=debug_dir)
    assert params.get("template") == "t_3gpp_events_3panel"

    extracted = params.get("extracted")
    assert isinstance(extracted, dict)

    panel_bounds = extracted.get("panel_bounds")
    assert isinstance(panel_bounds, list) and len(panel_bounds) == 3
    x0_values = [float(panel["x0"]) for panel in panel_bounds]
    assert x0_values == sorted(x0_values)

    panel_lookup = _panel_map(panel_bounds)

    t_positions = extracted.get("t_positions")
    assert isinstance(t_positions, list) and len(t_positions) == 3
    for item in t_positions:
        panel_id = str(item.get("panel_id"))
        bounds = panel_lookup.get(panel_id)
        assert bounds is not None
        t_start = float(item["t_start_x"])
        t_trigger = float(item["t_trigger_x"])
        assert t_start < t_trigger
        assert bounds["x0"] <= t_start <= bounds["x1"]
        assert bounds["x0"] <= t_trigger <= bounds["x1"]

    panel_axes = extracted.get("panel_axes")
    assert isinstance(panel_axes, list) and len(panel_axes) == 3
    for axes in panel_axes:
        panel_id = str(axes.get("panel_id"))
        bounds = panel_lookup.get(panel_id)
        assert bounds is not None
        y_axis = axes.get("y_axis")
        x_axis = axes.get("x_axis")
        assert isinstance(y_axis, dict)
        assert isinstance(x_axis, dict)
        y_axis_x = float(y_axis["x"])
        x_axis_y = float(x_axis["y"])
        assert bounds["x0"] <= y_axis_x <= bounds["x1"]
        assert bounds["y0"] <= x_axis_y <= bounds["y1"]

    assert float(params["t_start_ratio"]) < float(params["t_trigger_ratio"])

    assert (debug_dir / "preprocessed.png").exists()
    assert (debug_dir / "overlays.png").exists()
    assert (debug_dir / "extracted.json").exists()

    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True))
    output_svg = tmp_path / "out.svg"
    render_svg(INPUT_PNG, params_path, output_svg)
    assert output_svg.exists()
