from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from png2svg import convert_png
from validators.visual_diff import compute_visual_diff, rasterize_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = ROOT / "datasets" / "real_regression_v1" / "manifest.yaml"
CONTRACT = ROOT / "config" / "figure_contract.v1.yaml"
THRESHOLDS = ROOT / "config" / "validator_thresholds.v1.yaml"


def _load_manifest() -> list[dict[str, object]]:
    data = yaml.safe_load(REAL_MANIFEST.read_text())
    return data["cases"]


def test_real_regression_optional(tmp_path: Path) -> None:
    real_root = os.environ.get("REAL_PNG_DIR")
    if not real_root:
        pytest.skip("REAL_PNG_DIR is not set")
    if not REAL_MANIFEST.exists():
        pytest.skip("real regression manifest missing")

    cases = _load_manifest()[:3]
    for entry in cases:
        if not isinstance(entry, dict):
            continue
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            continue
        input_png = Path(real_root) / relative_path
        output_svg = tmp_path / f"{entry.get('id', 'case')}.svg"
        result = convert_png(
            input_png,
            output_svg,
            debug_dir=tmp_path / f"debug_{entry.get('id', 'case')}",
            topk=2,
            contract_path=CONTRACT,
            thresholds_path=THRESHOLDS,
        )
        assert result["status"] == "pass"
        selected = result.get("selected", {})
        validation = selected.get("validation", {})
        assert not validation.get("errors")

        expected_templates = []
        if "expected_template" in entry:
            expected_templates = [str(entry["expected_template"])]
        elif isinstance(entry.get("expected_templates"), list):
            expected_templates = [str(item) for item in entry["expected_templates"]]
        if expected_templates:
            assert result.get("selected_template") in expected_templates

        gates = entry.get("gates", {})
        max_rmse = float(gates["max_rmse"])
        max_bad_pixel_ratio = float(gates["max_bad_pixel_ratio"])
        pixel_tolerance = int(gates["pixel_tolerance"])

        output_png = tmp_path / f"{entry.get('id', 'case')}.png"
        rasterize_svg_to_png(output_svg, output_png)
        metrics = compute_visual_diff(output_png, input_png, pixel_tolerance)
        assert metrics.rmse <= max_rmse
        assert metrics.bad_pixel_ratio <= max_bad_pixel_ratio
