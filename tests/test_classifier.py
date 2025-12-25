from __future__ import annotations

import json
from pathlib import Path

import yaml

from png2svg import classify_png
from validators.visual_diff import rasterize_svg_to_png


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "regression_v0" / "manifest.yaml"


def _load_manifest() -> list[dict[str, str]]:
    data = yaml.safe_load(MANIFEST.read_text())
    return data["cases"]


def _case_dir(entry: dict[str, str]) -> Path:
    return ROOT / "datasets" / "regression_v0" / entry["dir"]


def _expected_template(case_dir: Path) -> str:
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


def test_classifier_regression_cases(tmp_path: Path) -> None:
    for entry in _load_manifest():
        case_dir = _case_dir(entry)
        expected_template = _expected_template(case_dir)
        image_path = _expected_png(case_dir, tmp_path)
        result = classify_png(image_path)
        top_two = [item["template_id"] for item in result["candidate_templates"][:2]]
        assert expected_template in top_two


def test_classifier_deterministic(tmp_path: Path) -> None:
    entry = _load_manifest()[0]
    case_dir = _case_dir(entry)
    image_path = _expected_png(case_dir, tmp_path)
    result_a = classify_png(image_path)
    result_b = classify_png(image_path)
    assert result_a == result_b


def test_classifier_debug_outputs(tmp_path: Path) -> None:
    entry = _load_manifest()[0]
    case_dir = _case_dir(entry)
    image_path = _expected_png(case_dir, tmp_path)
    debug_dir = tmp_path / "debug"
    classify_png(image_path, debug_dir=debug_dir)
    assert (debug_dir / "overlay.png").exists()
    assert (debug_dir / "features.json").exists()
