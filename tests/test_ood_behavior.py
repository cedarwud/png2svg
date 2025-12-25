from __future__ import annotations

from pathlib import Path

import pytest

from png2svg import Png2SvgError, classify_png, convert_png


ROOT = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "samples" / "ood"


def test_classifier_marks_ood_unknown() -> None:
    ood_images = sorted(OOD_DIR.glob("*.png"))
    assert ood_images, "OOD samples are missing."
    for image_path in ood_images:
        result = classify_png(image_path)
        assert result["decision"] == "unknown"
        assert result["template_id"] == "unknown"
        assert "LOW_CONFIDENCE" in result["reason_codes"]


def test_convert_fails_on_ood(tmp_path: Path) -> None:
    image_path = sorted(OOD_DIR.glob("*.png"))[0]
    output_svg = tmp_path / "out.svg"
    debug_dir = tmp_path / "debug"
    with pytest.raises(Png2SvgError) as exc_info:
        convert_png(image_path, output_svg, debug_dir=debug_dir, topk=2)
    assert exc_info.value.code == "E5107_CLASSIFY_UNKNOWN"
    assert not output_svg.exists()
    assert (debug_dir / "classification.json").exists()
    assert (debug_dir / "classify" / "overlay.png").exists()
