from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from png2svg import classify_png, convert_png
from png2svg.ocr import has_pytesseract, has_tesseract
from validators.config import load_thresholds

ROOT = Path(__file__).resolve().parents[1]
HARD_CASE_DIR = ROOT / "datasets" / "regression_hard_v1" / "cases" / "case_3gpp_fig1_like"
INPUT_PNG = HARD_CASE_DIR / "input.png"
THRESHOLDS = ROOT / "config" / "validator_thresholds.v1.yaml"
TEMPLATE_ID = "t_3gpp_events_3panel"


def _has_rasterizer() -> bool:
    if shutil.which("resvg"):
        return True
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_available() -> bool:
    backend = os.environ.get("PNG2SVG_OCR_BACKEND", "auto").lower()
    if backend == "none":
        return False
    if backend == "pytesseract":
        return has_pytesseract()
    if backend == "tesseract":
        return has_tesseract()
    return has_pytesseract() or has_tesseract()


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _gate_thresholds() -> dict[str, float | int]:
    thresholds = load_thresholds(THRESHOLDS)
    gate = thresholds.get("quality_gate_hard") or thresholds.get("quality_gate") or {}
    return {
        "rmse_max": float(gate.get("rmse_max", 1.0)),
        "bad_pixel_ratio_max": float(gate.get("bad_pixel_ratio_max", 1.0)),
        "pixel_tolerance": int(gate.get("pixel_tolerance", 10)),
    }


def _scale_to_canvas(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), resample=Image.BICUBIC)
    if new_w >= width or new_h >= height:
        left = max(int((new_w - width) / 2), 0)
        top = max(int((new_h - height) / 2), 0)
        return resized.crop((left, top, left + width, top + height))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    left = int((width - new_w) / 2)
    top = int((height - new_h) / 2)
    canvas.paste(resized, (left, top))
    return canvas


def _shift(image: Image.Image, dx: int, dy: int) -> Image.Image:
    width, height = image.size
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    canvas.paste(image, (dx, dy))
    return canvas


def _brightness_contrast(image: Image.Image, brightness: float, contrast: float) -> Image.Image:
    rgba = image.convert("RGBA")
    rgb = rgba.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    out = rgb.convert("RGBA")
    out.putalpha(rgba.split()[3])
    return out


def _add_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb + noise, 0.0, 255.0).astype(np.uint8)
    out = np.concatenate([rgb, alpha], axis=2)
    return Image.fromarray(out, mode="RGBA")


def _variants(base: Image.Image) -> list[tuple[str, Image.Image]]:
    return [
        ("scale_down", _scale_to_canvas(base, 0.98)),
        ("scale_up", _scale_to_canvas(base, 1.02)),
        ("shift", _shift(base, 2, -2)),
        ("brightness", _brightness_contrast(base, 1.02, 0.98)),
        ("noise", _add_noise(base, 1.5, _stable_seed("noise"))),
    ]


def _assert_top2_contains(result: dict[str, object], template_id: str) -> None:
    candidates = result.get("candidate_templates", [])
    top_two = [item["template_id"] for item in candidates[:2]]
    assert template_id in top_two


def test_hard_case_base_convert(tmp_path: Path) -> None:
    assert INPUT_PNG.exists()
    if not _has_rasterizer():
        pytest.skip("Rasterizer unavailable (resvg/cairosvg required).")
    if not _ocr_available():
        pytest.skip("OCR backend unavailable.")
    thresholds = _gate_thresholds()
    classification = classify_png(INPUT_PNG)
    _assert_top2_contains(classification, TEMPLATE_ID)
    output_svg = tmp_path / "base.svg"
    result = convert_png(
        INPUT_PNG,
        output_svg,
        debug_dir=tmp_path / "debug_base",
        topk=2,
        force_template=TEMPLATE_ID,
        gate_rmse_max=thresholds["rmse_max"],
        gate_bad_pixel_max=thresholds["bad_pixel_ratio_max"],
        gate_pixel_tolerance=thresholds["pixel_tolerance"],
    )
    assert result["status"] == "pass"
    gate = result.get("selected", {}).get("quality_gate", {})
    assert gate.get("status") == "pass"


def test_hard_case_metamorphic_variants(tmp_path: Path) -> None:
    assert INPUT_PNG.exists()
    if not _has_rasterizer():
        pytest.skip("Rasterizer unavailable (resvg/cairosvg required).")
    if not _ocr_available():
        pytest.skip("OCR backend unavailable.")
    thresholds = _gate_thresholds()
    with Image.open(INPUT_PNG) as base:
        base_rgba = base.convert("RGBA")
        for name, variant in _variants(base_rgba):
            variant_path = tmp_path / f"variant_{name}.png"
            variant.save(variant_path)
            classification = classify_png(variant_path)
            _assert_top2_contains(classification, TEMPLATE_ID)
            output_svg = tmp_path / f"out_{name}.svg"
            result = convert_png(
                variant_path,
                output_svg,
                debug_dir=tmp_path / f"debug_{name}",
                topk=2,
                force_template=TEMPLATE_ID,
                gate_rmse_max=thresholds["rmse_max"],
                gate_bad_pixel_max=thresholds["bad_pixel_ratio_max"],
                gate_pixel_tolerance=thresholds["pixel_tolerance"],
            )
            assert result["status"] == "pass"
            gate = result.get("selected", {}).get("quality_gate", {})
            assert gate.get("status") == "pass"
