from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from png2svg.ocr import has_tesseract, ocr_image


def _tesseract_available() -> bool:
    if os.environ.get("HAS_TESSERACT") == "1":
        return True
    return has_tesseract()


def _load_font() -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, 48)
    return ImageFont.load_default()


@pytest.mark.skipif(not _tesseract_available(), reason="tesseract not available")
def test_ocr_detects_keyword(tmp_path: Path) -> None:
    image = Image.new("RGB", (320, 120), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    draw.text((10, 20), "HELLO 123", fill="black", font=font)
    image = image.resize((960, 360), resample=Image.NEAREST)
    rois = [{"id": "full", "x": 0, "y": 0, "width": image.width, "height": image.height}]
    results = ocr_image(image, backend="tesseract", rois=rois)
    text = " ".join(item["text"] for item in results).upper()
    assert "HELLO" in text
