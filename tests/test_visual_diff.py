from __future__ import annotations

from pathlib import Path

from PIL import Image

from validators.visual_diff import compute_visual_diff


def _write_image(path: Path, color: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", (8, 8), color=color)
    image.save(path, format="PNG")


def test_visual_diff_identical_images(tmp_path: Path) -> None:
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    _write_image(left, (10, 20, 30, 255))
    _write_image(right, (10, 20, 30, 255))

    metrics = compute_visual_diff(left, right, pixel_tolerance=0)

    assert metrics.rmse == 0.0
    assert metrics.bad_pixel_ratio == 0.0
