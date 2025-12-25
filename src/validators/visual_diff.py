from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualDiffMetrics:
    rmse: float
    bad_pixel_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "bad_pixel_ratio": self.bad_pixel_ratio}


class RasterizeError(RuntimeError):
    pass


class DiffError(RuntimeError):
    pass


def _try_resvg(svg_path: Path, png_path: Path) -> bool:
    resvg = shutil.which("resvg")
    if not resvg:
        return False
    commands = [
        [resvg, str(svg_path), str(png_path)],
        [resvg, str(svg_path), "-o", str(png_path)],
        [resvg, str(svg_path), "--output", str(png_path)],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if png_path.exists():
                return True
        except subprocess.CalledProcessError:
            continue
    return False


def rasterize_svg_to_png(svg_path: Path, png_path: Path) -> str:
    if _try_resvg(svg_path, png_path):
        return "resvg"
    try:
        import cairosvg
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RasterizeError(
            "cairosvg not installed and resvg not available"
        ) from exc
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    except Exception as exc:  # noqa: BLE001
        raise RasterizeError(f"cairosvg failed: {exc}") from exc
    return "cairosvg"


def compute_visual_diff(
    actual_png: Path,
    expected_png: Path,
    pixel_tolerance: int,
) -> VisualDiffMetrics:
    with Image.open(actual_png) as img_a_raw, Image.open(expected_png) as img_b_raw:
        img_a = img_a_raw.convert("RGBA")
        img_b = img_b_raw.convert("RGBA")
    if img_a.size != img_b.size:
        raise DiffError(f"Image size mismatch: {img_a.size} vs {img_b.size}")
    arr_a = np.asarray(img_a, dtype=np.float32)
    arr_b = np.asarray(img_b, dtype=np.float32)
    diff = arr_a[:, :, :3] - arr_b[:, :, :3]
    rmse = float(np.sqrt(np.mean((diff / 255.0) ** 2)))
    bad_pixels = np.any(np.abs(diff) > pixel_tolerance, axis=2)
    bad_pixel_ratio = float(np.mean(bad_pixels))
    return VisualDiffMetrics(rmse=rmse, bad_pixel_ratio=bad_pixel_ratio)


def write_diff_image(
    actual_png: Path,
    expected_png: Path,
    output_png: Path,
    pixel_tolerance: int,
) -> None:
    with Image.open(actual_png) as img_a_raw, Image.open(expected_png) as img_b_raw:
        img_a = img_a_raw.convert("RGBA")
        img_b = img_b_raw.convert("RGBA")
    if img_a.size != img_b.size:
        raise DiffError(f"Image size mismatch: {img_a.size} vs {img_b.size}")
    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(img_b, dtype=np.int16)
    diff = np.abs(arr_a[:, :, :3] - arr_b[:, :, :3])
    mask = np.any(diff > pixel_tolerance, axis=2)
    heat = np.zeros_like(arr_a[:, :, :3], dtype=np.uint8)
    heat[:, :, 0] = np.clip(diff.max(axis=2), 0, 255).astype(np.uint8)
    heat[mask, 1] = 0
    heat[mask, 2] = 0
    out = Image.fromarray(heat, mode="RGB")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_png)
