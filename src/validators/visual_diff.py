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


def _flatten_png_background(png_path: Path, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    try:
        with Image.open(png_path) as image:
            if image.mode not in {"RGBA", "LA"} and not (
                image.mode == "P" and "transparency" in image.info
            ):
                return
            rgba = image.convert("RGBA")
    except FileNotFoundError:
        return
    base = Image.new("RGBA", rgba.size, (*color, 255))
    base.alpha_composite(rgba)
    base.convert("RGB").save(png_path)


def _flatten_image(image: Image.Image, color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, (*color, 255))
        base.alpha_composite(rgba)
        return base.convert("RGB")
    return image.convert("RGB")


def _diff_metrics(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    pixel_tolerance: int,
) -> tuple[float, float]:
    diff = arr_a - arr_b
    rmse = float(np.sqrt(np.mean((diff / 255.0) ** 2)))
    bad_pixels = np.any(np.abs(diff) > pixel_tolerance, axis=2)
    bad_pixel_ratio = float(np.mean(bad_pixels))
    return rmse, bad_pixel_ratio


def _choose_best_diff(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    pixel_tolerance: int,
    invert_threshold: float = 120.0,
) -> tuple[np.ndarray, float, float]:
    rmse, bad_pixel_ratio = _diff_metrics(arr_a, arr_b, pixel_tolerance)
    mean_expected = float(arr_b.mean())
    if mean_expected < invert_threshold:
        inverted = 255.0 - arr_b
        rmse_inv, bad_pixel_ratio_inv = _diff_metrics(arr_a, inverted, pixel_tolerance)
        if (rmse_inv, bad_pixel_ratio_inv) < (rmse, bad_pixel_ratio):
            return inverted, rmse_inv, bad_pixel_ratio_inv
    return arr_b, rmse, bad_pixel_ratio


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
        _flatten_png_background(png_path)
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
    _flatten_png_background(png_path)
    return "cairosvg"


def compute_visual_diff(
    actual_png: Path,
    expected_png: Path,
    pixel_tolerance: int,
) -> VisualDiffMetrics:
    with Image.open(actual_png) as img_a_raw, Image.open(expected_png) as img_b_raw:
        img_a = _flatten_image(img_a_raw)
        img_b = _flatten_image(img_b_raw)
    if img_a.size != img_b.size:
        raise DiffError(f"Image size mismatch: {img_a.size} vs {img_b.size}")
    arr_a = np.asarray(img_a, dtype=np.float32)
    arr_b = np.asarray(img_b, dtype=np.float32)
    _, rmse, bad_pixel_ratio = _choose_best_diff(arr_a, arr_b, pixel_tolerance)
    return VisualDiffMetrics(rmse=rmse, bad_pixel_ratio=bad_pixel_ratio)


def write_diff_image(
    actual_png: Path,
    expected_png: Path,
    output_png: Path,
    pixel_tolerance: int,
) -> None:
    with Image.open(actual_png) as img_a_raw, Image.open(expected_png) as img_b_raw:
        img_a = _flatten_image(img_a_raw)
        img_b = _flatten_image(img_b_raw)
    if img_a.size != img_b.size:
        raise DiffError(f"Image size mismatch: {img_a.size} vs {img_b.size}")
    arr_a = np.asarray(img_a, dtype=np.float32)
    arr_b = np.asarray(img_b, dtype=np.float32)
    arr_b, _, _ = _choose_best_diff(arr_a, arr_b, pixel_tolerance)
    diff = np.abs(arr_a - arr_b)
    mask = np.any(diff > pixel_tolerance, axis=2)
    heat = np.zeros_like(arr_a[:, :, :3], dtype=np.uint8)
    heat[:, :, 0] = np.clip(diff.max(axis=2), 0, 255).astype(np.uint8)
    heat[mask, 1] = 0
    heat[mask, 2] = 0
    out = Image.fromarray(heat, mode="RGB")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_png)
