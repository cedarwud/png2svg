from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from png2svg.errors import Png2SvgError


def _local_mean(gray: np.ndarray, block_size: int) -> np.ndarray:
    if block_size <= 1:
        return gray.astype(np.float32)
    pad = block_size // 2
    padded = np.pad(gray.astype(np.float32), pad, mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = gray.shape
    y0 = np.arange(height)
    y1 = y0 + block_size
    x0 = np.arange(width)
    x1 = x0 + block_size
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return sums / float(block_size * block_size)


def _adaptive_binarize(gray: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    block_size = int(config.get("block_size", 31))
    c_value = float(config.get("c", 10))
    local = _local_mean(gray, block_size)
    threshold = local - c_value
    return (gray < threshold).astype(np.uint8)


def _pad_roi(roi: dict[str, Any], pad_px: int, width: int, height: int) -> dict[str, Any]:
    x = max(int(roi["x"]) - pad_px, 0)
    y = max(int(roi["y"]) - pad_px, 0)
    w = int(roi["width"]) + pad_px * 2
    h = int(roi["height"]) + pad_px * 2
    w = min(w, width - x)
    h = min(h, height - y)
    padded = dict(roi)
    padded.update({"x": x, "y": y, "width": w, "height": h})
    return padded


def _load_image(path: Any) -> tuple[np.ndarray, int, int]:
    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
    except Exception as exc:  # noqa: BLE001
        raise Png2SvgError(
            code="E4001_IMAGE_READ",
            message=f"Failed to read image: {exc}",
            hint="Ensure the input is a valid PNG file.",
        ) from exc
    arr = np.asarray(rgba, dtype=np.uint8)
    height, width = arr.shape[0], arr.shape[1]
    return arr, width, height


def _ink_mask(rgba: np.ndarray, adaptive: dict[str, Any] | None = None) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    if adaptive and adaptive.get("binarization", {}).get("enabled"):
        bin_cfg = adaptive["binarization"]
        alpha_threshold = int(bin_cfg.get("alpha_threshold", 10))
        lum_threshold = int(bin_cfg.get("luminance_threshold", 245))
        return ((alpha > alpha_threshold) & (luminance < lum_threshold)).astype(np.uint8)
    return ((alpha > 0) & (luminance < 245)).astype(np.uint8)


def _preprocess_image(rgba: np.ndarray, adaptive: dict[str, Any] | None = None) -> np.ndarray:
    rgb = rgba[:, :, :3]
    gray = np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])
    if adaptive and adaptive.get("binarization", {}).get("enabled"):
        return _adaptive_binarize(gray, adaptive["binarization"])
    return (gray < 230).astype(np.uint8)


def _prepare_ocr_image(rgba: np.ndarray, adaptive: dict[str, Any] | None = None) -> Image.Image:
    rgb = Image.fromarray(rgba, mode="RGBA")
    gray = ImageOps.grayscale(rgb)
    ocr_cfg = adaptive.get("ocr") if adaptive else None
    if isinstance(ocr_cfg, dict):
        blur_radius = float(ocr_cfg.get("blur_radius", 0.0))
        if blur_radius > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        threshold = int(ocr_cfg.get("threshold", 190))
        gray = gray.point(lambda val: 255 if val > threshold else 0)
    return gray


def _neighbors(y: int, x: int, height: int, width: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny = y + dy
        nx = x + dx
        if 0 <= ny < height and 0 <= nx < width:
            points.append((ny, nx))
    return points


def _connected_components(mask: np.ndarray, min_area: int) -> list[dict[str, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    components: list[dict[str, int]] = []
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny, nx in _neighbors(cy, cx, height, width):
                    if mask[ny, nx] == 0 or visited[ny, nx]:
                        continue
                    visited[ny, nx] = 1
                    stack.append((ny, nx))
            if area < min_area:
                continue
            components.append(
                {
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                    "area": area,
                }
            )
    return components
