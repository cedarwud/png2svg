from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import yaml

from png2svg.classifier import classify_png
from png2svg.errors import Png2SvgError
from png2svg.normalize import normalize_params
from png2svg.ocr import has_pytesseract, has_tesseract, ocr_image, write_ocr_json
from common.svg_builder import DEFAULT_FONT_FAMILY


@dataclass(frozen=True)
class ExtractIssue:
    code: str
    message: str
    hint: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message, "hint": self.hint}
        if self.context:
            payload["context"] = self.context
        return payload


TEMPLATE_ALIASES = {
    "3gpp_3panel": "t_3gpp_events_3panel",
    "t_3gpp_events_3panel": "t_3gpp_events_3panel",
    "procedure_flow": "t_procedure_flow",
    "t_procedure_flow": "t_procedure_flow",
    "performance_lineplot": "t_performance_lineplot",
    "t_performance_lineplot": "t_performance_lineplot",
}

DEFAULT_SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
DEFAULT_DASHARRAY = [6, 4]
DEFAULT_ADAPTIVE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "extract_adaptive.v1.yaml"


def _clamp_value(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _odd_int(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise Png2SvgError(
            code="E4012_ADAPTIVE_CONFIG_INVALID",
            message=f"Adaptive config must be a mapping: {path}",
            hint="Ensure the adaptive config YAML is a mapping at the top level.",
        )
    return data


def _load_adaptive_config(path: Path | None = None) -> dict[str, Any]:
    resolved = path or DEFAULT_ADAPTIVE_CONFIG
    if not resolved.exists():
        raise Png2SvgError(
            code="E4011_ADAPTIVE_CONFIG_MISSING",
            message=f"Adaptive config not found: {resolved}",
            hint="Ensure config/extract_adaptive.v1.yaml exists.",
        )
    return _load_yaml(resolved)


def _effective_config(config: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    base_long_edge = int(config.get("base_long_edge", 800))
    scale_cfg = config.get("scale", {}) if isinstance(config.get("scale"), dict) else {}
    scale = max(width, height) / float(base_long_edge or 1)
    scale = _clamp_value(scale, float(scale_cfg.get("min", 0.6)), float(scale_cfg.get("max", 1.8)))

    bin_cfg = config.get("binarization", {}) if isinstance(config.get("binarization"), dict) else {}
    block_base = int(bin_cfg.get("block_size", 31))
    block_size = _odd_int(max(int(round(block_base * scale)), 3))
    block_size = int(
        _clamp_value(
            block_size,
            float(bin_cfg.get("block_size_min", 15)),
            float(bin_cfg.get("block_size_max", 61)),
        )
    )
    if block_size % 2 == 0:
        block_size += 1
    binarization = {
        "enabled": bool(bin_cfg.get("enabled", True)),
        "block_size": block_size,
        "c": int(round(float(bin_cfg.get("c", 10)))),
        "alpha_threshold": int(bin_cfg.get("alpha_threshold", 10)),
        "luminance_threshold": int(bin_cfg.get("luminance_threshold", 245)),
    }

    lines_cfg = config.get("lines", {}) if isinstance(config.get("lines"), dict) else {}
    long_line_ratio = float(lines_cfg.get("long_line_ratio", 0.5))
    long_line_min_len = max(int(round(height * long_line_ratio * scale)), 1)

    dash_cfg = config.get("dashes", {}) if isinstance(config.get("dashes"), dict) else {}
    dash_min_len = max(int(round(height * float(dash_cfg.get("min_len_ratio", 0.03)) * scale)), 2)
    dash_max_len = max(
        int(round(height * float(dash_cfg.get("max_len_ratio", 0.12)) * scale)),
        dash_min_len + 1,
    )
    dash_min_count = int(dash_cfg.get("min_count", 6))

    text_cfg = config.get("text_boxes", {}) if isinstance(config.get("text_boxes"), dict) else {}
    min_area_ratio = float(text_cfg.get("min_area_ratio", 0.00005))
    max_area_ratio = float(text_cfg.get("max_area_ratio", 0.02))
    min_area = max(int(width * height * min_area_ratio), 8)
    max_area = max(int(width * height * max_area_ratio), min_area + 1)
    text_boxes = {
        "min_area": min_area,
        "max_area": max_area,
        "min_size": int(text_cfg.get("min_size", 3)),
        "max_width_ratio": float(text_cfg.get("max_width_ratio", 0.6)),
        "max_height_ratio": float(text_cfg.get("max_height_ratio", 0.4)),
    }

    ocr_cfg = config.get("ocr", {}) if isinstance(config.get("ocr"), dict) else {}
    pad_ratio = float(ocr_cfg.get("roi_pad_ratio", 0.02))
    pad_px = int(round(min(width, height) * pad_ratio * scale))
    pad_px = int(
        _clamp_value(
            pad_px,
            float(ocr_cfg.get("roi_pad_min", 2)),
            float(ocr_cfg.get("roi_pad_max", 24)),
        )
    )
    min_conf = float(ocr_cfg.get("min_conf", 0.5))
    min_chars = int(ocr_cfg.get("min_chars", 2))
    min_bbox_height = int(round(float(ocr_cfg.get("min_bbox_height", 6)) * scale))
    min_bbox_width = int(round(float(ocr_cfg.get("min_bbox_width", 6)) * scale))
    ocr = {
        "roi_pad_px": pad_px,
        "min_conf": min_conf,
        "min_chars": min_chars,
        "min_bbox_height": min_bbox_height,
        "min_bbox_width": min_bbox_width,
        "max_bbox_height_ratio": float(ocr_cfg.get("max_bbox_height_ratio", 2.2)),
        "min_alnum_ratio": float(ocr_cfg.get("min_alnum_ratio", 0.35)),
        "min_alpha_ratio": float(ocr_cfg.get("min_alpha_ratio", 0.2)),
        "min_digit_ratio": float(ocr_cfg.get("min_digit_ratio", 0.15)),
        "min_ascii_ratio": float(ocr_cfg.get("min_ascii_ratio", 0.7)),
    }

    layout_cfg = config.get("text_layout", {}) if isinstance(config.get("text_layout"), dict) else {}
    text_layout = {
        "grid_px": float(layout_cfg.get("grid_px", 5.0)),
        "baseline_tolerance_px": float(layout_cfg.get("baseline_tolerance_px", 2.0)),
        "font_scale_ref_height": float(layout_cfg.get("font_scale_ref_height", 360.0)),
        "font_scale_min": float(layout_cfg.get("font_scale_min", 0.8)),
        "font_scale_max": float(layout_cfg.get("font_scale_max", 3.0)),
        "ocr_font_size_ratio": float(layout_cfg.get("ocr_font_size_ratio", 0.6)),
    }

    curves_cfg = config.get("curves", {}) if isinstance(config.get("curves"), dict) else {}
    curve_spacing = int(round(float(curves_cfg.get("sample_spacing_px", 120)) * scale))
    curve_spacing = max(curve_spacing, 20)
    curves = {
        "sample_spacing_px": curve_spacing,
        "max_points": int(curves_cfg.get("max_points", 8)),
        "min_segments": int(curves_cfg.get("min_segments", 4)),
        "max_segments": int(curves_cfg.get("max_segments", 8)),
        "smooth_window": int(curves_cfg.get("smooth_window", 7)),
        "hue_tolerance_deg": float(curves_cfg.get("hue_tolerance_deg", 22)),
        "saturation_min": float(curves_cfg.get("saturation_min", 0.25)),
        "value_min": float(curves_cfg.get("value_min", 0.2)),
    }

    return {
        "scale": scale,
        "binarization": binarization,
        "lines": {"long_line_min_len_px": long_line_min_len, "long_line_ratio": long_line_ratio},
        "dashes": {
            "min_len_px": dash_min_len,
            "max_len_px": dash_max_len,
            "min_count": dash_min_count,
        },
        "text_boxes": text_boxes,
        "ocr": ocr,
        "text_layout": text_layout,
        "curves": curves,
    }


def _local_mean(gray: np.ndarray, block_size: int) -> np.ndarray:
    pad = block_size // 2
    padded = np.pad(gray, pad, mode="edge").astype(np.float32)
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    height, width = gray.shape
    y0 = np.arange(0, height)
    x0 = np.arange(0, width)
    y1 = y0 + block_size
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


def _pad_roi(roi: dict[str, int], pad_px: int, width: int, height: int) -> dict[str, int]:
    x = max(int(roi["x"]) - pad_px, 0)
    y = max(int(roi["y"]) - pad_px, 0)
    w = int(roi["width"]) + pad_px * 2
    h = int(roi["height"]) + pad_px * 2
    w = min(w, width - x)
    h = min(h, height - y)
    return {"x": x, "y": y, "width": w, "height": h}


def _load_image(path: Path) -> tuple[np.ndarray, int, int]:
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
        bin_mask = _adaptive_binarize(luminance, bin_cfg)
        return (alpha > alpha_threshold) & (bin_mask > 0)
    alpha_threshold = 10
    lum_threshold = 245
    if adaptive and adaptive.get("binarization"):
        alpha_threshold = int(adaptive["binarization"].get("alpha_threshold", alpha_threshold))
        lum_threshold = int(adaptive["binarization"].get("luminance_threshold", lum_threshold))
    return (alpha > alpha_threshold) & (luminance < lum_threshold)


def _preprocess_image(rgba: np.ndarray, adaptive: dict[str, Any] | None = None) -> np.ndarray:
    mask = _ink_mask(rgba, adaptive)
    out = np.where(mask, 0, 255).astype(np.uint8)
    return out


def _prepare_ocr_image(rgba: np.ndarray, adaptive: dict[str, Any] | None = None) -> Image.Image:
    rgb = rgba[:, :, :3].astype(np.float32)
    gray = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(
        np.uint8
    )
    image = Image.fromarray(gray, mode="L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    bin_cfg = adaptive.get("binarization") if adaptive else None
    if isinstance(bin_cfg, dict) and bin_cfg.get("enabled", True):
        bin_cfg = dict(bin_cfg)
        bin_cfg["block_size"] = _odd_int(int(bin_cfg.get("block_size", 31)))
        mask = _adaptive_binarize(gray.astype(np.float32), bin_cfg)
        binary = np.where(mask > 0, 0, 255).astype(np.uint8)
        image = Image.fromarray(binary, mode="L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
    return image


def _neighbors(y: int, x: int, height: int, width: int) -> list[tuple[int, int]]:
    coords = []
    if y > 0:
        coords.append((y - 1, x))
    if y + 1 < height:
        coords.append((y + 1, x))
    if x > 0:
        coords.append((y, x - 1))
    if x + 1 < width:
        coords.append((y, x + 1))
    return coords


def _connected_components(mask: np.ndarray, min_area: int) -> list[dict[str, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, int]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                for ny, nx in _neighbors(cy, cx, height, width):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                components.append(
                    {
                        "x": int(min_x),
                        "y": int(min_y),
                        "width": int(max_x - min_x + 1),
                        "height": int(max_y - min_y + 1),
                        "area": int(area),
                    }
                )
    return components


def _max_run_length(values: np.ndarray) -> int:
    max_run = 0
    run = 0
    for value in values:
        if value:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


def _max_runs(mask: np.ndarray, axis: int) -> list[int]:
    runs: list[int] = []
    if axis == 0:
        for col in range(mask.shape[1]):
            runs.append(_max_run_length(mask[:, col]))
    else:
        for row in range(mask.shape[0]):
            runs.append(_max_run_length(mask[row, :]))
    return runs


def _cluster_indices(indices: list[int]) -> list[int]:
    if not indices:
        return []
    clusters: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx <= clusters[-1][-1] + 1:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    centers: list[int] = []
    for cluster in clusters:
        centers.append(int(round(sum(cluster) / len(cluster))))
    return centers


def _long_line_positions(mask: np.ndarray, axis: int, min_len: int) -> list[int]:
    runs = _max_runs(mask, axis=axis)
    positions = [idx for idx, run in enumerate(runs) if run >= min_len]
    return _cluster_indices(positions)


def _detect_text_boxes(rgba: np.ndarray, adaptive: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    mask = _ink_mask(rgba, adaptive)
    height, width = mask.shape
    text_cfg = adaptive.get("text_boxes") if adaptive else None
    if text_cfg:
        min_area = int(text_cfg.get("min_area", 8))
        max_area = int(text_cfg.get("max_area", min_area + 1))
        min_size = int(text_cfg.get("min_size", 3))
        max_width_ratio = float(text_cfg.get("max_width_ratio", 0.6))
        max_height_ratio = float(text_cfg.get("max_height_ratio", 0.4))
    else:
        min_area = max(int(height * width * 0.00005), 8)
        max_area = max(int(height * width * 0.02), min_area + 1)
        min_size = 3
        max_width_ratio = 0.6
        max_height_ratio = 0.4
    components = _connected_components(mask, min_area=min_area)
    boxes: list[dict[str, Any]] = []
    for comp in components:
        area = comp["area"]
        if area > max_area:
            continue
        w = comp["width"]
        h = comp["height"]
        if w < min_size or h < min_size:
            continue
        if w > width * max_width_ratio or h > height * max_height_ratio:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.2 or aspect > 12:
            continue
        boxes.append(
            {
                "text": None,
                "bbox": {
                    "x": comp["x"],
                    "y": comp["y"],
                    "width": w,
                    "height": h,
                },
                "confidence": 0.0,
            }
        )
    boxes.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    return boxes


def _text_items_from_boxes(text_boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not text_boxes:
        return []
    baselines: list[float] = []
    heights: list[float] = []
    boxes_sorted: list[tuple[float, dict[str, Any]]] = []
    for entry in text_boxes:
        bbox = entry.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            x = float(bbox["x"])
            y = float(bbox["y"])
            width = float(bbox["width"])
            height = float(bbox["height"])
        except (KeyError, TypeError, ValueError):
            continue
        baseline = y + height
        baselines.append(baseline)
        heights.append(height)
        boxes_sorted.append((baseline, {"x": x, "y": y, "width": width, "height": height}))
    if not boxes_sorted:
        return []
    heights.sort()
    median_height = heights[len(heights) // 2]
    tolerance = max(4.0, median_height * 0.6)
    boxes_sorted.sort(key=lambda item: (item[0], item[1]["x"]))

    clusters: list[list[dict[str, Any]]] = []
    cluster_baselines: list[float] = []
    for baseline, bbox in boxes_sorted:
        if not clusters or abs(baseline - cluster_baselines[-1]) > tolerance:
            clusters.append([bbox])
            cluster_baselines.append(baseline)
        else:
            clusters[-1].append(bbox)
            cluster_baselines[-1] = (cluster_baselines[-1] + baseline) / 2.0

    line_boxes: list[dict[str, Any]] = []
    for cluster, baseline in zip(clusters, cluster_baselines):
        min_x = min(box["x"] for box in cluster)
        max_x = max(box["x"] + box["width"] for box in cluster)
        min_y = min(box["y"] for box in cluster)
        max_y = max(box["y"] + box["height"] for box in cluster)
        line_boxes.append(
            {
                "min_x": float(min_x),
                "max_x": float(max_x),
                "min_y": float(min_y),
                "max_y": float(max_y),
                "baseline": float(baseline),
            }
        )

    line_boxes.sort(key=lambda item: (item["min_y"], item["min_x"]))
    line_gap = max(4.0, median_height * 1.8)
    blocks: list[dict[str, Any]] = []

    for line in line_boxes:
        best_idx: int | None = None
        best_overlap = 0.0
        for idx, block in enumerate(blocks):
            overlap = min(line["max_x"], block["max_x"]) - max(line["min_x"], block["min_x"])
            min_width = min(line["max_x"] - line["min_x"], block["max_x"] - block["min_x"])
            if min_width <= 0:
                continue
            overlap_ratio = overlap / min_width
            vertical_gap = line["min_y"] - block["max_y"]
            if 0 <= vertical_gap <= line_gap and overlap_ratio >= 0.4:
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_idx = idx
        if best_idx is None:
            blocks.append(
                {
                    "min_x": line["min_x"],
                    "max_x": line["max_x"],
                    "min_y": line["min_y"],
                    "max_y": line["max_y"],
                    "baseline": line["baseline"],
                }
            )
        else:
            block = blocks[best_idx]
            block["min_x"] = min(block["min_x"], line["min_x"])
            block["max_x"] = max(block["max_x"], line["max_x"])
            block["min_y"] = min(block["min_y"], line["min_y"])
            block["max_y"] = max(block["max_y"], line["max_y"])

    items: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        items.append(
            {
                "content": "Unknown",
                "text": "Unknown",
                "x": float(block["min_x"]),
                "y": float(block["baseline"]),
                "role": "annotation",
                "anchor": "start",
                "baseline_group": f"block_{idx}",
                "conf": 0.0,
                "bbox": {
                    "x": float(block["min_x"]),
                    "y": float(block["min_y"]),
                    "width": float(block["max_x"] - block["min_x"]),
                    "height": float(block["max_y"] - block["min_y"]),
                },
            }
        )
    return items


def _group_ocr_lines(
    ocr_results: list[dict[str, Any]], tolerance: float
) -> list[list[dict[str, Any]]]:
    if not ocr_results:
        return []
    sorted_results = sorted(
        ocr_results,
        key=lambda item: (
            float(item["bbox"]["y"]) + float(item["bbox"]["height"]),
            float(item["bbox"]["x"]),
        ),
    )
    clusters: list[list[dict[str, Any]]] = []
    cluster_baselines: list[float] = []
    for item in sorted_results:
        bbox = item["bbox"]
        baseline = float(bbox["y"]) + float(bbox["height"])
        if not clusters or abs(baseline - cluster_baselines[-1]) > tolerance:
            clusters.append([item])
            cluster_baselines.append(baseline)
        else:
            clusters[-1].append(item)
            cluster_baselines[-1] = (cluster_baselines[-1] + baseline) / 2.0
    split_clusters: list[list[dict[str, Any]]] = []
    for cluster in clusters:
        cluster.sort(key=lambda item: float(item["bbox"]["x"]))
        if len(cluster) <= 1:
            split_clusters.append(cluster)
            continue
        heights = sorted(float(item["bbox"]["height"]) for item in cluster)
        median_height = heights[len(heights) // 2] if heights else 10.0
        gap_threshold = max(median_height * 2.5, 24.0)
        current: list[dict[str, Any]] = [cluster[0]]
        for item in cluster[1:]:
            prev = current[-1]
            gap = float(item["bbox"]["x"]) - (
                float(prev["bbox"]["x"]) + float(prev["bbox"]["width"])
            )
            if gap > gap_threshold:
                split_clusters.append(current)
                current = [item]
            else:
                current.append(item)
        if current:
            split_clusters.append(current)
    return split_clusters


def _text_items_from_ocr(
    ocr_results: list[dict[str, Any]],
    width: int,
    height: int,
    adaptive: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not ocr_results:
        return []
    heights = sorted(float(item["bbox"]["height"]) for item in ocr_results)
    median_height = heights[len(heights) // 2]
    ocr_cfg = adaptive.get("ocr") if adaptive else None
    layout_cfg = adaptive.get("text_layout") if adaptive else None
    max_height_ratio = float(ocr_cfg.get("max_bbox_height_ratio", 2.2)) if isinstance(ocr_cfg, dict) else 2.2
    font_ratio = float(layout_cfg.get("ocr_font_size_ratio", 0.6)) if isinstance(layout_cfg, dict) else 0.6
    max_height = max(median_height * max_height_ratio, median_height + 1.0)
    filtered_results = [
        item
        for item in ocr_results
        if float(item.get("bbox", {}).get("height", 0.0)) <= max_height
    ]
    tolerance = max(4.0, median_height * 0.6)
    lines = _group_ocr_lines(filtered_results, tolerance)
    items: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        texts: list[str] = []
        word_heights: list[float] = []
        for item in line:
            raw = item.get("text")
            if not raw:
                continue
            text = _clean_text(str(raw))
            if not text:
                continue
            if len(text) == 1 and not text.isalnum():
                continue
            if texts and text == texts[-1]:
                continue
            texts.append(text)
            try:
                word_heights.append(float(item["bbox"]["height"]))
            except (TypeError, ValueError):
                continue
        if not texts:
            continue
        content = " ".join(texts)
        min_x = min(float(item["bbox"]["x"]) for item in line)
        min_y = min(float(item["bbox"]["y"]) for item in line)
        max_x = max(float(item["bbox"]["x"]) + float(item["bbox"]["width"]) for item in line)
        max_y = max(float(item["bbox"]["y"]) + float(item["bbox"]["height"]) for item in line)
        baseline = max_y
        conf_values = [float(item.get("conf", 0.0)) for item in line]
        conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        if word_heights:
            word_heights.sort()
            median_word_height = word_heights[len(word_heights) // 2]
        else:
            median_word_height = median_height
        font_size = max(6.0, median_word_height * font_ratio)
        bbox = {
            "x": float(min_x),
            "y": float(min_y),
            "width": float(max_x - min_x),
            "height": float(max_y - min_y),
        }
        bbox["x0"] = bbox["x"]
        bbox["y0"] = bbox["y"]
        bbox["x1"] = bbox["x"] + bbox["width"]
        bbox["y1"] = bbox["y"] + bbox["height"]
        items.append(
            {
                "content": content,
                "text": content,
                "x": float(min_x),
                "y": float(baseline),
                "role": "annotation",
                "anchor": "start",
                "baseline_group": f"line_{idx}",
                "conf": conf,
                "confidence": conf,
                "font_size": font_size,
                "bbox": bbox,
            }
        )
    return items


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _text_char_stats(value: str) -> dict[str, float]:
    total = len(value)
    if total == 0:
        return {"total": 0, "alnum": 0, "alpha": 0, "digit": 0, "ascii": 0}
    alnum = sum(1 for ch in value if ch.isalnum())
    alpha = sum(1 for ch in value if ch.isalpha())
    digit = sum(1 for ch in value if ch.isdigit())
    ascii_count = sum(1 for ch in value if ord(ch) < 128)
    return {
        "total": total,
        "alnum": alnum,
        "alpha": alpha,
        "digit": digit,
        "ascii": ascii_count,
    }


def _keep_text_item(item: dict[str, Any], cfg: dict[str, Any] | None) -> bool:
    text = _clean_text(str(item.get("content") or item.get("text") or ""))
    if not text:
        return False
    try:
        conf = float(item.get("conf", 1.0))
    except (TypeError, ValueError):
        conf = 1.0
    min_conf = float(cfg.get("min_conf", 0.5)) if isinstance(cfg, dict) else 0.5
    if conf < min_conf:
        return False
    min_chars = int(cfg.get("min_chars", 2)) if isinstance(cfg, dict) else 2
    if len(text) < min_chars:
        return False
    bbox = item.get("bbox")
    if isinstance(bbox, dict):
        min_h = int(cfg.get("min_bbox_height", 6)) if isinstance(cfg, dict) else 6
        min_w = int(cfg.get("min_bbox_width", 6)) if isinstance(cfg, dict) else 6
        try:
            if float(bbox.get("height", 0.0)) < min_h or float(bbox.get("width", 0.0)) < min_w:
                return False
        except (TypeError, ValueError):
            pass
    stats = _text_char_stats(text)
    total = stats["total"]
    if total <= 0:
        return False
    alnum_ratio = stats["alnum"] / total
    alpha_ratio = stats["alpha"] / total
    digit_ratio = stats["digit"] / total
    ascii_ratio = stats["ascii"] / total
    min_alnum = float(cfg.get("min_alnum_ratio", 0.35)) if isinstance(cfg, dict) else 0.35
    min_alpha = float(cfg.get("min_alpha_ratio", 0.2)) if isinstance(cfg, dict) else 0.2
    min_digit = float(cfg.get("min_digit_ratio", 0.15)) if isinstance(cfg, dict) else 0.15
    min_ascii = float(cfg.get("min_ascii_ratio", 0.7)) if isinstance(cfg, dict) else 0.7
    if ascii_ratio < min_ascii:
        return False
    if alnum_ratio < min_alnum:
        return False
    if alpha_ratio < min_alpha and digit_ratio < min_digit:
        return False
    return True


def _filter_text_items(text_items: list[dict[str, Any]], cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not text_items:
        return []
    filtered: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in text_items:
        if not _keep_text_item(item, cfg):
            continue
        content = _clean_text(str(item.get("content") or item.get("text") or ""))
        if not content:
            continue
        item["content"] = content
        item["text"] = content
        bbox = item.get("bbox")
        if isinstance(bbox, dict):
            try:
                key = (content, int(round(float(bbox.get("x", 0.0)))), int(round(float(bbox.get("y", 0.0)))))
            except (TypeError, ValueError):
                key = (content, 0, 0)
        else:
            key = (content, 0, 0)
        existing = seen.get(key)
        if existing is None:
            seen[key] = item
            filtered.append(item)
        else:
            try:
                if float(item.get("conf", 0.0)) > float(existing.get("conf", 0.0)):
                    seen[key] = item
                    idx = filtered.index(existing)
                    filtered[idx] = item
            except (TypeError, ValueError):
                continue
    filtered.sort(
        key=lambda item: (
            float(item.get("bbox", {}).get("y", 0.0)),
            float(item.get("bbox", {}).get("x", 0.0)),
            str(item.get("content") or item.get("text") or ""),
        )
    )
    return filtered


def _count_renderable_texts(text_items: list[dict[str, Any]]) -> int:
    count = 0
    for item in text_items:
        if item.get("render") is False:
            continue
        text_value = _clean_text(str(item.get("content") or item.get("text") or ""))
        if text_value:
            count += 1
    return count


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window, window), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window:-window]


def _find_separator(values: np.ndarray, target: int, span: int) -> int | None:
    start = max(int(target - span), 0)
    end = min(int(target + span), len(values))
    if end <= start:
        return None
    segment = values[start:end]
    if segment.size == 0:
        return None
    offset = int(np.argmin(segment))
    return start + offset


def _detect_panel_columns(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    col_ink = mask.sum(axis=0)
    window = max(int(width * 0.05), 5)
    smoothed = _smooth_series(col_ink, window)
    span = int(width * 0.12)
    sep1 = _find_separator(smoothed, int(width / 3), span)
    sep2 = _find_separator(smoothed, int(width * 2 / 3), span)
    if sep1 is None or sep2 is None:
        return []
    if sep2 - sep1 < int(width * 0.18):
        return []
    return sorted([sep1, sep2])


def _detect_panels(mask: np.ndarray, width: int, height: int) -> list[dict[str, Any]]:
    separators = _detect_panel_columns(mask)
    if not separators:
        return _default_panels(width, height)
    edges = [0, separators[0], separators[1], width]
    panels: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right - left < max(int(width * 0.15), 40):
            return _default_panels(width, height)
        submask = mask[:, left:right]
        row_ink = submask.sum(axis=1)
        row_threshold = max(int((right - left) * 0.02), 4)
        search_top = int(height * 0.15)
        search_bottom = int(height * 0.9)
        top_candidates = np.where(row_ink[search_top:search_bottom] > row_threshold)[0]
        if top_candidates.size > 0:
            top = int(top_candidates[0]) + search_top
        else:
            top = int(height * 0.2)
        bottom_candidates = np.where(row_ink[search_top:search_bottom] > row_threshold)[0]
        if bottom_candidates.size > 0:
            bottom = int(bottom_candidates[-1]) + search_top
        else:
            bottom = int(height * 0.8)
        if bottom - top < int(height * 0.2):
            top = int(height * 0.2)
            bottom = int(height * 0.8)
        margin = max(int((right - left) * 0.04), 10)
        panel = {
            "id": ["A3", "A4", "A5"][idx],
            "label": ["A3", "A4", "A5"][idx],
            "x": float(left + margin),
            "y": float(top),
            "width": float(max(right - left - margin * 2, 20)),
            "height": float(max(bottom - top, 20)),
        }
        panels.append(panel)
    return panels


def _line_extent(mask: np.ndarray, axis: int, idx: int) -> tuple[int, int] | None:
    if axis == 0:
        column = mask[:, idx]
        points = np.where(column)[0]
    else:
        row = mask[idx, :]
        points = np.where(row)[0]
    if points.size == 0:
        return None
    return int(points[0]), int(points[-1])


def _detect_axes_lines(
    mask: np.ndarray,
    panel: dict[str, Any],
    adaptive: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    x0 = int(panel["x"])
    x1 = int(panel["x"] + panel["width"])
    y0 = int(panel["y"])
    y1 = int(panel["y"] + panel["height"])
    submask = mask[y0:y1, x0:x1]
    if submask.size == 0:
        return []
    height, width = submask.shape
    ratio = 0.6
    if adaptive and adaptive.get("lines"):
        ratio = float(adaptive["lines"].get("long_line_ratio", ratio))
        scale = float(adaptive.get("scale", 1.0))
        ratio = max(ratio * scale, 0.1)
    long_v = _long_line_positions(submask, axis=0, min_len=max(int(height * ratio), 1))
    long_h = _long_line_positions(submask, axis=1, min_len=max(int(width * ratio), 1))
    lines: list[dict[str, Any]] = []
    if long_v:
        vx = min(long_v)
        extent = _line_extent(submask, axis=0, idx=vx)
        if extent:
            y_start, y_end = extent
            lines.append(
                {
                    "x1": x0 + vx,
                    "y1": y0 + y_start,
                    "x2": x0 + vx,
                    "y2": y0 + y_end,
                    "role": "axis",
                    "stroke": "#000000",
                    "stroke_width": 2,
                }
            )
    if long_h:
        hy = max(long_h)
        extent = _line_extent(submask, axis=1, idx=hy)
        if extent:
            x_start, x_end = extent
            lines.append(
                {
                    "x1": x0 + x_start,
                    "y1": y0 + hy,
                    "x2": x0 + x_end,
                    "y2": y0 + hy,
                    "role": "axis",
                    "stroke": "#000000",
                    "stroke_width": 2,
                }
            )
    return lines


def _run_lengths(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _dashed_line_candidates(
    mask: np.ndarray,
    axis: int,
    min_len: int,
    max_len: int,
    min_count: int,
    min_span_ratio: float,
    min_coverage: float,
    max_coverage: float,
    min_gap_ratio: float,
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    if axis == 0:
        length = mask.shape[1]
        for idx in range(length):
            runs = _run_lengths(mask[:, idx])
            segments = [(start, end) for start, end in runs if min_len <= (end - start + 1) <= max_len]
            if len(segments) < min_count:
                continue
            lengths = [end - start + 1 for start, end in segments]
            total_len = sum(lengths)
            span_start = min(start for start, _ in segments)
            span_end = max(end for _, end in segments)
            span_ratio = (span_end - span_start + 1) / float(mask.shape[0])
            coverage = total_len / float(mask.shape[0])
            if coverage < min_coverage or coverage > max_coverage or span_ratio < min_span_ratio:
                continue
            gaps = []
            segments_sorted = sorted(segments, key=lambda seg: seg[0])
            for (s0, e0), (s1, _) in zip(segments_sorted, segments_sorted[1:]):
                gaps.append(max(s1 - e0 - 1, 0))
            median_gap = float(np.median(gaps)) if gaps else 0.0
            median_len = float(np.median(lengths)) if lengths else 0.0
            if median_gap < median_len * min_gap_ratio:
                continue
            candidates.append(
                {
                    "idx": float(idx),
                    "span_ratio": span_ratio,
                    "coverage": coverage,
                    "count": float(len(segments)),
                }
            )
    else:
        length = mask.shape[0]
        for idx in range(length):
            runs = _run_lengths(mask[idx, :])
            segments = [(start, end) for start, end in runs if min_len <= (end - start + 1) <= max_len]
            if len(segments) < min_count:
                continue
            lengths = [end - start + 1 for start, end in segments]
            total_len = sum(lengths)
            span_start = min(start for start, _ in segments)
            span_end = max(end for _, end in segments)
            span_ratio = (span_end - span_start + 1) / float(mask.shape[1])
            coverage = total_len / float(mask.shape[1])
            if coverage < min_coverage or coverage > max_coverage or span_ratio < min_span_ratio:
                continue
            gaps = []
            segments_sorted = sorted(segments, key=lambda seg: seg[0])
            for (s0, e0), (s1, _) in zip(segments_sorted, segments_sorted[1:]):
                gaps.append(max(s1 - e0 - 1, 0))
            median_gap = float(np.median(gaps)) if gaps else 0.0
            median_len = float(np.median(lengths)) if lengths else 0.0
            if median_gap < median_len * min_gap_ratio:
                continue
            candidates.append(
                {
                    "idx": float(idx),
                    "span_ratio": span_ratio,
                    "coverage": coverage,
                    "count": float(len(segments)),
                }
            )
    return candidates


def _detect_dashed_lines(
    mask: np.ndarray,
    adaptive: dict[str, Any] | None = None,
    panels: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    height, width = mask.shape
    dash_cfg = adaptive.get("dashes") if adaptive else None
    if dash_cfg:
        min_len_v = max(int(dash_cfg.get("min_len_px", 2)), 2)
        max_len_v = max(int(dash_cfg.get("max_len_px", min_len_v + 1)), min_len_v + 1)
        min_len_h = min_len_v
        max_len_h = max_len_v
        min_count = int(dash_cfg.get("min_count", 6))
        min_span_ratio = float(dash_cfg.get("min_span_ratio", 0.6))
        min_coverage = float(dash_cfg.get("min_coverage", 0.02))
        max_coverage = float(dash_cfg.get("max_coverage", 0.35))
        min_gap_ratio = float(dash_cfg.get("min_gap_ratio", 0.5))
        max_lines_per_panel = int(dash_cfg.get("max_lines_per_panel", 2))
    else:
        min_len_v = max(int(height * 0.03), 2)
        max_len_v = max(int(height * 0.12), min_len_v + 1)
        min_len_h = max(int(width * 0.03), 2)
        max_len_h = max(int(width * 0.12), min_len_h + 1)
        min_count = 6
        min_span_ratio = 0.6
        min_coverage = 0.02
        max_coverage = 0.35
        min_gap_ratio = 0.5
        max_lines_per_panel = 2
    lines: list[dict[str, Any]] = []
    panel_list = panels or []
    panel_bounds = [
        (
            float(panel.get("x", 0.0)),
            float(panel.get("x", 0.0)) + float(panel.get("width", 0.0)),
            float(panel.get("y", 0.0)),
            float(panel.get("y", 0.0)) + float(panel.get("height", 0.0)),
        )
        for panel in panel_list
    ]
    regions = panel_bounds if panel_bounds else [(0.0, float(width), 0.0, float(height))]
    for px0, px1, py0, py1 in regions:
        x0 = int(max(px0, 0.0))
        x1 = int(min(px1, width))
        y0 = int(max(py0, 0.0))
        y1 = int(min(py1, height))
        if x1 <= x0 or y1 <= y0:
            continue
        sub = mask[y0:y1, x0:x1]
        dashed_v = _dashed_line_candidates(
            sub,
            axis=0,
            min_len=min_len_v,
            max_len=max_len_v,
            min_count=min_count,
            min_span_ratio=min_span_ratio,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            min_gap_ratio=min_gap_ratio,
        )
        dashed_h: list[dict[str, float]] = []
        dashed_v.sort(key=lambda item: (-item["count"], -item["span_ratio"], item["idx"]))
        selected: list[dict[str, float]] = []
        min_sep = max(int((x1 - x0) * 0.1), 12)
        for candidate in dashed_v:
            if any(abs(candidate["idx"] - kept["idx"]) < min_sep for kept in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_lines_per_panel:
                break
        for candidate in selected:
            x = candidate["idx"] + x0
            extent = _line_extent(sub, axis=0, idx=int(candidate["idx"]))
            if not extent:
                continue
            ey0, ey1 = extent
            if panel_bounds:
                height_span = ey1 - ey0
                if height_span < (y1 - y0) * 0.5:
                    continue
            lines.append(
                {
                    "x1": float(x),
                    "y1": float(ey0 + y0),
                    "x2": float(x),
                    "y2": float(ey1 + y0),
                    "role": "threshold",
                    "stroke": "#555555",
                    "stroke_width": 1,
                    "dashed": True,
                    "dasharray": DEFAULT_DASHARRAY,
                }
            )
        for candidate in dashed_h[:max_lines_per_panel]:
            y = candidate["idx"] + y0
            extent = _line_extent(sub, axis=1, idx=int(candidate["idx"]))
            if not extent:
                continue
            ex0, ex1 = extent
            if panel_bounds:
                width_span = ex1 - ex0
                if width_span < (x1 - x0) * 0.5:
                    continue
            lines.append(
                {
                    "x1": float(ex0 + x0),
                    "y1": float(y),
                    "x2": float(ex1 + x0),
                    "y2": float(y),
                    "role": "threshold",
                    "stroke": "#555555",
                    "stroke_width": 1,
                    "dashed": True,
                    "dasharray": DEFAULT_DASHARRAY,
                }
            )
    return lines


def _detect_markers(rgba: np.ndarray) -> list[dict[str, Any]]:
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3]
    red = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100) & (alpha > 10)
    components = _connected_components(red, min_area=6)
    markers: list[dict[str, Any]] = []
    for comp in components:
        width = comp["width"]
        height = comp["height"]
        if width > 30 or height > 30:
            continue
        if width < 2 or height < 2:
            continue
        cx = comp["x"] + width / 2
        cy = comp["y"] + height / 2
        radius = max(width, height) / 2
        markers.append(
            {
                "x": float(cx),
                "y": float(cy),
                "radius": float(radius),
                "fill": "#dd6b20",
                "role": "event_marker",
            }
        )
    return markers


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    hue = np.zeros_like(cmax)
    mask = delta > 1e-6
    mask_r = (cmax == r) & mask
    mask_g = (cmax == g) & mask
    mask_b = (cmax == b) & mask
    hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    hue[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0
    hue[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0
    sat = np.zeros_like(cmax)
    nonzero = cmax > 1e-6
    sat[nonzero] = delta[nonzero] / cmax[nonzero]
    val = cmax
    return hue, sat, val


def _hue_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _hue_in_range(hue: np.ndarray, low: float, high: float) -> np.ndarray:
    low = low % 360.0
    high = high % 360.0
    if low <= high:
        return (hue >= low) & (hue <= high)
    return (hue >= low) | (hue <= high)


def _pick_hue_center(hue: np.ndarray, mask: np.ndarray, target: float, max_distance: float) -> float | None:
    if mask.sum() == 0:
        return None
    hue_vals = hue[mask]
    bins = ((hue_vals // 10) % 36).astype(np.int32)
    counts = np.bincount(bins, minlength=36)
    best_bin = None
    best_count = 0
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        center = idx * 10.0 + 5.0
        if _hue_distance(center, target) <= max_distance and count > best_count:
            best_bin = idx
            best_count = int(count)
    if best_bin is None:
        return None
    return best_bin * 10.0 + 5.0


def _curve_color_mask(
    rgba: np.ndarray,
    target_hue: float,
    adaptive: dict[str, Any],
) -> np.ndarray:
    hue, sat, val = _rgb_to_hsv(rgba[:, :, :3])
    alpha = rgba[:, :, 3]
    sat_min = float(adaptive.get("saturation_min", 0.25))
    val_min = float(adaptive.get("value_min", 0.2))
    hue_tol = float(adaptive.get("hue_tolerance_deg", 22.0))

    def _build_mask(sat_threshold: float, val_threshold: float, tol: float) -> np.ndarray:
        base = (alpha > 10) & (sat >= sat_threshold) & (val >= val_threshold)
        center = _pick_hue_center(hue, base, target_hue, max_distance=tol * 2.0)
        if center is None:
            center = target_hue
        return base & _hue_in_range(hue, center - tol, center + tol)

    mask = _build_mask(sat_min, val_min, hue_tol)
    min_pixels = max(int(rgba.shape[0] * rgba.shape[1] * 0.0005), 8)
    if mask.sum() < min_pixels:
        relaxed_sat = max(sat_min * 0.6, 0.05)
        relaxed_val = max(val_min * 0.6, 0.05)
        relaxed_tol = min(hue_tol * 1.4, 60.0)
        mask = _build_mask(relaxed_sat, relaxed_val, relaxed_tol)
    return mask


def _point_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        dx = point[0] - start[0]
        dy = point[1] - start[1]
        return (dx * dx + dy * dy) ** 0.5
    x0, y0 = point
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_dist = 0.0
    index = 0
    for idx, point in enumerate(points[1:-1], start=1):
        dist = _point_distance(point, start, end)
        if dist > max_dist:
            max_dist = dist
            index = idx
    if max_dist > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def _simplify_curve_points(
    points: list[tuple[float, float]],
    max_points: int,
    min_points: int,
    epsilon: float,
) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    simplified = _rdp(points, epsilon)
    step = 0
    while len(simplified) > max_points and step < 6:
        epsilon *= 1.6
        simplified = _rdp(points, epsilon)
        step += 1
    if len(simplified) > max_points:
        keep = [points[0]]
        if max_points > 2:
            stride = max(1, int((len(points) - 2) / (max_points - 2)))
            keep.extend(points[1:-1:stride])
        keep.append(points[-1])
        simplified = keep[:max_points]
    if len(simplified) < min_points:
        if len(points) >= min_points:
            simplified = points[:min_points]
        else:
            simplified = points
    return simplified


def _snap_flat_series(values: np.ndarray, tolerance: float) -> np.ndarray:
    if values.size == 0:
        return values
    snapped = values.copy()
    last = snapped[0]
    for idx in range(1, snapped.size):
        if abs(snapped[idx] - last) <= tolerance:
            snapped[idx] = last
        else:
            last = snapped[idx]
    return snapped


def _curve_centerline_points(
    mask: np.ndarray,
    adaptive: dict[str, Any],
) -> list[tuple[float, float]]:
    height, width = mask.shape
    xs: list[int] = []
    ys: list[float] = []
    min_samples = max(int(height * 0.01), 1)
    for x in range(width):
        ys_col = np.where(mask[:, x])[0]
        if ys_col.size >= min_samples:
            xs.append(x)
            ys.append(float(np.median(ys_col)))
    if len(xs) < 4:
        return []
    xs_arr = np.array(xs, dtype=np.int32)
    ys_arr = np.array(ys, dtype=np.float32)
    xs_full = np.arange(xs_arr[0], xs_arr[-1] + 1)
    ys_full = np.interp(xs_full, xs_arr.astype(np.float32), ys_arr)
    smooth_window = int(adaptive.get("smooth_window", 7))
    smooth_window = max(smooth_window, 3)
    if smooth_window % 2 == 0:
        smooth_window += 1
    ys_smooth = _smooth_series(ys_full, smooth_window)
    ys_smooth = _snap_flat_series(ys_smooth, tolerance=1.5)
    points = list(zip(xs_full.astype(np.float32), ys_smooth.astype(np.float32)))

    sample_spacing = max(int(adaptive.get("sample_spacing_px", 120)), 10)
    min_segments = int(adaptive.get("min_segments", 4))
    max_segments = int(adaptive.get("max_segments", 8))
    max_points_cfg = int(adaptive.get("max_points", max_segments + 1))
    min_points = max(min_segments + 1, 4)
    max_points = max(min_points, max_points_cfg)
    max_segments = max(min_segments, min(max_segments, max_points - 1))
    target_points = int(width / float(sample_spacing)) + 2
    target_points = max(min_points, min(max_points, target_points))
    epsilon = max(height * 0.01, 1.0)
    simplified = _simplify_curve_points(points, target_points, min_points, epsilon)
    return simplified


def _text_bbox_center(item: dict[str, Any]) -> tuple[float, float]:
    bbox = item.get("bbox", {})
    return (
        float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0)) / 2.0,
        float(bbox.get("y", 0.0)) + float(bbox.get("height", 0.0)) / 2.0,
    )


def _text_bbox_width(item: dict[str, Any]) -> float:
    bbox = item.get("bbox", {})
    return float(bbox.get("width", 0.0))


def _text_bbox_height(item: dict[str, Any]) -> float:
    bbox = item.get("bbox", {})
    return float(bbox.get("height", 0.0))


def _panel_label_from_text(text: str) -> str | None:
    cleaned = _clean_text(text).upper()
    for label in ("A3", "A4", "A5"):
        if label in cleaned:
            return label
    return None


def _assign_roles_3gpp(
    text_items: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    title: str | None = None
    title_style: dict[str, Any] | None = None
    top_candidates = [
        item
        for item in text_items
        if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.1
        and _text_bbox_width(item) > width * 0.5
    ]
    if not top_candidates:
        top_candidates = [
            item
            for item in text_items
            if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.14
            and _text_bbox_width(item) > width * 0.45
        ]
    if top_candidates:
        top_sorted = sorted(
            top_candidates,
            key=lambda item: (
                float(item.get("bbox", {}).get("y", 0.0)),
                float(item.get("bbox", {}).get("x", 0.0)),
            ),
        )
        lines: list[str] = []
        font_sizes: list[float] = []
        min_x = None
        max_x = None
        min_y = None
        y_positions: list[float] = []
        for item in top_sorted:
            text_value = _clean_text(str(item.get("content") or item.get("text") or ""))
            if not text_value:
                continue
            if not lines or text_value != lines[-1]:
                lines.append(text_value)
            try:
                font_sizes.append(float(item.get("font_size", 0.0)))
            except (TypeError, ValueError):
                pass
            bbox = item.get("bbox", {})
            try:
                x0 = float(bbox.get("x", 0.0))
                x1 = float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0))
                y0 = float(bbox.get("y", 0.0))
            except (TypeError, ValueError):
                x0 = x1 = y0 = 0.0
            min_x = x0 if min_x is None else min(min_x, x0)
            max_x = x1 if max_x is None else max(max_x, x1)
            min_y = y0 if min_y is None else min(min_y, y0)
            try:
                y_positions.append(float(item.get("y", 0.0)))
            except (TypeError, ValueError):
                pass
        if lines:
            title = "\n".join(lines)
            if font_sizes:
                font_sizes.sort()
                font_size = font_sizes[len(font_sizes) // 2]
            else:
                font_size = max(12.0, height * 0.02)
            anchor = "start"
            title_x = min_x if min_x is not None else 10.0
            if min_x is not None and max_x is not None:
                center_x = (min_x + max_x) / 2.0
                if abs(center_x - width / 2.0) <= width * 0.1:
                    anchor = "middle"
                    title_x = width / 2.0
            title_y = min(y_positions) if y_positions else (min_y + font_size if min_y is not None else 20.0)
            title_style = {
                "x": float(title_x),
                "y": float(title_y),
                "font_size": float(font_size),
                "font_weight": "bold",
                "anchor": anchor,
            }
        for item in top_candidates:
            item["role"] = "title"
            item["anchor"] = "middle"
            item["x"] = float(width) / 2.0
            item["render"] = False
    for panel in panels:
        px = panel["x"]
        py = panel["y"]
        pw = panel["width"]
        ph = panel["height"]
        for item in text_items:
            cx, cy = _text_bbox_center(item)
            if (
                px <= cx <= px + pw
                and py <= cy <= py + ph * 0.3
                and _text_bbox_width(item) <= pw * 0.4
            ):
                item["role"] = "panel_label"
                item["anchor"] = "start"
                item["render"] = False
    for item in text_items:
        if item.get("role") == "panel_label" or item.get("role") == "title":
            continue
        cx, cy = _text_bbox_center(item)
        matched = False
        for panel in panels:
            if (
                panel["x"] <= cx <= panel["x"] + panel["width"]
                and panel["y"] + panel["height"] * 0.8 <= cy <= panel["y"] + panel["height"]
            ):
                item["role"] = "axis_label"
                item["anchor"] = "middle"
                matched = True
                break
        if not matched:
            item.setdefault("role", "annotation")
    return title, title_style, text_items


def _snap_grid(value: float, grid: float) -> float:
    if grid <= 0:
        return value
    return round(value / grid) * grid


def _text_layout_config(adaptive: dict[str, Any] | None, height: int) -> dict[str, float]:
    cfg = adaptive.get("text_layout") if adaptive else None
    if not isinstance(cfg, dict):
        cfg = {}
    grid = float(cfg.get("grid_px", 5.0))
    baseline_tol = float(cfg.get("baseline_tolerance_px", 2.0))
    ref_height = float(cfg.get("font_scale_ref_height", 360.0))
    scale_min = float(cfg.get("font_scale_min", 0.8))
    scale_max = float(cfg.get("font_scale_max", 3.0))
    scale = height / ref_height if ref_height > 0 else 1.0
    scale = _clamp_value(scale, scale_min, scale_max)
    return {
        "grid_px": grid,
        "baseline_tolerance_px": baseline_tol,
        "font_scale": scale,
    }


def _baseline_groups(text_items: list[dict[str, Any]], tolerance: float) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in text_items:
        group = item.get("baseline_group")
        if isinstance(group, str) and group:
            groups.setdefault(group, []).append(item)
    if groups:
        return groups
    candidates = []
    for idx, item in enumerate(text_items):
        try:
            y = float(item.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        candidates.append((idx, y))
    candidates.sort(key=lambda pair: pair[1])
    clusters: list[list[int]] = []
    current: list[int] = []
    last_y = None
    for idx, y in candidates:
        if last_y is None or abs(y - last_y) <= tolerance:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
        last_y = y
    if current:
        clusters.append(current)
    for group_idx, indices in enumerate(clusters):
        group_key = f"baseline_{group_idx}"
        groups[group_key] = [text_items[idx] for idx in indices]
        for item in groups[group_key]:
            item["baseline_group"] = group_key
    return groups


def _apply_text_layout(
    text_items: list[dict[str, Any]],
    template_id: str,
    width: int,
    height: int,
    adaptive: dict[str, Any] | None,
) -> None:
    if not text_items:
        return
    layout_cfg = _text_layout_config(adaptive, height)
    grid = layout_cfg["grid_px"]
    baseline_tol = layout_cfg["baseline_tolerance_px"]
    font_scale = layout_cfg["font_scale"]
    role_styles = {
        "title": {"font_size": 16, "font_weight": "bold", "anchor": "middle"},
        "panel_label": {"font_size": 12, "font_weight": "bold", "anchor": "start"},
        "axis_label": {"font_size": 10, "font_weight": "normal", "anchor": "middle"},
        "axis_label_x": {"font_size": 10, "font_weight": "normal", "anchor": "middle"},
        "axis_label_y": {"font_size": 10, "font_weight": "normal", "anchor": "middle"},
        "legend": {"font_size": 9, "font_weight": "normal", "anchor": "start"},
        "node_label": {"font_size": 11, "font_weight": "normal", "anchor": "start"},
        "curve_label_serving": {"font_size": 9, "font_weight": "normal", "anchor": "start"},
        "curve_label_neighbor": {"font_size": 9, "font_weight": "normal", "anchor": "start"},
        "annotation": {"font_size": 7, "font_weight": "normal", "anchor": "start"},
    }
    for item in text_items:
        role = str(item.get("role") or "annotation")
        style = role_styles.get(role, role_styles["annotation"])
        item.setdefault("font_size", style["font_size"] * font_scale)
        item.setdefault("font_weight", style["font_weight"])
        item.setdefault("anchor", style["anchor"])
        item.setdefault("font_family", DEFAULT_FONT_FAMILY)
        item.setdefault("dominant_baseline", "alphabetic")
        if role == "title" and "x" not in item:
            item["x"] = float(width) / 2.0

    groups = _baseline_groups(text_items, baseline_tol)
    for group_items in groups.values():
        if not group_items:
            continue
        baselines = []
        for item in group_items:
            try:
                baselines.append(float(item.get("y", 0.0)))
            except (TypeError, ValueError):
                continue
        if not baselines:
            continue
        baseline = sum(baselines) / len(baselines)
        baseline = _snap_grid(baseline, grid)
        for item in group_items:
            try:
                item["y"] = baseline
                item["x"] = _snap_grid(float(item.get("x", 0.0)), grid)
            except (TypeError, ValueError):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, dict):
                continue
            width_val = float(bbox.get("width", 0.0))
            height_val = float(bbox.get("height", 0.0))
            anchor = str(item.get("anchor") or "start")
            if anchor == "middle":
                x0 = float(item.get("x", 0.0)) - width_val / 2.0
            elif anchor == "end":
                x0 = float(item.get("x", 0.0)) - width_val
            else:
                x0 = float(item.get("x", 0.0))
            y0 = baseline - height_val
            x0 = _snap_grid(x0, grid)
            y0 = _snap_grid(y0, grid)
            bbox.update(
                {
                    "x": x0,
                    "y": y0,
                    "width": width_val,
                    "height": height_val,
                    "x0": x0,
                    "y0": y0,
                    "x1": x0 + width_val,
                    "y1": y0 + height_val,
                }
            )


def _estimate_text_color_3gpp(rgba: np.ndarray, bbox: dict[str, Any]) -> str | None:
    try:
        x0 = int(max(float(bbox.get("x", 0.0)), 0.0))
        y0 = int(max(float(bbox.get("y", 0.0)), 0.0))
        x1 = int(float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0)))
        y1 = int(float(bbox.get("y", 0.0)) + float(bbox.get("height", 0.0)))
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    x1 = min(x1, rgba.shape[1])
    y1 = min(y1, rgba.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    region = rgba[y0:y1, x0:x1, :3]
    if region.size == 0:
        return None
    hue, sat, val = _rgb_to_hsv(region)
    mask = (sat > 0.3) & (val > 0.2) & (val < 0.9)
    if mask.sum() < 6:
        return None
    hue_vals = hue[mask]
    median_hue = float(np.median(hue_vals)) if hue_vals.size else None
    if median_hue is None:
        return None
    if _hue_distance(median_hue, 220.0) <= 25.0:
        return "#2b6cb0"
    if _hue_distance(median_hue, 30.0) <= 20.0:
        return "#dd6b20"
    return None


def _assign_text_colors_3gpp(text_items: list[dict[str, Any]], rgba: np.ndarray) -> None:
    for item in text_items:
        if item.get("render") is False:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, dict):
            continue
        color = _estimate_text_color_3gpp(rgba, bbox)
        if color:
            item["fill"] = color


def _assign_roles_lineplot(
    text_items: list[dict[str, Any]], plot: dict[str, Any], width: int, height: int
) -> tuple[str | None, str | None, str | None, list[str]]:
    title: str | None = None
    axis_x: str | None = None
    axis_y: str | None = None
    legend_labels: list[str] = []

    top_candidates = [
        item
        for item in text_items
        if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.1
        and _text_bbox_width(item) > width * 0.5
    ]
    if not top_candidates:
        top_candidates = [
            item
            for item in text_items
            if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.14
            and _text_bbox_width(item) > width * 0.45
        ]
    if top_candidates:
        top_sorted = sorted(
            top_candidates,
            key=lambda item: (
                float(item.get("bbox", {}).get("y", 0.0)),
                float(item.get("bbox", {}).get("x", 0.0)),
            ),
        )
        lines: list[str] = []
        for item in top_sorted:
            text_value = _clean_text(str(item.get("content") or item.get("text") or ""))
            if not text_value:
                continue
            if not lines or text_value != lines[-1]:
                lines.append(text_value)
        if lines:
            title = "\n".join(lines)
        for item in top_candidates:
            item["role"] = "title"
            item["anchor"] = "middle"
            item["x"] = float(width) / 2.0
            item["render"] = False

    plot_x = float(plot.get("x", 0.0))
    plot_y = float(plot.get("y", 0.0))
    plot_w = float(plot.get("width", width))
    plot_h = float(plot.get("height", height))
    for item in text_items:
        if item.get("role") == "title":
            continue
        cx, cy = _text_bbox_center(item)
        if cy > plot_y + plot_h + max(20.0, _text_bbox_height(item)):
            item["role"] = "axis_label_x"
            item["anchor"] = "middle"
            axis_x = str(item.get("content") or item.get("text") or axis_x or "")
            item["render"] = False
        elif cx < plot_x - max(20.0, _text_bbox_width(item)):
            item["role"] = "axis_label_y"
            item["anchor"] = "middle"
            axis_y = str(item.get("content") or item.get("text") or axis_y or "")
            item["render"] = False
        elif cx > plot_x + plot_w * 0.6 and cy < plot_y + plot_h * 0.4:
            item["role"] = "legend"
            item["anchor"] = "start"
            legend_labels.append(str(item.get("content") or item.get("text") or ""))
            item["render"] = False
        else:
            item.setdefault("role", "annotation")
    return title, axis_x, axis_y, legend_labels


def _assign_roles_flow(
    text_items: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> dict[str, str]:
    node_text: dict[str, str] = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        nx = float(node.get("x", 0.0))
        ny = float(node.get("y", 0.0))
        nw = float(node.get("width", 0.0))
        nh = float(node.get("height", 0.0))
        node_items = []
        for item in text_items:
            cx, cy = _text_bbox_center(item)
            if nx <= cx <= nx + nw and ny <= cy <= ny + nh:
                node_items.append(item)
                item["role"] = "node_label"
                item["anchor"] = "start"
                item["render"] = False
        if node_items:
            node_items.sort(key=lambda item: (item.get("y", 0.0), item.get("x", 0.0)))
            text_value = "\n".join(
                str(item.get("content") or item.get("text") or "") for item in node_items
            )
            node_text[str(node_id)] = text_value
    return node_text


def _default_panels(width: int, height: int) -> list[dict[str, Any]]:
    margin_x = max(int(width * 0.05), 20)
    margin_top = max(int(height * 0.2), 40)
    panel_height = max(int(height * 0.6), 100)
    available_width = width - margin_x * 2
    gap = max(int(available_width * 0.05), 20)
    panel_width = int((available_width - 2 * gap) / 3)
    panels = []
    x = margin_x
    for panel_id in ("A3", "A4", "A5"):
        panels.append(
            {
                "id": panel_id,
                "label": panel_id,
                "x": x,
                "y": margin_top,
                "width": panel_width,
                "height": panel_height,
            }
        )
        x += panel_width + gap
    return panels


def _extract_3gpp(
    width: int,
    height: int,
    mask: np.ndarray,
    rgba: np.ndarray,
    text_items: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    min_len = max(int(height * 0.5), 1)
    if adaptive and adaptive.get("lines"):
        min_len = int(adaptive["lines"].get("long_line_min_len_px", min_len))
    long_v = _long_line_positions(mask, axis=0, min_len=min_len)
    panels = _detect_panels(mask, width, height)
    if len(panels) == 3:
        for panel in panels:
            panel.setdefault("label", str(panel.get("id") or ""))
    if len(long_v) < 3:
        warnings.append(
            ExtractIssue(
                code="W4001_PANELS_FALLBACK",
                message="Panel detection incomplete; using default layout.",
                hint="Verify panel bounding boxes and adjust manually if needed.",
            )
        )
    axes_lines: list[dict[str, Any]] = []
    for panel in panels:
        axes_lines.extend(_detect_axes_lines(mask, panel, adaptive))

    dashed_lines = _detect_dashed_lines(mask, adaptive, panels=panels)
    if max(width, height) >= 900:
        for line in dashed_lines:
            line["stroke"] = "#000000"
    markers = _detect_markers(rgba)
    title, title_style, text_items = _assign_roles_3gpp(text_items, panels, width, height)
    for panel in panels:
        candidates = [
            item
            for item in text_items
            if item.get("role") == "panel_label"
            and panel["x"] <= _text_bbox_center(item)[0] <= panel["x"] + panel["width"]
            and panel["y"] <= _text_bbox_center(item)[1] <= panel["y"] + panel["height"] * 0.3
        ]
        if candidates:
            label = None
            label_font_size = None
            panel_id = str(panel.get("id") or "")
            for item in candidates:
                text_value = _clean_text(str(item.get("content") or item.get("text") or ""))
                if not text_value:
                    continue
                candidate = _panel_label_from_text(text_value)
                if candidate == panel_id:
                    label = text_value
                    try:
                        label_font_size = float(item.get("font_size", 0.0))
                    except (TypeError, ValueError):
                        label_font_size = None
                    item["render"] = False
                    break
            if label:
                panel["label"] = label
                if label_font_size:
                    panel["label_font_size"] = label_font_size

    curve_label_panels: set[str] = set()
    for item in text_items:
        if item.get("role") in {"panel_label", "title"}:
            continue
        text_value = _clean_text(str(item.get("content") or item.get("text") or "")).lower()
        if not text_value:
            continue
        cx, cy = _text_bbox_center(item)
        panel_id = None
        for panel in panels:
            if (
                panel["x"] <= cx <= panel["x"] + panel["width"]
                and panel["y"] <= cy <= panel["y"] + panel["height"]
            ):
                panel_id = str(panel.get("id") or "")
                break
        if not panel_id:
            continue
        if "serving" in text_value and "beam" in text_value:
            item["role"] = "curve_label_serving"
            item["anchor"] = "start"
            curve_label_panels.add(panel_id)
        elif "neighbor" in text_value or "target" in text_value:
            item["role"] = "curve_label_neighbor"
            item["anchor"] = "start"
            curve_label_panels.add(panel_id)

    if curve_label_panels:
        for panel in panels:
            panel_id = str(panel.get("id") or "")
            if panel_id in curve_label_panels:
                panel["show_curve_labels"] = False

    _apply_text_layout(text_items, "t_3gpp_events_3panel", width, height, adaptive)
    _assign_text_colors_3gpp(text_items, rgba)

    t_start_ratio = 0.2
    t_trigger_ratio = 0.6
    dashed_vertical = [line for line in dashed_lines if abs(line["x1"] - line["x2"]) <= 0.1]
    if dashed_vertical:
        for panel in panels:
            panel_lines = [
                line
                for line in dashed_vertical
                if panel["x"] <= line["x1"] <= panel["x"] + panel["width"]
            ]
            if len(panel_lines) >= 2:
                panel_lines.sort(key=lambda line: line["x1"])
                t_start_ratio = (panel_lines[0]["x1"] - panel["x"]) / panel["width"]
                t_trigger_ratio = (panel_lines[-1]["x1"] - panel["x"]) / panel["width"]
                break

    include_panel_rects = max(width, height) < 900
    params = {
        "template": "t_3gpp_events_3panel",
        "canvas": {"width": width, "height": height},
        "title": title,
        "panels": panels,
        "t_start_ratio": t_start_ratio,
        "t_trigger_ratio": t_trigger_ratio,
        "curves": {
            "serving": [
                {"x": 0.05, "y": 0.2},
                {"x": 0.5, "y": 0.8},
                {"x": 0.95, "y": 0.4},
            ],
            "neighbor": [
                {"x": 0.05, "y": 0.4},
                {"x": 0.5, "y": 0.3},
                {"x": 0.95, "y": 0.8},
            ],
        },
        "texts": text_items,
        "geometry": {
            "lines": axes_lines + dashed_lines,
            "rects": [
                {
                    "x": panel["x"],
                    "y": panel["y"],
                    "width": panel["width"],
                    "height": panel["height"],
                    "stroke": "#555555",
                    "stroke_width": 1,
                    "fill": "none",
                    "role": "panel",
                }
                for panel in panels
            ]
            if include_panel_rects
            else [],
            "markers": markers,
        },
        "extracted": {
            "axes_candidates": {"vertical": long_v},
            "text_blocks": text_boxes,
            "dash_candidates": [],
        },
    }
    if title_style:
        params["title_style"] = title_style
    overlay = {
        "panels": panels,
        "lines": axes_lines + dashed_lines,
        "markers": markers,
        "text_boxes": text_boxes,
    }
    return params, overlay


def _panel_bounds(panel: dict[str, Any]) -> dict[str, Any]:
    x = float(panel.get("x", 0.0))
    y = float(panel.get("y", 0.0))
    width = float(panel.get("width", 0.0))
    height = float(panel.get("height", 0.0))
    return {
        "id": panel.get("id"),
        "x0": x,
        "y0": y,
        "x1": x + width,
        "y1": y + height,
    }


def _snap_to_edge(value: float, edge: float, tolerance: float) -> float:
    return edge if abs(value - edge) <= tolerance else value


def _panel_axes_from_lines(
    panel: dict[str, Any],
    lines: list[dict[str, Any]],
    tolerance: float = 3.0,
) -> dict[str, Any]:
    x0 = float(panel.get("x", 0.0))
    y0 = float(panel.get("y", 0.0))
    x1 = x0 + float(panel.get("width", 0.0))
    y1 = y0 + float(panel.get("height", 0.0))
    verticals = [
        line
        for line in lines
        if abs(float(line.get("x1", 0.0)) - float(line.get("x2", 0.0))) <= 1.0
        and x0 <= float(line.get("x1", 0.0)) <= x1
        and y0 <= float(line.get("y1", 0.0)) <= y1
    ]
    horizontals = [
        line
        for line in lines
        if abs(float(line.get("y1", 0.0)) - float(line.get("y2", 0.0))) <= 1.0
        and x0 <= float(line.get("x1", 0.0)) <= x1
        and y0 <= float(line.get("y1", 0.0)) <= y1
    ]
    if verticals:
        verticals.sort(key=lambda line: float(line.get("x1", 0.0)))
        y_axis_x = float(verticals[0].get("x1", x0))
        y_axis_width = float(verticals[0].get("stroke_width", 2))
    else:
        y_axis_x = x0
        y_axis_width = 2.0
    if horizontals:
        horizontals.sort(key=lambda line: float(line.get("y1", 0.0)), reverse=True)
        x_axis_y = float(horizontals[0].get("y1", y1))
        x_axis_width = float(horizontals[0].get("stroke_width", 2))
    else:
        x_axis_y = y1
        x_axis_width = 2.0
    y_axis_x = _snap_to_edge(y_axis_x, x0, tolerance)
    x_axis_y = _snap_to_edge(x_axis_y, y1, tolerance)
    return {
        "panel_id": panel.get("id"),
        "y_axis": {"x": y_axis_x, "direction": "up", "stroke_width": y_axis_width},
        "x_axis": {"y": x_axis_y, "direction": "right", "stroke_width": x_axis_width},
    }


def _panel_t_positions(
    panel: dict[str, Any],
    dashed_lines: list[dict[str, Any]],
    default_start: float,
    default_trigger: float,
) -> dict[str, Any]:
    x0 = float(panel.get("x", 0.0))
    width = float(panel.get("width", 0.0))
    verticals = [
        line
        for line in dashed_lines
        if abs(float(line.get("x1", 0.0)) - float(line.get("x2", 0.0))) <= 1.0
        and x0 <= float(line.get("x1", 0.0)) <= x0 + width
    ]
    source = "fallback"
    if len(verticals) >= 2:
        verticals.sort(key=lambda line: float(line.get("x1", 0.0)))
        t_start_x = float(verticals[0].get("x1", x0))
        t_trigger_x = float(verticals[-1].get("x1", x0 + width))
        source = "dashed"
    else:
        t_start_x = x0 + width * default_start
        t_trigger_x = x0 + width * default_trigger
    if t_start_x > t_trigger_x:
        t_start_x, t_trigger_x = t_trigger_x, t_start_x
    t_start_ratio = (t_start_x - x0) / width if width > 0 else default_start
    t_trigger_ratio = (t_trigger_x - x0) / width if width > 0 else default_trigger
    return {
        "panel_id": panel.get("id"),
        "t_start_x": t_start_x,
        "t_trigger_x": t_trigger_x,
        "t_start_ratio": t_start_ratio,
        "t_trigger_ratio": t_trigger_ratio,
        "source": source,
    }


def _estimate_ttt_fill_color(
    rgba: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> str | None:
    x0i = max(int(round(x0)), 0)
    y0i = max(int(round(y0)), 0)
    x1i = min(int(round(x1)), rgba.shape[1])
    y1i = min(int(round(y1)), rgba.shape[0])
    if x1i <= x0i or y1i <= y0i:
        return None
    region = rgba[y0i:y1i, x0i:x1i, :3].astype(np.float32)
    if region.size == 0:
        return None
    mean_rgb = region.reshape(-1, 3).mean(axis=0)
    r, g, b = (mean_rgb / 255.0).tolist()
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    if cmax <= 0:
        return None
    sat = (cmax - cmin) / cmax
    if sat < 0.05:
        return None
    if (b - r) > 0.05 and b >= g:
        return "#d9d3e8"
    return "#f2e6b8"


def _points_to_ratio(
    points: list[tuple[float, float]],
    panel: dict[str, Any],
) -> list[dict[str, float]]:
    x0 = float(panel.get("x", 0.0))
    y0 = float(panel.get("y", 0.0))
    width = float(panel.get("width", 1.0))
    height = float(panel.get("height", 1.0))
    ratio_points: list[dict[str, float]] = []
    for x, y in points:
        if width <= 0 or height <= 0:
            continue
        rx = (x - x0) / width
        ry = 1.0 - (y - y0) / height
        rx = max(0.0, min(1.0, rx))
        ry = max(0.0, min(1.0, ry))
        ratio_points.append({"x": round(rx, 4), "y": round(ry, 4)})
    return ratio_points


def _ratio_to_points(
    points: list[dict[str, Any]],
    panel: dict[str, Any],
) -> list[tuple[float, float]]:
    x0 = float(panel.get("x", 0.0))
    y0 = float(panel.get("y", 0.0))
    width = float(panel.get("width", 0.0))
    height = float(panel.get("height", 0.0))
    out: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            rx = float(point.get("x", 0.0))
            ry = float(point.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        x = x0 + width * rx
        y = y0 + height * (1.0 - ry)
        out.append((x, y))
    return out


def extract_3gpp_events_3panel_v1(
    width: int,
    height: int,
    mask: np.ndarray,
    rgba: np.ndarray,
    text_items: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    separators = _detect_panel_columns(mask)
    params, overlay = _extract_3gpp(
        width, height, mask, rgba, text_items, text_boxes, warnings, adaptive=adaptive
    )
    panels = params.get("panels", [])
    geometry = params.get("geometry", {})
    axes_lines = [
        line for line in geometry.get("lines", []) if isinstance(line, dict) and line.get("role") == "axis"
    ]
    dashed_lines = [
        line
        for line in geometry.get("lines", [])
        if isinstance(line, dict) and line.get("dashed")
    ]
    panel_bounds = sorted((_panel_bounds(panel) for panel in panels), key=lambda item: item["x0"])
    panel_axes = [_panel_axes_from_lines(panel, axes_lines) for panel in panels]
    default_start = float(params.get("t_start_ratio", 0.2) or 0.2)
    default_trigger = float(params.get("t_trigger_ratio", 0.6) or 0.6)
    if default_start >= default_trigger:
        default_start, default_trigger = 0.2, 0.6
    t_positions = [
        _panel_t_positions(panel, dashed_lines, default_start, default_trigger)
        for panel in panels
    ]
    ttt_rects = [
        {
            "panel_id": panel.get("id"),
            "x0": float(pos["t_start_x"]),
            "y0": float(panel.get("y", 0.0)),
            "x1": float(pos["t_trigger_x"]),
            "y1": float(panel.get("y", 0.0)) + float(panel.get("height", 0.0)),
        }
        for panel, pos in zip(panels, t_positions)
    ]

    style = params.get("style")
    if not isinstance(style, dict):
        style = {}
    fill_by_panel: dict[str, str] = {}
    for panel, pos in zip(panels, t_positions):
        panel_id = str(panel.get("id") or "")
        color = _estimate_ttt_fill_color(
            rgba,
            float(pos["t_start_x"]),
            float(panel.get("y", 0.0)),
            float(pos["t_trigger_x"]),
            float(panel.get("y", 0.0)) + float(panel.get("height", 0.0)),
        )
        if color:
            fill_by_panel[panel_id] = color
    if fill_by_panel:
        style["ttt_fill_by_panel"] = fill_by_panel
        style.setdefault("ttt_fill_opacity", 0.35)
    if max(width, height) >= 900:
        style.setdefault("text_color", "#000000")
        style.setdefault("t_line_stroke", "#000000")
        style.setdefault("t_label_fill", "#000000")
    if style:
        params["style"] = style

    curve_cfg = adaptive.get("curves") if adaptive else {}
    curves_by_panel: dict[str, Any] = {}
    curve_points_debug: list[dict[str, Any]] = []
    fallback_curves = params.get("curves") if isinstance(params.get("curves"), dict) else {}

    for panel in panels:
        panel_id = str(panel.get("id") or "")
        x0 = int(panel.get("x", 0.0))
        y0 = int(panel.get("y", 0.0))
        x1 = int(panel.get("x", 0.0) + panel.get("width", 0.0))
        y1 = int(panel.get("y", 0.0) + panel.get("height", 0.0))
        if x1 <= x0 or y1 <= y0:
            continue
        sub = rgba[y0:y1, x0:x1]
        blue_mask = _curve_color_mask(sub, target_hue=220.0, adaptive=curve_cfg)
        orange_mask = _curve_color_mask(sub, target_hue=30.0, adaptive=curve_cfg)
        blue_points_local = _curve_centerline_points(blue_mask, curve_cfg)
        orange_points_local = _curve_centerline_points(orange_mask, curve_cfg)

        def _to_abs(points_local: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [(x + x0, y + y0) for x, y in points_local]

        serving_source = "color"
        neighbor_source = "color"
        serving_abs = _to_abs(blue_points_local)
        neighbor_abs = _to_abs(orange_points_local)
        serving_ratio = _points_to_ratio(serving_abs, panel)
        neighbor_ratio = _points_to_ratio(neighbor_abs, panel)

        if not serving_ratio:
            raw = fallback_curves.get("serving", [])
            if isinstance(raw, list):
                serving_abs = _ratio_to_points(raw, panel)
                serving_ratio = raw
                serving_source = "fallback"
        if not neighbor_ratio:
            raw = fallback_curves.get("neighbor", [])
            if isinstance(raw, list):
                neighbor_abs = _ratio_to_points(raw, panel)
                neighbor_ratio = raw
                neighbor_source = "fallback"

        curves_by_panel[panel_id] = {
            "serving": {
                "points": serving_ratio,
                "stroke": "#2b6cb0",
                "dashed": False,
            },
            "neighbor": {
                "points": neighbor_ratio,
                "stroke": "#dd6b20",
                "dashed": True,
                "dasharray": DEFAULT_DASHARRAY,
            },
        }
        curve_points_debug.append(
            {
                "panel_id": panel_id,
                "curve_id": "serving",
                "points": [{"x": float(x), "y": float(y)} for x, y in serving_abs],
                "points_ratio": serving_ratio,
                "source": serving_source,
            }
        )
        curve_points_debug.append(
            {
                "panel_id": panel_id,
                "curve_id": "neighbor",
                "points": [{"x": float(x), "y": float(y)} for x, y in neighbor_abs],
                "points_ratio": neighbor_ratio,
                "source": neighbor_source,
            }
        )

    if curves_by_panel:
        params["curves_by_panel"] = curves_by_panel
        first_panel_id = str(panels[0].get("id") or "") if panels else ""
        first_curves = curves_by_panel.get(first_panel_id)
        if isinstance(first_curves, dict):
            params["curves"] = {
                "serving": first_curves.get("serving", {}).get("points", []),
                "neighbor": first_curves.get("neighbor", {}).get("points", []),
            }
    params["canvas_w"] = width
    params["canvas_h"] = height
    extracted = params.get("extracted")
    if not isinstance(extracted, dict):
        extracted = {}
        params["extracted"] = extracted
    extracted["panel_separators"] = separators
    extracted["panel_bounds"] = panel_bounds
    extracted["panel_axes"] = panel_axes
    extracted["t_positions"] = t_positions
    extracted["ttt_rects"] = ttt_rects
    extracted["curve_points"] = curve_points_debug
    return params, overlay


def _finalize_3gpp_v1_metadata(params: dict[str, Any]) -> None:
    panels = params.get("panels", [])
    geometry = params.get("geometry", {})
    extracted = params.get("extracted")
    if not isinstance(extracted, dict):
        return
    axes_lines = [
        line
        for line in geometry.get("lines", [])
        if isinstance(line, dict) and line.get("role") == "axis"
    ]
    panel_bounds = sorted((_panel_bounds(panel) for panel in panels), key=lambda item: item["x0"])
    panel_axes = [_panel_axes_from_lines(panel, axes_lines) for panel in panels]
    try:
        t_start_ratio = float(params.get("t_start_ratio", 0.2))
        t_trigger_ratio = float(params.get("t_trigger_ratio", 0.6))
    except (TypeError, ValueError):
        t_start_ratio, t_trigger_ratio = 0.2, 0.6
    if t_start_ratio >= t_trigger_ratio:
        t_start_ratio, t_trigger_ratio = 0.2, 0.6
    t_positions = []
    ttt_rects = []
    existing_positions = extracted.get("t_positions")
    if not isinstance(existing_positions, list) or len(existing_positions) != len(panels):
        existing_positions = [None] * len(panels)
    for panel, existing in zip(panels, existing_positions):
        x0 = float(panel.get("x", 0.0))
        y0 = float(panel.get("y", 0.0))
        width = float(panel.get("width", 0.0))
        height = float(panel.get("height", 0.0))
        ratio_start = t_start_ratio
        ratio_trigger = t_trigger_ratio
        source = "ratio"
        if isinstance(existing, dict):
            try:
                ratio_start = float(existing.get("t_start_ratio", ratio_start))
                ratio_trigger = float(existing.get("t_trigger_ratio", ratio_trigger))
            except (TypeError, ValueError):
                ratio_start, ratio_trigger = t_start_ratio, t_trigger_ratio
            source = str(existing.get("source") or source)
        if ratio_start >= ratio_trigger:
            ratio_start, ratio_trigger = t_start_ratio, t_trigger_ratio
        t_start_x = x0 + width * ratio_start
        t_trigger_x = x0 + width * ratio_trigger
        t_positions.append(
            {
                "panel_id": panel.get("id"),
                "t_start_x": t_start_x,
                "t_trigger_x": t_trigger_x,
                "t_start_ratio": ratio_start,
                "t_trigger_ratio": ratio_trigger,
                "source": source,
            }
        )
        ttt_rects.append(
            {
                "panel_id": panel.get("id"),
                "x0": t_start_x,
                "y0": y0,
                "x1": t_trigger_x,
                "y1": y0 + height,
            }
        )
    canvas = params.get("canvas", {})
    if isinstance(canvas, dict):
        params["canvas_w"] = canvas.get("width")
        params["canvas_h"] = canvas.get("height")
    extracted["panel_bounds"] = panel_bounds
    extracted["panel_axes"] = panel_axes
    extracted["t_positions"] = t_positions
    extracted["ttt_rects"] = ttt_rects


def _default_plot(width: int, height: int) -> dict[str, Any]:
    margin_x = max(int(width * 0.1), 40)
    margin_top = max(int(height * 0.15), 30)
    margin_bottom = max(int(height * 0.2), 40)
    plot_width = max(width - margin_x * 2, 100)
    plot_height = max(height - margin_top - margin_bottom, 80)
    return {
        "x": margin_x,
        "y": margin_top,
        "width": plot_width,
        "height": plot_height,
    }


def _extract_lineplot(
    width: int,
    height: int,
    mask: np.ndarray,
    rgba: np.ndarray,
    text_items: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ratio = 0.5
    if adaptive and adaptive.get("lines"):
        ratio = float(adaptive["lines"].get("long_line_ratio", ratio))
        scale = float(adaptive.get("scale", 1.0))
        ratio = max(ratio * scale, 0.1)
    long_v = _long_line_positions(mask, axis=0, min_len=max(int(height * ratio), 1))
    long_h = _long_line_positions(mask, axis=1, min_len=max(int(width * ratio), 1))
    plot = _default_plot(width, height)
    axes_lines: list[dict[str, Any]] = []
    if long_v and long_h:
        x_axis = min(long_v)
        y_axis = max(long_h)
        plot = {
            "x": float(x_axis),
            "y": float(min(long_h)),
            "width": float(max(width - x_axis - 20, 100)),
            "height": float(max(y_axis - min(long_h), 80)),
        }
        axes_lines.append(
            {
                "x1": float(x_axis),
                "y1": float(plot["y"]),
                "x2": float(x_axis),
                "y2": float(plot["y"] + plot["height"]),
                "stroke": "#000000",
                "stroke_width": 2,
                "role": "axis",
            }
        )
        axes_lines.append(
            {
                "x1": float(x_axis),
                "y1": float(y_axis),
                "x2": float(x_axis + plot["width"]),
                "y2": float(y_axis),
                "stroke": "#000000",
                "stroke_width": 2,
                "role": "axis",
            }
        )
    else:
        warnings.append(
            ExtractIssue(
                code="W4002_AXES_FALLBACK",
                message="Axes detection incomplete; using default plot area.",
                hint="Adjust axes plot area manually if needed.",
            )
        )

    title, axis_x, axis_y, legend_labels = _assign_roles_lineplot(
        text_items, plot, width, height
    )

    _apply_text_layout(text_items, "t_performance_lineplot", width, height, adaptive)

    series_count = 2
    if plot["width"] > 0 and plot["height"] > 0:
        x0 = max(int(plot["x"]), 0)
        y0 = max(int(plot["y"]), 0)
        x1 = min(int(plot["x"] + plot["width"]), width)
        y1 = min(int(plot["y"] + plot["height"]), height)
        sub = rgba[y0:y1, x0:x1, :3]
        sub_mask = _ink_mask(rgba[y0:y1, x0:x1])
        colors = sub[sub_mask]
        if colors.size > 0:
            quantized = (colors // 32) * 32
            unique = {tuple(color.tolist()) for color in quantized}
            series_count = min(max(len(unique), 1), 4)
    curve_cfg = adaptive.get("curves") if adaptive else None
    spacing_px = int(curve_cfg.get("sample_spacing_px", 120)) if curve_cfg else 120
    max_points = int(curve_cfg.get("max_points", 8)) if curve_cfg else 8
    plot_width = float(plot.get("width", width))
    points_count = max(2, min(int(plot_width / max(spacing_px, 1)) + 2, max_points))
    series: list[dict[str, Any]] = []
    for idx in range(series_count):
        label = legend_labels[idx] if idx < len(legend_labels) else f"Series {idx+1}"
        start_y = 0.2 + idx * 0.1
        end_y = 0.8 - idx * 0.1
        if points_count <= 2:
            points = [{"x": 0, "y": start_y}, {"x": 1, "y": end_y}]
        else:
            points = []
            for point_idx in range(points_count):
                x = point_idx / float(points_count - 1)
                y = start_y + (end_y - start_y) * x
                points.append({"x": x, "y": y})
        series.append(
            {
                "id": f"series_{idx+1}",
                "label": label,
                "color": DEFAULT_SERIES_COLORS[idx % len(DEFAULT_SERIES_COLORS)],
                "dashed": idx == 1,
                "stroke_width": 2,
                "points": points,
            }
        )
    params = {
        "template": "t_performance_lineplot",
        "canvas": {"width": width, "height": height},
        "title": title,
        "axes": {
            "plot": plot,
            "x": {"label": axis_x or "", "min": 0, "max": 1, "ticks": [0, 0.5, 1]},
            "y": {"label": axis_y or "", "min": 0, "max": 1, "ticks": [0, 0.5, 1]},
        },
        "series": series,
        "texts": text_items,
        "geometry": {
            "lines": axes_lines,
            "rects": [
                {
                    "x": plot["x"],
                    "y": plot["y"],
                    "width": plot["width"],
                    "height": plot["height"],
                    "stroke": "#555555",
                    "stroke_width": 1,
                    "fill": "none",
                    "role": "plot_area",
                }
            ],
            "markers": [],
        },
        "extracted": {
            "axes_candidates": {"vertical": long_v, "horizontal": long_h},
            "legend_candidates": [],
            "text_blocks": text_boxes,
        },
    }
    overlay = {
        "axes_plot": plot,
        "lines": axes_lines,
        "text_boxes": text_boxes,
    }
    return params, overlay


def _default_nodes(width: int, height: int) -> list[dict[str, Any]]:
    node_width = max(int(width * 0.2), 120)
    node_height = max(int(height * 0.2), 60)
    gap = max(int(width * 0.08), 40)
    start_x = max(int((width - (node_width * 3 + gap * 2)) / 2), 20)
    y = max(int(height * 0.35), 60)
    nodes = []
    for idx in range(3):
        nodes.append(
            {
                "id": f"node_{idx+1}",
                "x": start_x + idx * (node_width + gap),
                "y": y,
                "width": node_width,
                "height": node_height,
                "rx": 8,
                "ry": 8,
                "text": "Unknown",
            }
        )
    return nodes


def _extract_flow(
    width: int,
    height: int,
    mask: np.ndarray,
    text_items: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    min_area = max(int(width * height * 0.01), 200)
    components = _connected_components(mask, min_area=min_area)
    boxes = []
    for comp in components:
        w = comp["width"]
        h = comp["height"]
        if w < width * 0.08 or h < height * 0.08:
            continue
        if w > width * 0.9 or h > height * 0.9:
            continue
        boxes.append(comp)
    boxes.sort(key=lambda item: (item["x"], item["y"]))
    if not boxes:
        warnings.append(
            ExtractIssue(
                code="W4003_NODES_FALLBACK",
                message="No node boxes detected; using default nodes.",
                hint="Adjust node positions and sizes manually if needed.",
            )
        )
        nodes = _default_nodes(width, height)
    else:
        nodes = []
        for idx, box in enumerate(boxes[:6], start=1):
            nodes.append(
                {
                    "id": f"node_{idx}",
                    "x": box["x"],
                    "y": box["y"],
                    "width": box["width"],
                    "height": box["height"],
                    "rx": 8,
                    "ry": 8,
                    "text": "Unknown",
                }
            )
    nodes_sorted = sorted(nodes, key=lambda item: (item["x"], item["y"]))
    edges = []
    for prev, nxt in zip(nodes_sorted, nodes_sorted[1:]):
        edges.append({"from": prev["id"], "to": nxt["id"], "label": "", "dashed": False})
    node_text = _assign_roles_flow(text_items, nodes)
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id in node_text:
            node["text"] = node_text[node_id]

    _apply_text_layout(text_items, "t_procedure_flow", width, height, adaptive)
    nodes_by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    geometry_lines = []
    for edge in edges:
        points = edge.get("points")
        if isinstance(points, list) and points:
            start = points[0]
            end = points[-1]
            x1, y1 = start.get("x", 0), start.get("y", 0)
            x2, y2 = end.get("x", 0), end.get("y", 0)
        else:
            from_node = nodes_by_id.get(edge.get("from"))
            to_node = nodes_by_id.get(edge.get("to"))
            if not isinstance(from_node, dict) or not isinstance(to_node, dict):
                continue
            x1 = float(from_node.get("x", 0)) + float(from_node.get("width", 0)) / 2
            y1 = float(from_node.get("y", 0)) + float(from_node.get("height", 0)) / 2
            x2 = float(to_node.get("x", 0)) + float(to_node.get("width", 0)) / 2
            y2 = float(to_node.get("y", 0)) + float(to_node.get("height", 0)) / 2
        geometry_lines.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "stroke": "#000000",
                "stroke_width": 2,
                "role": "edge",
            }
        )

    params = {
        "template": "t_procedure_flow",
        "canvas": {"width": width, "height": height},
        "title": None,
        "lanes": [],
        "nodes": nodes,
        "edges": edges,
        "texts": text_items,
        "geometry": {
            "lines": geometry_lines,
            "rects": [
                {
                    "x": node["x"],
                    "y": node["y"],
                    "width": node["width"],
                    "height": node["height"],
                    "stroke": "#000000",
                    "stroke_width": 2,
                    "fill": "none",
                    "role": "node",
                }
                for node in nodes
            ],
            "markers": [],
        },
        "extracted": {
            "node_candidates": nodes,
            "arrow_candidates": [],
            "text_blocks": text_boxes,
        },
    }
    overlay = {
        "nodes": nodes,
        "lines": geometry_lines,
        "text_boxes": text_boxes,
    }
    return params, overlay


def _write_debug_artifacts(
    debug_dir: Path,
    rgba: np.ndarray,
    preprocessed: np.ndarray,
    overlay: dict[str, Any],
    ocr: list[dict[str, Any]],
    report: dict[str, Any],
    params: dict[str, Any],
    input_png: Path,
    effective_config: dict[str, Any],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "effective_config.json").write_text(
        json.dumps(effective_config, indent=2, sort_keys=True)
    )
    pre = Image.fromarray(preprocessed, mode="L")
    pre.save(debug_dir / "01_preprocessed.png")
    pre.save(debug_dir / "preprocessed.png")

    overlay_img = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(overlay_img, "RGBA")

    for panel in overlay.get("panels", []):
        draw.rectangle(
            [
                panel["x"],
                panel["y"],
                panel["x"] + panel["width"],
                panel["y"] + panel["height"],
            ],
            outline=(0, 128, 255, 200),
            width=2,
        )
    plot = overlay.get("axes_plot")
    if plot:
        draw.rectangle(
            [plot["x"], plot["y"], plot["x"] + plot["width"], plot["y"] + plot["height"]],
            outline=(0, 200, 0, 200),
            width=2,
        )
    for node in overlay.get("nodes", []):
        draw.rectangle(
            [
                node["x"],
                node["y"],
                node["x"] + node["width"],
                node["y"] + node["height"],
            ],
            outline=(255, 165, 0, 200),
            width=2,
        )
    for line in overlay.get("lines", []):
        try:
            draw.line(
                [(line["x1"], line["y1"]), (line["x2"], line["y2"])],
                fill=(120, 120, 255, 180),
                width=2,
            )
        except Exception:
            continue
    for marker in overlay.get("markers", []):
        try:
            x = marker["x"]
            y = marker["y"]
            r = marker.get("radius", 3)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 180), width=2)
        except Exception:
            continue
    for text in overlay.get("text_boxes", []):
        bbox = text["bbox"]
        draw.rectangle(
            [
                bbox["x"],
                bbox["y"],
                bbox["x"] + bbox["width"],
                bbox["y"] + bbox["height"],
            ],
            outline=(255, 0, 0, 180),
            width=1,
        )

    overlay_img.save(debug_dir / "02_overlay.png")
    overlay_img.save(debug_dir / "overlays.png")
    write_ocr_json(debug_dir / "03_ocr_raw.json", ocr)
    (debug_dir / "04_params.json").write_text(json.dumps(params, indent=2, sort_keys=True))
    (debug_dir / "extracted.json").write_text(json.dumps(params, indent=2, sort_keys=True))
    (debug_dir / "extract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    extracted = params.get("extracted", {})
    curve_points = extracted.get("curve_points") if isinstance(extracted, dict) else None
    if isinstance(curve_points, list) and curve_points:
        (debug_dir / "curves_points.json").write_text(
            json.dumps(curve_points, indent=2, sort_keys=True)
        )
        curves_img = Image.fromarray(rgba, mode="RGBA")
        curves_draw = ImageDraw.Draw(curves_img, "RGBA")
        color_map = {
            "serving": (43, 108, 176, 220),
            "neighbor": (221, 107, 32, 220),
        }
        for curve in curve_points:
            curve_id = str(curve.get("curve_id", ""))
            color = color_map.get(curve_id, (0, 0, 0, 200))
            points = curve.get("points", [])
            if not isinstance(points, list):
                continue
            coords: list[tuple[float, float]] = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    x = float(point.get("x", 0.0))
                    y = float(point.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                coords.append((x, y))
                curves_draw.ellipse([x - 2, y - 2, x + 2, y + 2], outline=color, width=2)
            if len(coords) >= 2:
                curves_draw.line(coords, fill=color, width=2)
        curves_img.save(debug_dir / "overlays_curves.png")

    try:
        from png2svg.renderer import render_svg
        from validators.visual_diff import RasterizeError, rasterize_svg_to_png

        snap_svg = debug_dir / "05_snap_preview.svg"
        snap_png = debug_dir / "05_snap_preview.png"
        snap_params = debug_dir / "04_params.json"
        render_svg(input_png, snap_params, snap_svg)
        try:
            rasterize_svg_to_png(snap_svg, snap_png)
        except RasterizeError:
            pass
    except Exception:
        pass


def extract_skeleton(
    input_png: Path,
    template: str,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    if template == "auto":
        template = classify_png(input_png)["template_id"]
    template_id = TEMPLATE_ALIASES.get(template)
    if not template_id:
        raise Png2SvgError(
            code="E4000_TEMPLATE_UNKNOWN",
            message=f"Unknown template '{template}'.",
            hint="Use one of: t_3gpp_events_3panel, t_procedure_flow, t_performance_lineplot, or auto.",
        )

    rgba, width, height = _load_image(input_png)
    adaptive_config = _load_adaptive_config()
    effective_config = _effective_config(adaptive_config, width, height)
    preprocessed = _preprocess_image(rgba, effective_config)
    mask = _ink_mask(rgba, effective_config)
    text_boxes = _detect_text_boxes(rgba, effective_config)
    warnings: list[ExtractIssue] = []
    errors: list[ExtractIssue] = []
    ocr_backend = os.environ.get("PNG2SVG_OCR_BACKEND", "auto")
    ocr_rois: list[dict[str, int]] | None = None
    if template_id == "t_3gpp_events_3panel":
        panels = _detect_panels(mask, width, height)
        ocr_rois = [
            {"x": 0, "y": 0, "width": width, "height": int(height * 0.18)},
        ]
        for panel in panels:
            ocr_rois.append(
                {
                    "x": int(panel["x"]),
                    "y": int(panel["y"]),
                    "width": int(panel["width"]),
                    "height": int(panel["height"] * 0.25),
                }
            )
            ocr_rois.append(
                {
                    "x": int(panel["x"]),
                    "y": int(panel["y"] + panel["height"] * 0.75),
                    "width": int(panel["width"]),
                    "height": int(panel["height"] * 0.25),
                }
            )
    elif template_id == "t_performance_lineplot":
        plot = _default_plot(width, height)
        ocr_rois = [
            {"x": 0, "y": 0, "width": width, "height": int(height * 0.18)},
            {
                "x": int(plot["x"]),
                "y": int(plot["y"] + plot["height"]),
                "width": int(plot["width"]),
                "height": int(height * 0.2),
            },
            {
                "x": 0,
                "y": int(plot["y"]),
                "width": int(plot["x"]),
                "height": int(plot["height"]),
            },
        ]
    if ocr_rois:
        pad_px = int(effective_config.get("ocr", {}).get("roi_pad_px", 0))
        ocr_rois = [_pad_roi(roi, pad_px, width, height) for roi in ocr_rois]
    backend_value = ocr_backend.lower()
    if backend_value == "pytesseract":
        ocr_available = has_pytesseract()
    elif backend_value == "tesseract":
        ocr_available = has_tesseract()
    elif backend_value == "none":
        ocr_available = False
    else:
        ocr_available = has_pytesseract() or has_tesseract()

    ocr_results: list[dict[str, Any]] = []
    if ocr_available:
        ocr_image_input = _prepare_ocr_image(rgba, effective_config)
        try:
            ocr_results = ocr_image(ocr_image_input, backend=ocr_backend, rois=ocr_rois)
        except ValueError as exc:
            warnings.append(
                ExtractIssue(
                    code="W4012_OCR_BACKEND_INVALID",
                    message=f"OCR backend error: {exc}",
                    hint="Use PNG2SVG_OCR_BACKEND=auto|tesseract|pytesseract|none.",
                )
            )
    text_items = _text_items_from_ocr(ocr_results, width, height, effective_config)
    ocr_cfg = effective_config.get("ocr") if isinstance(effective_config.get("ocr"), dict) else None
    text_items = _filter_text_items(text_items, ocr_cfg)

    if backend_value == "none":
        warnings.append(
            ExtractIssue(
                code="W4010_OCR_DISABLED",
                message="OCR backend set to none; skipping text recognition.",
                hint="Install tesseract or set PNG2SVG_OCR_BACKEND=tesseract.",
            )
        )
    elif not ocr_available:
        warnings.append(
            ExtractIssue(
                code="W4010_OCR_UNAVAILABLE",
                message="Tesseract not available; OCR skipped.",
                hint="Install tesseract-ocr (and optional pytesseract) or set PNG2SVG_OCR_BACKEND=none.",
            )
        )
    elif not ocr_results:
        warnings.append(
            ExtractIssue(
                code="W4011_OCR_EMPTY",
                message="OCR returned no text boxes.",
                hint="Check text contrast or adjust OCR preprocessing.",
            )
        )
    elif not text_items:
        warnings.append(
            ExtractIssue(
                code="W4013_OCR_FILTERED_EMPTY",
                message="OCR text was filtered out after cleanup.",
                hint="Lower OCR thresholds or review preprocessing settings.",
            )
        )

    if not text_boxes:
        warnings.append(
            ExtractIssue(
                code="W4004_OCR_EMPTY",
                message="No OCR text boxes detected.",
                hint="Inspect text regions and fill in labels manually.",
            )
        )

    if template_id == "t_3gpp_events_3panel":
        params, overlay = extract_3gpp_events_3panel_v1(
            width, height, mask, rgba, text_items, text_boxes, warnings, adaptive=effective_config
        )
    elif template_id == "t_performance_lineplot":
        params, overlay = _extract_lineplot(
            width, height, mask, rgba, text_items, text_boxes, warnings, adaptive=effective_config
        )
    elif template_id == "t_procedure_flow":
        params, overlay = _extract_flow(
            width, height, mask, text_items, text_boxes, warnings, adaptive=effective_config
        )
    else:
        errors.append(
            ExtractIssue(
                code="E4002_TEMPLATE_UNSUPPORTED",
                message=f"Template '{template_id}' not supported by extractor.",
                hint="Use a supported template or update the extractor.",
            )
        )
        params = {
            "template": template_id,
            "canvas": {"width": width, "height": height},
        }
        overlay = {}

    extracted = params.get("extracted")
    if not isinstance(extracted, dict):
        extracted = {}
        params["extracted"] = extracted
    extracted["text_items"] = text_items
    extracted["texts_detected"] = _count_renderable_texts(text_items) if ocr_results else 0
    extracted["ocr_backend"] = ocr_backend
    extracted["effective_config"] = effective_config
    params = normalize_params(template_id, params)
    if template_id == "t_3gpp_events_3panel":
        _finalize_3gpp_v1_metadata(params)
    if overlay.get("panels") is not None:
        overlay["panels"] = params.get("panels", overlay.get("panels"))
    if overlay.get("axes_plot") is not None:
        axes = params.get("axes", {})
        if isinstance(axes, dict) and isinstance(axes.get("plot"), dict):
            overlay["axes_plot"] = axes["plot"]
    if overlay.get("nodes") is not None:
        overlay["nodes"] = params.get("nodes", overlay.get("nodes"))
    if overlay.get("text_boxes") is not None:
        extracted = params.get("extracted", {})
        if isinstance(extracted, dict) and isinstance(extracted.get("text_blocks"), list):
            overlay["text_boxes"] = extracted["text_blocks"]

    report = {
        "status": "pass" if not errors else "fail",
        "template_id": template_id,
        "errors": [issue.to_dict() for issue in errors],
        "warnings": [issue.to_dict() for issue in warnings],
    }

    if debug_dir is not None:
        _write_debug_artifacts(
            debug_dir,
            rgba,
            preprocessed,
            overlay,
            ocr_results,
            report,
            params,
            input_png,
            effective_config,
        )

    if errors:
        raise Png2SvgError(
            code=errors[0].code,
            message=errors[0].message,
            hint=errors[0].hint,
        )
    return params
