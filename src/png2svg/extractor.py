from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml

from png2svg.classifier import classify_png
from png2svg.errors import Png2SvgError
from png2svg.normalize import normalize_params
from png2svg.ocr import has_tesseract, ocr_image, write_ocr_json


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
    ocr = {"roi_pad_px": pad_px}

    curves_cfg = config.get("curves", {}) if isinstance(config.get("curves"), dict) else {}
    curve_spacing = int(round(float(curves_cfg.get("sample_spacing_px", 120)) * scale))
    curve_spacing = max(curve_spacing, 20)
    curves = {
        "sample_spacing_px": curve_spacing,
        "max_points": int(curves_cfg.get("max_points", 8)),
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
    for cluster in clusters:
        cluster.sort(key=lambda item: float(item["bbox"]["x"]))
    return clusters


def _text_items_from_ocr(
    ocr_results: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    if not ocr_results:
        return []
    heights = sorted(float(item["bbox"]["height"]) for item in ocr_results)
    median_height = heights[len(heights) // 2]
    tolerance = max(4.0, median_height * 0.6)
    lines = _group_ocr_lines(ocr_results, tolerance)
    items: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        texts = [item["text"] for item in line if item.get("text")]
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
                "bbox": {
                    "x": float(min_x),
                    "y": float(min_y),
                    "width": float(max_x - min_x),
                    "height": float(max_y - min_y),
                },
            }
        )
    return items


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


def _dashed_line_positions(
    mask: np.ndarray,
    axis: int,
    min_len: int,
    max_len: int,
    min_count: int,
) -> list[int]:
    positions: list[int] = []
    if axis == 0:
        length = mask.shape[1]
        for idx in range(length):
            run = 0
            count = 0
            for value in mask[:, idx]:
                if value:
                    run += 1
                else:
                    if min_len <= run <= max_len:
                        count += 1
                    run = 0
            if min_len <= run <= max_len:
                count += 1
            if count >= min_count:
                positions.append(idx)
    else:
        length = mask.shape[0]
        for idx in range(length):
            run = 0
            count = 0
            for value in mask[idx, :]:
                if value:
                    run += 1
                else:
                    if min_len <= run <= max_len:
                        count += 1
                    run = 0
            if min_len <= run <= max_len:
                count += 1
            if count >= min_count:
                positions.append(idx)
    return _cluster_indices(positions)


def _detect_dashed_lines(mask: np.ndarray, adaptive: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    height, width = mask.shape
    dash_cfg = adaptive.get("dashes") if adaptive else None
    if dash_cfg:
        min_len_v = max(int(dash_cfg.get("min_len_px", 2)), 2)
        max_len_v = max(int(dash_cfg.get("max_len_px", min_len_v + 1)), min_len_v + 1)
        min_len_h = min_len_v
        max_len_h = max_len_v
        min_count = int(dash_cfg.get("min_count", 6))
    else:
        min_len_v = max(int(height * 0.03), 2)
        max_len_v = max(int(height * 0.12), min_len_v + 1)
        min_len_h = max(int(width * 0.03), 2)
        max_len_h = max(int(width * 0.12), min_len_h + 1)
        min_count = 6
    dashed_v = _dashed_line_positions(mask, axis=0, min_len=min_len_v, max_len=max_len_v, min_count=min_count)
    dashed_h = _dashed_line_positions(mask, axis=1, min_len=min_len_h, max_len=max_len_h, min_count=min_count)
    lines: list[dict[str, Any]] = []
    for x in dashed_v:
        extent = _line_extent(mask, axis=0, idx=x)
        if not extent:
            continue
        y0, y1 = extent
        lines.append(
            {
                "x1": float(x),
                "y1": float(y0),
                "x2": float(x),
                "y2": float(y1),
                "role": "threshold",
                "stroke": "#555555",
                "stroke_width": 1,
                "dashed": True,
                "dasharray": DEFAULT_DASHARRAY,
            }
        )
    for y in dashed_h:
        extent = _line_extent(mask, axis=1, idx=y)
        if not extent:
            continue
        x0, x1 = extent
        lines.append(
            {
                "x1": float(x0),
                "y1": float(y),
                "x2": float(x1),
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


def _assign_roles_3gpp(
    text_items: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    title: str | None = None
    top_candidates = [
        item for item in text_items if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.18
    ]
    if top_candidates:
        best = max(top_candidates, key=_text_bbox_width)
        title = str(best.get("content") or best.get("text") or "")
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
            if px <= cx <= px + pw and py <= cy <= py + ph * 0.3:
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
    return title, text_items


def _assign_roles_lineplot(
    text_items: list[dict[str, Any]], plot: dict[str, Any], width: int, height: int
) -> tuple[str | None, str | None, str | None, list[str]]:
    title: str | None = None
    axis_x: str | None = None
    axis_y: str | None = None
    legend_labels: list[str] = []

    top_candidates = [
        item for item in text_items if float(item.get("bbox", {}).get("y", 0.0)) < height * 0.18
    ]
    if top_candidates:
        best = max(top_candidates, key=_text_bbox_width)
        title = str(best.get("content") or best.get("text") or "")
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

    dashed_lines = _detect_dashed_lines(mask, adaptive)
    markers = _detect_markers(rgba)
    title, text_items = _assign_roles_3gpp(text_items, panels, width, height)
    for panel in panels:
        candidates = [
            item
            for item in text_items
            if item.get("role") == "panel_label"
            and panel["x"] <= _text_bbox_center(item)[0] <= panel["x"] + panel["width"]
            and panel["y"] <= _text_bbox_center(item)[1] <= panel["y"] + panel["height"] * 0.3
        ]
        if candidates:
            best = max(candidates, key=_text_bbox_width)
            label = str(best.get("content") or best.get("text") or panel["label"])
            panel["label"] = label

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
            ],
            "markers": markers,
        },
        "extracted": {
            "axes_candidates": {"vertical": long_v},
            "text_blocks": text_boxes,
            "dash_candidates": [],
        },
    }
    overlay = {
        "panels": panels,
        "lines": axes_lines + dashed_lines,
        "markers": markers,
        "text_boxes": text_boxes,
    }
    return params, overlay


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
    write_ocr_json(debug_dir / "03_ocr_raw.json", ocr)
    (debug_dir / "04_params.json").write_text(json.dumps(params, indent=2, sort_keys=True))
    (debug_dir / "extract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

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
    ocr_results: list[dict[str, Any]] = []
    if ocr_backend.lower() != "none" and has_tesseract():
        image = Image.fromarray(rgba, mode="RGBA")
        ocr_results = ocr_image(image, backend=ocr_backend, rois=ocr_rois)
    text_items = _text_items_from_ocr(ocr_results, width, height)
    warnings: list[ExtractIssue] = []
    errors: list[ExtractIssue] = []

    if ocr_backend.lower() == "none":
        warnings.append(
            ExtractIssue(
                code="W4010_OCR_DISABLED",
                message="OCR backend set to none; skipping text recognition.",
                hint="Install tesseract or set PNG2SVG_OCR_BACKEND=tesseract.",
            )
        )
    elif not has_tesseract():
        warnings.append(
            ExtractIssue(
                code="W4010_OCR_UNAVAILABLE",
                message="Tesseract not available; OCR skipped.",
                hint="Install tesseract or set PNG2SVG_OCR_BACKEND=none.",
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

    if not text_boxes:
        warnings.append(
            ExtractIssue(
                code="W4004_OCR_EMPTY",
                message="No OCR text boxes detected.",
                hint="Inspect text regions and fill in labels manually.",
            )
        )

    if template_id == "t_3gpp_events_3panel":
        params, overlay = _extract_3gpp(
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
    extracted["texts_detected"] = len(text_items) if ocr_results else 0
    extracted["ocr_backend"] = ocr_backend
    extracted["effective_config"] = effective_config
    params = normalize_params(template_id, params)
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
