from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from png2svg.classifier import classify_png
from png2svg.errors import Png2SvgError
from png2svg.normalize import normalize_params


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


def _ink_mask(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return (alpha > 10) & (luminance < 245)


def _preprocess_image(rgba: np.ndarray) -> np.ndarray:
    mask = _ink_mask(rgba)
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


def _detect_text_boxes(rgba: np.ndarray) -> list[dict[str, Any]]:
    mask = _ink_mask(rgba)
    height, width = mask.shape
    min_area = max(int(height * width * 0.00005), 8)
    max_area = max(int(height * width * 0.02), min_area + 1)
    components = _connected_components(mask, min_area=min_area)
    boxes: list[dict[str, Any]] = []
    for comp in components:
        area = comp["area"]
        if area > max_area:
            continue
        w = comp["width"]
        h = comp["height"]
        if w < 3 or h < 3:
            continue
        if w > width * 0.6 or h > height * 0.4:
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
                "bbox": {
                    "x": float(block["min_x"]),
                    "y": float(block["min_y"]),
                    "width": float(block["max_x"] - block["min_x"]),
                    "height": float(block["max_y"] - block["min_y"]),
                },
            }
        )
    return items


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
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
) -> tuple[dict[str, Any], dict[str, Any]]:
    long_v = _long_line_positions(mask, axis=0, min_len=max(int(height * 0.5), 1))
    panels = _default_panels(width, height)
    if len(long_v) < 3:
        warnings.append(
            ExtractIssue(
                code="W4001_PANELS_FALLBACK",
                message="Panel detection incomplete; using default layout.",
                hint="Verify panel bounding boxes and adjust manually if needed.",
            )
        )
    params = {
        "template": "t_3gpp_events_3panel",
        "canvas": {"width": width, "height": height},
        "title": None,
        "panels": panels,
        "t_start_ratio": 0.2,
        "t_trigger_ratio": 0.6,
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
        "extracted": {
            "axes_candidates": {"vertical": long_v},
            "text_blocks": text_boxes,
            "dash_candidates": [],
        },
    }
    overlay = {
        "panels": panels,
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
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
) -> tuple[dict[str, Any], dict[str, Any]]:
    long_v = _long_line_positions(mask, axis=0, min_len=max(int(height * 0.5), 1))
    long_h = _long_line_positions(mask, axis=1, min_len=max(int(width * 0.5), 1))
    plot = _default_plot(width, height)
    if not long_v or not long_h:
        warnings.append(
            ExtractIssue(
                code="W4002_AXES_FALLBACK",
                message="Axes detection incomplete; using default plot area.",
                hint="Adjust axes plot area manually if needed.",
            )
        )
    series = [
        {
            "id": "series_1",
            "label": "Series 1",
            "color": DEFAULT_SERIES_COLORS[0],
            "dashed": False,
            "stroke_width": 2,
            "points": [{"x": 0, "y": 0.2}, {"x": 1, "y": 0.8}],
        },
        {
            "id": "series_2",
            "label": "Series 2",
            "color": DEFAULT_SERIES_COLORS[1],
            "dashed": True,
            "stroke_width": 2,
            "points": [{"x": 0, "y": 0.1}, {"x": 1, "y": 0.6}],
        },
    ]
    params = {
        "template": "t_performance_lineplot",
        "canvas": {"width": width, "height": height},
        "title": None,
        "axes": {
            "plot": plot,
            "x": {"label": "", "min": 0, "max": 1, "ticks": [0, 0.5, 1]},
            "y": {"label": "", "min": 0, "max": 1, "ticks": [0, 0.5, 1]},
        },
        "series": series,
        "extracted": {
            "axes_candidates": {"vertical": long_v, "horizontal": long_h},
            "legend_candidates": [],
            "text_blocks": text_boxes,
        },
    }
    overlay = {
        "axes_plot": plot,
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
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
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
    params = {
        "template": "t_procedure_flow",
        "canvas": {"width": width, "height": height},
        "title": None,
        "lanes": [],
        "nodes": nodes,
        "edges": edges,
        "extracted": {
            "node_candidates": nodes,
            "arrow_candidates": [],
            "text_blocks": text_boxes,
        },
    }
    overlay = {
        "nodes": nodes,
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
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    pre = Image.fromarray(preprocessed, mode="L")
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

    overlay_img.save(debug_dir / "overlay.png")
    (debug_dir / "ocr.json").write_text(json.dumps(ocr, indent=2, sort_keys=True))
    (debug_dir / "extract_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


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
    preprocessed = _preprocess_image(rgba)
    mask = _ink_mask(rgba)
    text_boxes = _detect_text_boxes(rgba)
    text_items = _text_items_from_boxes(text_boxes)
    warnings: list[ExtractIssue] = []
    errors: list[ExtractIssue] = []

    if not text_boxes:
        warnings.append(
            ExtractIssue(
                code="W4004_OCR_EMPTY",
                message="No OCR text boxes detected.",
                hint="Inspect text regions and fill in labels manually.",
            )
        )

    if template_id == "t_3gpp_events_3panel":
        params, overlay = _extract_3gpp(width, height, mask, text_boxes, warnings)
    elif template_id == "t_performance_lineplot":
        params, overlay = _extract_lineplot(width, height, mask, text_boxes, warnings)
    elif template_id == "t_procedure_flow":
        params, overlay = _extract_flow(width, height, mask, text_boxes, warnings)
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
    extracted["texts_detected"] = len(text_items)
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
        _write_debug_artifacts(debug_dir, rgba, preprocessed, overlay, text_boxes, report)

    if errors:
        raise Png2SvgError(
            code=errors[0].code,
            message=errors[0].message,
            hint=errors[0].hint,
        )
    return params
