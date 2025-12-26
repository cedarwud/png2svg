from __future__ import annotations

from typing import Any

import numpy as np

from png2svg.extractor_constants import DEFAULT_DASHARRAY, DEFAULT_SERIES_COLORS
from png2svg.extractor_curves import _curve_centerline_points, _curve_color_mask
from png2svg.extractor_geometry import (
    _detect_axes_lines,
    _detect_dashed_lines,
    _detect_markers,
    _detect_panels,
    _long_line_positions,
)
from png2svg.extractor_text import (
    _apply_text_layout,
    _assign_roles_3gpp,
    _assign_roles_flow,
    _assign_roles_lineplot,
    _assign_text_colors_3gpp,
    _enforce_curve_label_colors_3gpp,
    _merge_stacked_text_items,
    _normalize_curve_labels_3gpp,
    _normalize_panel_mid_text,
    _panel_label_from_text,
    _text_bbox_center,
    _text_bbox_width,
)
from png2svg.extractor_types import ExtractIssue


def _default_panels(width: int, height: int) -> list[dict[str, Any]]:
    margin_top = int(height * 0.18)
    margin_bottom = int(height * 0.12)
    margin_x = int(width * 0.04)
    available_width = width - margin_x * 2
    gap = max(int(width * 0.02), 8)
    panel_width = (available_width - gap * 2) / 3.0
    panel_height = height - margin_top - margin_bottom
    panels: list[dict[str, Any]] = []
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


def _extract_project_architecture_v1(
    width: int,
    height: int,
    adaptive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params: dict[str, Any] = {
        "template": "t_project_architecture_v1",
        "canvas": {"width": width, "height": height},
        "title": "Project Architecture",
        "subtitle": "Work Packages (WP1-WP4)",
        "panels": [
            {
                "id": "A",
                "title": "Panel A: Core Platform",
                "bullets": ["Common services", "Interfaces and APIs", "Scalable runtime"],
            },
            {
                "id": "B",
                "title": "Panel B: Data and Analytics",
                "bullets": ["Ingestion and storage", "Analytics pipelines", "Dashboards"],
            },
            {
                "id": "C",
                "title": "Panel C: Integration",
                "bullets": ["External systems", "Security and compliance", "Deployment ops"],
            },
        ],
        "work_packages": [
            {
                "id": "WP1",
                "title": "WP1",
                "goal": "Requirements and scope",
                "output": "Architecture brief",
            },
            {
                "id": "WP2",
                "title": "WP2",
                "goal": "Core platform build",
                "output": "MVP services",
            },
            {
                "id": "WP3",
                "title": "WP3",
                "goal": "Data pipeline and UI",
                "output": "Reports and dashboards",
            },
            {
                "id": "WP4",
                "title": "WP4",
                "goal": "Integration and rollout",
                "output": "Release package",
            },
        ],
    }
    return params, {}


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
    if len(panels) != 3:
        panels = _default_panels(width, height)
        warnings.append(
            ExtractIssue(
                code="W4001_PANELS_FALLBACK",
                message="Panel detection incomplete; using default layout.",
                hint="Verify panel bounding boxes and adjust manually if needed.",
            )
        )
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
    text_items = _merge_stacked_text_items(text_items, "panel_mid_")
    _normalize_panel_mid_text(text_items)
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
                text_value = str(item.get("content") or item.get("text") or "").strip()
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

    curve_label_panels: dict[str, set[str]] = {}
    for item in text_items:
        if item.get("role") in {"panel_label", "title"}:
            continue
        text_value = str(item.get("content") or item.get("text") or "").lower()
        if not text_value:
            continue
        cx, cy = _text_bbox_center(item)
        panel_id = None
        for panel in panels:
            if (
                panel["x"] <= cx <= panel["x"] + panel["width"]
                and panel["y"] <= cy <= panel["y"] + panel["height"]
            ):
                mid_top = panel["y"] + panel["height"] * 0.25
                mid_bottom = panel["y"] + panel["height"] * 0.8
                if mid_top <= cy <= mid_bottom:
                    panel_id = str(panel.get("id") or "")
                break
        if not panel_id:
            continue
        normalized = text_value.replace("/", " ")
        if "beam" in normalized:
            if "neighbor" in normalized or "target" in normalized:
                item["role"] = "curve_label_neighbor"
                item["anchor"] = "start"
                item["content"] = "Neighbor/Target beam"
                item["text"] = item["content"]
                curve_label_panels.setdefault(panel_id, set()).add("neighbor")
            elif "serv" in normalized or normalized.startswith("s") or normalized == "beam":
                item["role"] = "curve_label_serving"
                item["anchor"] = "start"
                item["content"] = "Serving beam"
                item["text"] = item["content"]
                curve_label_panels.setdefault(panel_id, set()).add("serving")
        elif "neighbor" in normalized or "target" in normalized:
            item["role"] = "curve_label_neighbor"
            item["anchor"] = "start"
            curve_label_panels.setdefault(panel_id, set()).add("neighbor")

    if curve_label_panels:
        for panel in panels:
            panel_id = str(panel.get("id") or "")
            labels = curve_label_panels.get(panel_id)
            if not labels:
                continue
            panel["show_curve_labels"] = False
            if "serving" in labels and "neighbor" not in labels:
                text_items.append(
                    {
                        "content": "Neighbor/Target beam",
                        "text": "Neighbor/Target beam",
                        "x": panel["x"] + panel["width"] * 0.05,
                        "y": panel["y"] + panel["height"] * 0.55,
                        "role": "curve_label_neighbor",
                        "anchor": "start",
                        "fill": "#dd6b20",
                    }
                )
            elif "neighbor" in labels and "serving" not in labels:
                text_items.append(
                    {
                        "content": "Serving beam",
                        "text": "Serving beam",
                        "x": panel["x"] + panel["width"] * 0.05,
                        "y": panel["y"] + panel["height"] * 0.35,
                        "role": "curve_label_serving",
                        "anchor": "start",
                        "fill": "#2b6cb0",
                    }
                )

    _normalize_curve_labels_3gpp(text_items)
    _apply_text_layout(text_items, "t_3gpp_events_3panel", width, height, adaptive)
    _assign_text_colors_3gpp(text_items, rgba)
    _enforce_curve_label_colors_3gpp(text_items)

    t_start_ratio = 0.2
    t_trigger_ratio = 0.6
    dashed_vertical = [line for line in dashed_lines if abs(line["x1"] - line["x2"]) <= 0.1]
    if dashed_vertical:
        dashed_vertical.sort(key=lambda line: float(line["x1"]))
        if len(dashed_vertical) >= 2:
            t_start_ratio = float(dashed_vertical[0]["x1"]) / width
            t_trigger_ratio = float(dashed_vertical[-1]["x1"]) / width
            t_start_ratio = max(0.05, min(0.95, t_start_ratio))
            t_trigger_ratio = max(0.05, min(0.95, t_trigger_ratio))

    curves_by_panel: dict[str, Any] = {}
    if adaptive is None:
        adaptive = {}
    curve_cfg = adaptive.get("curves", {}) if isinstance(adaptive.get("curves"), dict) else {}
    for panel in panels:
        panel_id = str(panel.get("id") or "")
        panel_bounds = _panel_bounds(panel)
        panel_mask = mask[
            int(panel_bounds["y"]) : int(panel_bounds["y"] + panel_bounds["height"]),
            int(panel_bounds["x"]) : int(panel_bounds["x"] + panel_bounds["width"]),
        ]
        serving_mask = _curve_color_mask(rgba, 220.0, curve_cfg)
        neighbor_mask = _curve_color_mask(rgba, 30.0, curve_cfg)
        serving_crop = serving_mask[
            int(panel_bounds["y"]) : int(panel_bounds["y"] + panel_bounds["height"]),
            int(panel_bounds["x"]) : int(panel_bounds["x"] + panel_bounds["width"]),
        ]
        neighbor_crop = neighbor_mask[
            int(panel_bounds["y"]) : int(panel_bounds["y"] + panel_bounds["height"]),
            int(panel_bounds["x"]) : int(panel_bounds["x"] + panel_bounds["width"]),
        ]
        serving_points = _curve_centerline_points(serving_crop, curve_cfg)
        neighbor_points = _curve_centerline_points(neighbor_crop, curve_cfg)
        curves_by_panel[panel_id] = {
            "serving": {
                "points": _points_to_ratio(serving_points, panel),
                "stroke": "#2b6cb0",
            },
            "neighbor": {
                "points": _points_to_ratio(neighbor_points, panel),
                "stroke": "#dd6b20",
                "dashed": True,
                "dasharray": DEFAULT_DASHARRAY,
            },
        }

    params = {
        "template": "t_3gpp_events_3panel",
        "canvas": {"width": width, "height": height},
        "title": title,
        "title_style": title_style,
        "t_start_ratio": t_start_ratio,
        "t_trigger_ratio": t_trigger_ratio,
        "panels": panels,
        "curves_by_panel": curves_by_panel,
        "texts": text_items,
        "axes": {"lines": axes_lines},
        "dashed_lines": dashed_lines,
        "markers": markers,
        "geometry": {"lines": axes_lines + dashed_lines, "rects": [], "markers": markers},
        "extracted": {
            "text_blocks": text_boxes,
            "curve_points": [],
            "dashed_lines": dashed_lines,
            "axes_lines": axes_lines,
        },
        "style": {
            "show_curve_labels": True,
            "ttt_fill": None,
        },
    }

    ttt_fill = None
    if panels:
        fill_colors: dict[str, str] = {}
        for panel in panels:
            t_positions = _panel_t_positions(panel, dashed_lines, t_start_ratio, t_trigger_ratio)
            fill = _estimate_ttt_fill_color(
                rgba,
                t_positions["t_start_x"],
                panel["y"],
                t_positions["t_trigger_x"],
                panel["y"] + panel["height"],
            )
            if fill:
                fill_colors[str(panel.get("id") or "")] = fill
        if fill_colors:
            params["style"]["ttt_fill_by_panel"] = fill_colors
            params["style"]["ttt_fill_opacity"] = 0.35
        else:
            ttt_fill = _estimate_ttt_fill_color(
                rgba,
                width * t_start_ratio,
                min(panel["y"] for panel in panels),
                width * t_trigger_ratio,
                max(panel["y"] + panel["height"] for panel in panels),
            )
            params["style"]["ttt_fill"] = ttt_fill

    overlay = {
        "panels": panels,
        "axes_plot": None,
        "lines": axes_lines + dashed_lines,
        "markers": markers,
        "text_boxes": text_boxes,
    }
    return params, overlay


def _panel_bounds(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": float(panel.get("x", 0.0)),
        "y": float(panel.get("y", 0.0)),
        "width": float(panel.get("width", 0.0)),
        "height": float(panel.get("height", 0.0)),
    }


def _snap_to_edge(value: float, edge: float, tolerance: float) -> float:
    if abs(value - edge) <= tolerance:
        return edge
    return value


def _panel_axes_from_lines(
    panel: dict[str, Any],
    axes_lines: list[dict[str, Any]],
    tolerance: float = 3.0,
) -> dict[str, Any]:
    x0 = float(panel.get("x", 0.0))
    y0 = float(panel.get("y", 0.0))
    x1 = float(panel.get("x", 0.0)) + float(panel.get("width", 0.0))
    y1 = float(panel.get("y", 0.0)) + float(panel.get("height", 0.0))
    y_axis_x = x0
    x_axis_y = y1
    y_axis_width = 2
    x_axis_width = 2
    for line in axes_lines:
        try:
            x1_line = float(line.get("x1", 0.0))
            x2_line = float(line.get("x2", 0.0))
            y1_line = float(line.get("y1", 0.0))
            y2_line = float(line.get("y2", 0.0))
        except (TypeError, ValueError):
            continue
        if abs(x1_line - x2_line) < 1.0 and x0 <= x1_line <= x1:
            y_axis_x = x1_line
            y_axis_width = int(line.get("stroke_width", 2))
        if abs(y1_line - y2_line) < 1.0 and y0 <= y1_line <= y1:
            x_axis_y = y1_line
            x_axis_width = int(line.get("stroke_width", 2))
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
    width = float(panel.get("width", 1.0))
    height = float(panel.get("height", 1.0))
    out: list[tuple[float, float]] = []
    for point in points:
        try:
            rx = float(point.get("x", 0.0))
            ry = float(point.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        x = x0 + rx * width
        y = y0 + (1.0 - ry) * height
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
    params, overlay = _extract_3gpp(
        width,
        height,
        mask,
        rgba,
        text_items,
        text_boxes,
        warnings,
        adaptive=adaptive,
    )
    return params, overlay


def _finalize_3gpp_v1_metadata(params: dict[str, Any]) -> None:
    extracted = params.get("extracted")
    if not isinstance(extracted, dict):
        extracted = {}
        params["extracted"] = extracted
    panels = params.get("panels", [])
    dashed = params.get("dashed_lines", [])
    axes = params.get("axes", {}).get("lines", [])
    t_start_ratio = float(params.get("t_start_ratio", 0.2))
    t_trigger_ratio = float(params.get("t_trigger_ratio", 0.6))
    if isinstance(panels, list):
        extracted["panels_detected"] = len(panels)
        panel_bounds: list[dict[str, Any]] = []
        panel_axes: list[dict[str, Any]] = []
        t_positions: list[dict[str, Any]] = []
        for panel in panels:
            panel_id = str(panel.get("id") or "")
            bounds = _panel_bounds(panel)
            panel_bounds.append(
                {
                    "id": panel_id,
                    "x0": bounds["x"],
                    "y0": bounds["y"],
                    "x1": bounds["x"] + bounds["width"],
                    "y1": bounds["y"] + bounds["height"],
                }
            )
            panel_axes.append(_panel_axes_from_lines(panel, axes))
            t_values = _panel_t_positions(panel, dashed, t_start_ratio, t_trigger_ratio)
            t_positions.append(
                {
                    "panel_id": panel_id,
                    "t_start_x": t_values["t_start_x"],
                    "t_trigger_x": t_values["t_trigger_x"],
                }
            )
        extracted["panel_bounds"] = panel_bounds
        extracted["panel_axes"] = panel_axes
        extracted["t_positions"] = t_positions
    if isinstance(dashed, list):
        extracted["dashed_lines_detected"] = len(dashed)
    if isinstance(axes, list):
        extracted["axes_lines_detected"] = len(axes)


def _default_plot(width: int, height: int) -> dict[str, Any]:
    margin_left = int(width * 0.12)
    margin_right = int(width * 0.08)
    margin_top = int(height * 0.18)
    margin_bottom = int(height * 0.12)
    return {
        "x": margin_left,
        "y": margin_top,
        "width": width - margin_left - margin_right,
        "height": height - margin_top - margin_bottom,
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
    plot = _default_plot(width, height)
    title, axis_x, axis_y, legend_labels = _assign_roles_lineplot(text_items, plot, width, height)
    _apply_text_layout(text_items, "t_performance_lineplot", width, height, adaptive)

    curve_cfg = adaptive.get("curves", {}) if adaptive else {}
    curve_points: list[dict[str, Any]] = []
    curves = []
    series_colors = DEFAULT_SERIES_COLORS
    for idx, color in enumerate(series_colors):
        hue = [220.0, 30.0, 120.0, 0.0][idx % 4]
        mask_color = _curve_color_mask(rgba, hue, curve_cfg)
        points = _curve_centerline_points(mask_color, curve_cfg)
        if not points:
            continue
        curve_points.append({"curve_id": f"series_{idx}", "points": [{"x": x, "y": y} for x, y in points]})
        curves.append(
            {
                "id": f"series_{idx}",
                "points": _points_to_ratio(points, plot),
                "stroke": color,
                "dashed": idx % 2 == 1,
                "dasharray": DEFAULT_DASHARRAY if idx % 2 == 1 else None,
            }
        )

    if not curves:
        curves = [
            {
                "id": "series_0",
                "points": [{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 0.5}],
                "stroke": series_colors[0],
                "dashed": False,
                "dasharray": None,
            }
        ]

    axis_label_x = axis_x or "x"
    axis_label_y = axis_y or "y"
    axes = {
        "plot": plot,
        "x": {"label": axis_label_x, "ticks": [0, 0.5, 1.0], "min": 0.0, "max": 1.0},
        "y": {"label": axis_label_y, "ticks": [0, 0.5, 1.0], "min": 0.0, "max": 1.0},
    }

    params = {
        "template": "t_performance_lineplot",
        "canvas": {"width": width, "height": height},
        "title": title,
        "axis_x": axis_x,
        "axis_y": axis_y,
        "series": curves,
        "legend": legend_labels,
        "texts": text_items,
        "axes": axes,
        "geometry": {"lines": [], "rects": [], "markers": []},
        "extracted": {"curve_points": curve_points, "text_blocks": text_boxes},
    }
    overlay = {"axes_plot": plot, "text_boxes": text_boxes}
    return params, overlay


def _default_nodes(width: int, height: int) -> list[dict[str, Any]]:
    node_width = int(width * 0.22)
    node_height = int(height * 0.16)
    gap_x = int(width * 0.06)
    start_x = int(width * 0.1)
    start_y = int(height * 0.2)
    nodes = []
    for idx in range(3):
        nodes.append(
            {
                "id": f"n{idx+1}",
                "x": start_x + idx * (node_width + gap_x),
                "y": start_y,
                "width": node_width,
                "height": node_height,
                "label": f"Node {idx+1}",
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
    nodes = _default_nodes(width, height)
    text_items = _assign_roles_flow(text_items, nodes)
    _apply_text_layout(text_items, "t_procedure_flow", width, height, adaptive)

    edges = []
    geometry_lines = []
    for idx in range(len(nodes) - 1):
        start = nodes[idx]
        end = nodes[idx + 1]
        x1 = start["x"] + start["width"]
        y1 = start["y"] + start["height"] / 2
        x2 = end["x"]
        y2 = end["y"] + end["height"] / 2
        edges.append(
            {
                "from": start["id"],
                "to": end["id"],
                "dashed": False,
            }
        )
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
