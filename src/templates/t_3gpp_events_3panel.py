from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from common.svg_builder import DEFAULT_FONT_FAMILY, DEFAULT_TEXT_ANCHOR, SvgBuilder
from png2svg.arrow_detector import ArrowType, generate_arrow_points
from png2svg.errors import Png2SvgError


@dataclass(frozen=True)
class Panel:
    panel_id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    label_font_size: float | None = None
    show_curve_labels: bool = True


def _style_value(style: dict[str, Any] | None, key: str, default: Any) -> Any:
    if isinstance(style, dict) and key in style and style[key] is not None:
        return style[key]
    return default


def _require_list(params: dict[str, Any], key: str) -> list[Any]:
    value = params.get(key)
    if not isinstance(value, list):
        raise Png2SvgError(
            code="E2100_PANELS_MISSING",
            message=f"'{key}' must be a list of panel definitions.",
            hint="Provide panels as a list of 3 panel objects.",
        )
    return value


def _parse_panels(params: dict[str, Any], canvas: tuple[int, int]) -> list[Panel]:
    panels_raw = _require_list(params, "panels")
    if len(panels_raw) != 3:
        raise Png2SvgError(
            code="E2101_PANELS_COUNT",
            message="panels must include exactly 3 entries.",
            hint="Provide panels for A3, A4, and A5.",
        )
    panels: list[Panel] = []
    for raw in panels_raw:
        if not isinstance(raw, dict):
            raise Png2SvgError(
                code="E2102_PANEL_TYPE",
                message="Each panel must be an object with geometry.",
                hint="Use keys: id, label, x, y, width, height.",
            )
        panel_id = str(raw.get("id", "")).strip()
        label = str(raw.get("label", panel_id)).strip() or panel_id
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["width"])
            height = float(raw["height"])
        except KeyError as exc:
            raise Png2SvgError(
                code="E2103_PANEL_GEOMETRY",
                message=f"Panel is missing geometry field: {exc}",
                hint="Provide x, y, width, and height for each panel.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2104_PANEL_GEOMETRY",
                message="Panel geometry values must be numeric.",
                hint="Use numeric x, y, width, height values.",
            ) from exc
        if width <= 0 or height <= 0:
            raise Png2SvgError(
                code="E2105_PANEL_GEOMETRY",
                message="Panel width/height must be positive.",
                hint="Use positive panel sizes.",
            )
        label_font_size = raw.get("label_font_size")
        try:
            label_font_size = float(label_font_size) if label_font_size is not None else None
        except (TypeError, ValueError):
            label_font_size = None
        show_curve_labels = bool(raw.get("show_curve_labels", True))
        panels.append(
            Panel(
                panel_id=panel_id,
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
                label_font_size=label_font_size,
                show_curve_labels=show_curve_labels,
            )
        )
    canvas_width, canvas_height = canvas
    for panel in panels:
        if panel.x < 0 or panel.y < 0:
            raise Png2SvgError(
                code="E2106_PANEL_BOUNDS",
                message="Panel positions must be non-negative.",
                hint="Ensure panel x/y are >= 0.",
            )
        if panel.x + panel.width > canvas_width or panel.y + panel.height > canvas_height:
            raise Png2SvgError(
                code="E2107_PANEL_BOUNDS",
                message="Panel exceeds canvas bounds.",
                hint="Ensure panels fit within the canvas.",
            )
    return panels


def _parse_ratio(params: dict[str, Any], key: str) -> float:
    try:
        value = float(params[key])
    except KeyError as exc:
        raise Png2SvgError(
            code="E2110_RATIO_MISSING",
            message=f"Missing required ratio '{key}'.",
            hint="Provide t_start_ratio and t_trigger_ratio values.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise Png2SvgError(
            code="E2111_RATIO_INVALID",
            message=f"Ratio '{key}' must be numeric.",
            hint="Provide ratio values between 0 and 1.",
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise Png2SvgError(
            code="E2112_RATIO_RANGE",
            message=f"Ratio '{key}' must be between 0 and 1.",
            hint="Use a ratio value within [0, 1].",
        )
    return value


def _parse_curve_entry(
    curves: dict[str, Any] | None,
    key: str,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    if not isinstance(curves, dict):
        raise Png2SvgError(
            code="E2120_CURVES_MISSING",
            message="curves must be an object with serving and neighbor points.",
            hint="Provide curves.serving and curves.neighbor point arrays.",
        )
    entry = curves.get(key)
    dashed = False
    dasharray = None
    stroke = None
    raw_points = entry
    if isinstance(entry, dict):
        raw_points = entry.get("points", [])
        dashed = bool(entry.get("dashed", False))
        dasharray = entry.get("dasharray")
        stroke = entry.get("stroke")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise Png2SvgError(
            code="E2121_CURVES_POINTS",
            message=f"curves.{key} must include at least 2 points.",
            hint="Provide at least 2 {x,y} ratio points for the curve.",
        )
    points: list[tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            raise Png2SvgError(
                code="E2122_CURVES_POINTS",
                message="Curve points must be objects with x and y.",
                hint="Provide each point as {x: ratio, y: ratio}.",
            )
        try:
            x = float(point["x"])
            y = float(point["y"])
        except KeyError as exc:
            raise Png2SvgError(
                code="E2123_CURVES_POINTS",
                message=f"Curve point missing field: {exc}",
                hint="Provide x and y for each curve point.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2124_CURVES_POINTS",
                message="Curve point values must be numeric.",
                hint="Provide numeric x/y values between 0 and 1.",
            ) from exc
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise Png2SvgError(
                code="E2125_CURVES_POINTS",
                message="Curve point ratios must be between 0 and 1.",
                hint="Clamp curve point ratios within [0, 1].",
            )
        points.append((x, y))
    return points, {"dashed": dashed, "dasharray": dasharray, "stroke": stroke}


def _curve_path(points: list[tuple[float, float]]) -> str:
    """Convert points to SVG path using optimal Bezier curve fitting.

    Uses the Philip J. Schneider algorithm for high-quality curve fitting
    when points are dense, falls back to Catmull-Rom for sparse points.
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points for a curve path.")

    # Use the new Schneider algorithm for better curve fitting
    try:
        from png2svg.curve_fitting import fit_and_convert_to_path
        # Use tighter error tolerance for smoother curves
        path = fit_and_convert_to_path(points, max_error=1.5)
        if path:
            return path
    except ImportError:
        pass  # Fall back to legacy method

    # Legacy Catmull-Rom based method as fallback
    if len(points) == 2:
        p0, p1 = points[0], points[1]
        c1 = (p0[0] + (p1[0] - p0[0]) / 3.0, p0[1] + (p1[1] - p0[1]) / 3.0)
        c2 = (p0[0] + 2 * (p1[0] - p0[0]) / 3.0, p0[1] + 2 * (p1[1] - p0[1]) / 3.0)
        return (
            f"M {p0[0]:.2f} {p0[1]:.2f} "
            f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p1[0]:.2f} {p1[1]:.2f}"
        )
    if len(points) == 3:
        p0, p1, p2 = points[0], points[1], points[2]
        path = [
            f"M {p0[0]:.2f} {p0[1]:.2f}",
            "C "
            f"{p0[0] + (p1[0] - p0[0]) * 0.35:.2f} {p0[1]:.2f} "
            f"{p0[0] + (p1[0] - p0[0]) * 0.65:.2f} {p1[1]:.2f} "
            f"{p1[0]:.2f} {p1[1]:.2f}",
            "C "
            f"{p1[0] + (p2[0] - p1[0]) * 0.35:.2f} {p1[1]:.2f} "
            f"{p1[0] + (p2[0] - p1[0]) * 0.65:.2f} {p2[1]:.2f} "
            f"{p2[0]:.2f} {p2[1]:.2f}",
        ]
        return " ".join(path)
    segments = []
    for idx in range(len(points) - 1):
        p0 = points[idx - 1] if idx - 1 >= 0 else points[idx]
        p1 = points[idx]
        p2 = points[idx + 1]
        p3 = points[idx + 2] if idx + 2 < len(points) else points[idx + 1]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        segments.append((p1, c1, c2, p2))
    path_cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for _, c1, c2, p2 in segments:
        path_cmds.append(
            f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}"
        )
    return " ".join(path_cmds)


def _panel_point(panel: Panel, point: tuple[float, float]) -> tuple[float, float]:
    return (
        panel.x + panel.width * point[0],
        panel.y + panel.height * (1.0 - point[1]),
    )


def _add_text(
    builder: SvgBuilder,
    text: str,
    x: float,
    y: float,
    text_id: str,
    font_size: int = 12,
    fill: str = "#000000",
    font_weight: str | None = None,
    text_anchor: str = DEFAULT_TEXT_ANCHOR,
    font_family: str = DEFAULT_FONT_FAMILY,
    dominant_baseline: str = "alphabetic",
) -> None:
    kwargs = {
        "insert": (x, y),
        "id": text_id,
        "font_family": font_family,
        "text_anchor": text_anchor,
        "font_size": font_size,
        "fill": fill,
    }
    if font_weight:
        kwargs["font_weight"] = font_weight
    if dominant_baseline:
        kwargs["dominant_baseline"] = dominant_baseline
    builder.groups["g_text"].add(builder.drawing.text(text, **kwargs))


def _add_rotated_text(
    builder: SvgBuilder,
    text: str,
    x: float,
    y: float,
    angle: float,
    text_id: str,
    font_size: int = 12,
    fill: str = "#000000",
    font_weight: str | None = None,
    text_anchor: str = DEFAULT_TEXT_ANCHOR,
    font_family: str = DEFAULT_FONT_FAMILY,
    dominant_baseline: str = "alphabetic",
) -> None:
    kwargs = {
        "insert": (x, y),
        "id": text_id,
        "font_family": font_family,
        "text_anchor": text_anchor,
        "font_size": font_size,
        "fill": fill,
    }
    if font_weight:
        kwargs["font_weight"] = font_weight
    if dominant_baseline:
        kwargs["dominant_baseline"] = dominant_baseline
    text_element = builder.drawing.text(text, **kwargs)
    text_element.rotate(angle, center=(x, y))
    builder.groups["g_text"].add(text_element)


def _add_text_block(
    builder: SvgBuilder,
    text_id: str,
    lines: list[str],
    x: float,
    y: float,
    font_size: int,
    fill: str,
    font_weight: str | None = None,
    text_anchor: str = DEFAULT_TEXT_ANCHOR,
) -> None:
    if not lines:
        return
    if len(lines) == 1:
        _add_text(
            builder,
            lines[0],
            x,
            y,
            text_id,
            font_size=font_size,
            fill=fill,
            font_weight=font_weight,
            text_anchor=text_anchor,
        )
        return
    kwargs = {
        "insert": (x, y),
        "id": text_id,
        "font_family": DEFAULT_FONT_FAMILY,
        "text_anchor": text_anchor,
        "font_size": font_size,
        "fill": fill,
    }
    if font_weight:
        kwargs["font_weight"] = font_weight
    text = builder.drawing.text("", **kwargs)
    line_height = float(font_size) * 1.25
    for idx, line in enumerate(lines):
        if idx == 0:
            tspan = builder.drawing.tspan(line, x=[x], y=[y], id=f"{text_id}_line{idx}")
        else:
            tspan = builder.drawing.tspan(line, x=[x], dy=[line_height], id=f"{text_id}_line{idx}")
        text.add(tspan)
    builder.groups["g_text"].add(text)


def _wrap_lines(lines: list[str], max_chars: int) -> list[str]:
    if max_chars <= 4:
        return lines
    wrapped: list[str] = []
    for line in lines:
        words = [word for word in line.split() if word]
        if not words:
            continue
        current: list[str] = []
        for word in words:
            proposed = (" ".join(current + [word])).strip()
            if len(proposed) > max_chars and current:
                wrapped.append(" ".join(current).strip())
                current = [word]
            else:
                current.append(word)
        if current:
            wrapped.append(" ".join(current).strip())
    return wrapped if wrapped else lines


def _ensure_font_size(value: Any, min_size: float) -> int:
    try:
        size = float(value)
    except (TypeError, ValueError):
        size = float(min_size)
    if min_size > 0:
        size = max(size, min_size)
    return int(round(size))


def _resolve_panel_text_position(
    panels: list[Panel],
    item: dict[str, Any],
) -> tuple[float | None, float | None]:
    panel_id = str(item.get("panel_id") or "")
    if not panel_id:
        return (None, None)
    try:
        rx = float(item.get("x_ratio"))
        ry = float(item.get("y_ratio"))
    except (TypeError, ValueError):
        return (None, None)
    for panel in panels:
        if panel.panel_id == panel_id:
            x = panel.x + panel.width * rx
            y = panel.y + panel.height * ry
            return (x, y)
    return (None, None)


def _text_lines_from_item(item: dict[str, Any]) -> list[str]:
    if isinstance(item.get("lines"), list):
        return [str(line).strip() for line in item.get("lines", []) if str(line).strip()]
    text_value = str(item.get("text") or item.get("content") or "")
    return [line.strip() for line in text_value.splitlines() if line.strip()]


def _render_text_items(
    builder: SvgBuilder,
    panels: list[Panel],
    items: list[dict[str, Any]],
    min_font_size: float,
) -> None:
    for idx, item in enumerate(items):
        if item.get("render") is False:
            continue
        text_id = str(item.get("id") or f"txt_auto_{idx}")
        lines = _text_lines_from_item(item)
        if not lines:
            continue
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            rx, ry = _resolve_panel_text_position(panels, item)
            if rx is None or ry is None:
                continue
            x, y = rx, ry
        try:
            x_val = float(x)
            y_val = float(y)
        except (TypeError, ValueError):
            continue
        font_size = _ensure_font_size(item.get("font_size", 12), min_font_size)
        font_weight = item.get("font_weight")
        anchor = str(item.get("anchor") or DEFAULT_TEXT_ANCHOR)
        fill = str(item.get("fill") or "#000000")
        _add_text_block(
            builder,
            text_id,
            lines,
            x_val,
            y_val,
            font_size=font_size,
            fill=fill,
            font_weight=font_weight,
            text_anchor=anchor,
        )
        item["render"] = False


def _draw_arrow_head(
    builder: SvgBuilder,
    points: list[tuple[float, float]],
    arrow_id: str,
    fill: str,
    arrow_type: str = "triangle",
    stroke: str | None = None,
    stroke_width: float = 1.0,
) -> None:
    """Draw an arrow head with specified type.

    Args:
        builder: SVG builder
        points: Polygon points for the arrow (legacy, used for triangle)
        arrow_id: Unique ID for the arrow element
        fill: Fill color
        arrow_type: Type of arrow (triangle, open, line, dot, diamond, none)
        stroke: Stroke color (for open and line types)
        stroke_width: Stroke width
    """
    group = builder.groups["g_markers"]

    if arrow_type == "none":
        return

    if arrow_type == "dot" and len(points) >= 1:
        # For dot type, use the tip position and draw a circle
        cx, cy = points[0]
        radius = 3.0  # Default small radius
        group.add(builder.drawing.circle(center=(cx, cy), r=radius, fill=fill, id=arrow_id))
        return

    if arrow_type == "open":
        # Open triangle - stroke only, no fill
        group.add(builder.drawing.polygon(
            points=points,
            fill="none",
            stroke=stroke or fill,
            stroke_width=stroke_width,
            id=arrow_id,
        ))
        return

    if arrow_type == "line" and len(points) >= 3:
        # Line arrow (<): two lines from tip
        # Points expected: [left_point, tip, right_point]
        left = points[0]
        tip = points[1]
        right = points[2]
        group.add(builder.drawing.line(
            start=left,
            end=tip,
            stroke=stroke or fill,
            stroke_width=stroke_width,
        ))
        group.add(builder.drawing.line(
            start=tip,
            end=right,
            stroke=stroke or fill,
            stroke_width=stroke_width,
            id=arrow_id,
        ))
        return

    if arrow_type == "diamond":
        # Diamond shape - filled polygon
        group.add(builder.drawing.polygon(points=points, fill=fill, id=arrow_id))
        return

    # Default: filled triangle
    group.add(builder.drawing.polygon(points=points, fill=fill, id=arrow_id))


def _compute_arrow_points_from_line(
    x: float,
    y: float,
    direction: float,
    size: float,
    arrow_type: str = "triangle",
) -> list[tuple[float, float]]:
    """Compute arrow head points from line endpoint.

    Args:
        x, y: Position of arrow tip
        direction: Direction in degrees (0 = right, 90 = down)
        size: Size of arrow head
        arrow_type: Type of arrow

    Returns:
        List of polygon points
    """
    try:
        atype = ArrowType(arrow_type)
    except ValueError:
        atype = ArrowType.TRIANGLE
    return generate_arrow_points((x, y), direction, size, atype)


def _draw_line_with_arrows(
    builder: SvgBuilder,
    line: dict[str, Any],
    group_name: str = "g_axes",
    default_arrow_size: float = 6.0,
) -> None:
    """Draw a line with optional arrow heads at endpoints.

    Args:
        builder: SVG builder
        line: Line dict with x1, y1, x2, y2, and optional arrow_start, arrow_end
        group_name: Name of the group to add the line to
        default_arrow_size: Default arrow size if not specified
    """
    try:
        x1 = float(line.get("x1", 0))
        y1 = float(line.get("y1", 0))
        x2 = float(line.get("x2", 0))
        y2 = float(line.get("y2", 0))
    except (TypeError, ValueError):
        return

    stroke = str(line.get("stroke", "#000000"))
    stroke_width = float(line.get("stroke_width", 2))
    line_id = str(line.get("id", "line"))

    # Draw the main line
    group = builder.groups.get(group_name)
    if group is None:
        return

    line_kwargs: dict[str, Any] = {
        "start": (x1, y1),
        "end": (x2, y2),
        "stroke": stroke,
        "stroke_width": stroke_width,
    }
    dasharray = line.get("dasharray")
    if isinstance(dasharray, list):
        line_kwargs["stroke_dasharray"] = ",".join(str(v) for v in dasharray)
    group.add(builder.drawing.line(**line_kwargs))

    # Calculate line direction
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return

    angle_to_end = math.degrees(math.atan2(dy, dx))
    angle_to_start = (angle_to_end + 180) % 360

    # Draw arrow at start
    arrow_start = line.get("arrow_start", {})
    if isinstance(arrow_start, dict):
        start_type = str(arrow_start.get("type", "none"))
        if start_type != "none":
            start_size = float(arrow_start.get("size", default_arrow_size))
            start_points = _compute_arrow_points_from_line(
                x1, y1, angle_to_start, start_size, start_type
            )
            _draw_arrow_head(
                builder,
                start_points,
                f"{line_id}_arrow_start",
                stroke,
                arrow_type=start_type,
            )

    # Draw arrow at end
    arrow_end = line.get("arrow_end", {})
    if isinstance(arrow_end, dict):
        end_type = str(arrow_end.get("type", "none"))
        if end_type != "none":
            end_size = float(arrow_end.get("size", default_arrow_size))
            end_points = _compute_arrow_points_from_line(
                x2, y2, angle_to_end, end_size, end_type
            )
            _draw_arrow_head(
                builder,
                end_points,
                f"{line_id}_arrow_end",
                stroke,
                arrow_type=end_type,
            )


def _draw_double_arrow(
    builder: SvgBuilder,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    stroke_width: float,
    arrow_size: float,
    arrow_id: str,
) -> None:
    builder.groups["g_annotations"].add(
        builder.drawing.line(start=(x1, y1), end=(x2, y2), stroke=stroke, stroke_width=stroke_width)
    )
    if abs(y1 - y2) <= 1.0:
        left = [(x1, y1), (x1 + arrow_size, y1 - arrow_size / 2), (x1 + arrow_size, y1 + arrow_size / 2)]
        right = [(x2, y2), (x2 - arrow_size, y2 - arrow_size / 2), (x2 - arrow_size, y2 + arrow_size / 2)]
    else:
        left = [(x1, y1), (x1 - arrow_size / 2, y1 + arrow_size), (x1 + arrow_size / 2, y1 + arrow_size)]
        right = [(x2, y2), (x2 - arrow_size / 2, y2 - arrow_size), (x2 + arrow_size / 2, y2 - arrow_size)]
    _draw_arrow_head(builder, left, f"{arrow_id}_start", stroke)
    _draw_arrow_head(builder, right, f"{arrow_id}_end", stroke)


def _draw_annotations(builder: SvgBuilder, annotations: list[dict[str, Any]] | None) -> None:
    if not annotations:
        return
    for idx, entry in enumerate(annotations):
        if not isinstance(entry, dict):
            continue
        raw_text = entry.get("text")
        if not raw_text:
            continue
        text = " ".join(str(raw_text).splitlines())
        try:
            x = float(entry.get("x", 0.0))
            y = float(entry.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        if x <= 0 or y <= 0:
            continue
        text_id = str(entry.get("id") or f"txt_ann_{idx}")
        try:
            font_size = int(float(entry.get("font_size", 10)))
        except (TypeError, ValueError):
            font_size = 10
        anchor = str(entry.get("anchor") or DEFAULT_TEXT_ANCHOR)
        fill = str(entry.get("fill") or "#000000")
        _add_text(
            builder,
            text,
            x,
            y,
            text_id,
            font_size=font_size,
            fill=fill,
            text_anchor=anchor,
        )


def _draw_panel_guides(
    builder: SvgBuilder,
    panel: Panel,
    t_start: float,
    t_trigger: float,
    style: dict[str, Any] | None = None,
) -> None:
    axes_group = builder.groups["g_axes"]
    annotations_group = builder.groups["g_annotations"]
    markers_group = builder.groups["g_markers"]

    axis_stroke = _style_value(style, "axis_stroke", "#000000")
    axis_width = float(_style_value(style, "axis_stroke_width", 2))
    t_line_stroke = _style_value(style, "t_line_stroke", "#555555")
    t_line_width = float(_style_value(style, "t_line_stroke_width", 1))
    text_color = _style_value(style, "text_color", "#000000")
    t_label_fill = _style_value(style, "t_label_fill", text_color)
    guide_dasharray = _style_value(style, "guide_dasharray", None)
    axes_arrows = bool(_style_value(style, "axes_arrows", False))
    axis_arrow_size = float(_style_value(style, "axes_arrow_size", 6.0))
    guide_font_size = float(_style_value(style, "guide_font_size", 10))
    min_font_size = float(_style_value(style, "min_font_size", 0))
    t_label_position = str(_style_value(style, "t_label_position", "top")).lower()
    t_label_offset = float(_style_value(style, "t_label_offset", 12.0))
    show_t_markers = bool(_style_value(style, "show_t_markers", True))

    x0 = panel.x
    y0 = panel.y
    x1 = panel.x + panel.width
    y1 = panel.y + panel.height

    axes_group.add(
        builder.drawing.line(
            start=(x0, y1),
            end=(x1, y1),
            stroke=axis_stroke,
            stroke_width=axis_width,
        )
    )
    axes_group.add(
        builder.drawing.line(
            start=(x0, y1),
            end=(x0, y0),
            stroke=axis_stroke,
            stroke_width=axis_width,
        )
    )
    if axes_arrows:
        arrow_size = axis_arrow_size
        _draw_arrow_head(
            builder,
            [(x1, y1), (x1 - arrow_size, y1 - arrow_size / 2), (x1 - arrow_size, y1 + arrow_size / 2)],
            f"arrow_axis_x_{panel.panel_id}",
            str(axis_stroke),
        )
        _draw_arrow_head(
            builder,
            [(x0, y0), (x0 - arrow_size / 2, y0 + arrow_size), (x0 + arrow_size / 2, y0 + arrow_size)],
            f"arrow_axis_y_{panel.panel_id}",
            str(axis_stroke),
        )

    line_kwargs = {
        "stroke": t_line_stroke,
        "stroke_width": t_line_width,
    }
    if isinstance(guide_dasharray, list):
        line_kwargs["stroke_dasharray"] = ",".join(str(val) for val in guide_dasharray)
    annotations_group.add(
        builder.drawing.line(start=(t_start, y1), end=(t_start, y0), **line_kwargs)
    )
    annotations_group.add(
        builder.drawing.line(start=(t_trigger, y1), end=(t_trigger, y0), **line_kwargs)
    )

    if show_t_markers:
        markers_group.add(
            builder.drawing.circle(center=(t_start, y0 + 6), r=3, fill=t_line_stroke)
        )
        markers_group.add(
            builder.drawing.circle(center=(t_trigger, y0 + 6), r=3, fill=t_line_stroke)
        )

    if t_label_position == "bottom":
        label_y = y1 + t_label_offset
        label_anchor = "middle"
        label_offset = 0.0
    else:
        label_y = y0 + t_label_offset
        label_anchor = DEFAULT_TEXT_ANCHOR
        label_offset = 4.0

    _add_text(
        builder,
        "t_start",
        t_start + label_offset,
        label_y,
        f"txt_t_start_{panel.panel_id}",
        font_size=_ensure_font_size(guide_font_size, min_font_size),
        fill=t_label_fill,
        text_anchor=label_anchor,
    )
    _add_text(
        builder,
        "t_trigger",
        t_trigger + label_offset,
        label_y,
        f"txt_t_trigger_{panel.panel_id}",
        font_size=_ensure_font_size(guide_font_size, min_font_size),
        fill=t_label_fill,
        text_anchor=label_anchor,
    )


def _is_near_white(hex_color: str, threshold: int = 245) -> bool:
    """Check if a hex color is near white using luminance."""
    try:
        hex_val = hex_color.lstrip("#")
        r = int(hex_val[0:2], 16)
        g = int(hex_val[2:4], 16)
        b = int(hex_val[4:6], 16)
        # Calculate luminance (perceived brightness)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance >= threshold
    except (ValueError, IndexError):
        return True


def _draw_panel_background(
    builder: SvgBuilder,
    panel: Panel,
    panel_dict: dict[str, Any],
    style: dict[str, Any] | None = None,
) -> None:
    """Draw panel background fill if extracted from image."""
    if not _style_value(style, "panel_fill_enabled", False):
        return

    fill_color = panel_dict.get("fill")
    if not fill_color:
        return

    # Skip if fill is near-white (threshold 245 means very light colors)
    if _is_near_white(fill_color, threshold=245):
        return

    axes_group = builder.groups["g_axes"]
    axes_group.add(
        builder.drawing.rect(
            insert=(panel.x, panel.y),
            size=(panel.width, panel.height),
            fill=fill_color,
            fill_opacity=0.4,
            id=f"panel_bg_{panel.panel_id}",
        )
    )


def _draw_threshold_bands(
    builder: SvgBuilder,
    panel: Panel,
    panel_dict: dict[str, Any],
    style: dict[str, Any] | None = None,
) -> None:
    """Draw threshold bands (colored horizontal regions) from extracted data."""
    # Check if feature is enabled
    if not _style_value(style, "threshold_bands_enabled", False):
        return

    bands = panel_dict.get("threshold_bands", [])
    if not bands:
        return

    annotations_group = builder.groups["g_annotations"]
    for idx, band in enumerate(bands):
        try:
            band_y = float(band.get("y", 0))
            band_height = float(band.get("height", 0))
            band_fill = str(band.get("fill", "#e0e0e0"))
        except (TypeError, ValueError):
            continue

        if band_height < 10:
            continue

        # Skip light-colored bands (likely background, not actual threshold bands)
        # Use threshold of 200 to only render bands with distinct colors
        if _is_near_white(band_fill, threshold=200):
            continue

        # Convert relative y to absolute
        abs_y = panel.y + band_y

        annotations_group.add(
            builder.drawing.rect(
                insert=(panel.x, abs_y),
                size=(panel.width, band_height),
                fill=band_fill,
                fill_opacity=0.3,
                id=f"threshold_band_{panel.panel_id}_{idx}",
            )
        )


def _draw_panel_shading(
    builder: SvgBuilder,
    panel: Panel,
    t_start: float,
    t_trigger: float,
    style: dict[str, Any] | None = None,
) -> None:
    annotations_group = builder.groups["g_annotations"]
    fill_by_panel = _style_value(style, "ttt_fill_by_panel", {})
    fill_color = None
    if isinstance(fill_by_panel, dict):
        fill_color = fill_by_panel.get(panel.panel_id)
    fill_color = fill_color or _style_value(style, "ttt_fill", "#d0d0d0")
    fill_opacity = float(_style_value(style, "ttt_fill_opacity", 0.3))
    annotations_group.add(
        builder.drawing.rect(
            insert=(t_start, panel.y),
            size=(t_trigger - t_start, panel.height),
            fill=fill_color,
            fill_opacity=fill_opacity,
        )
    )


def _draw_curve(
    builder: SvgBuilder,
    panel: Panel,
    points: list[tuple[float, float]],
    stroke: str,
    curve_id: str,
    dashed: bool = False,
    dasharray: list[float] | None = None,
    show_markers: bool = True,
) -> None:
    curve_points = [_panel_point(panel, point) for point in points]
    path = _curve_path(curve_points)
    path_kwargs = {
        "d": path,
        "id": curve_id,
        "fill": "none",
        "stroke": stroke,
        "stroke_width": 2,
    }
    if dashed:
        dasharray = dasharray or [6, 4]
        path_kwargs["stroke_dasharray"] = ",".join(str(value) for value in dasharray)
        path_kwargs["stroke_linecap"] = "round"
        path_kwargs["class_"] = "dashed"
    builder.groups["g_curves"].add(builder.drawing.path(**path_kwargs))
    if show_markers and curve_points:
        start = curve_points[0]
        end = curve_points[-1]
        markers = builder.groups["g_markers"]
        markers.add(builder.drawing.circle(center=start, r=3, fill=stroke))
        markers.add(builder.drawing.circle(center=end, r=3, fill=stroke))


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    style = params.get("style") if isinstance(params.get("style"), dict) else None
    title_style = params.get("title_style") if isinstance(params.get("title_style"), dict) else None
    title = params.get("title")
    if title:
        if title_style:
            builder.add_title(
                str(title),
                x=float(title_style.get("x", 10)),
                y=float(title_style.get("y", 20)),
                font_size=title_style.get("font_size"),
                font_weight=title_style.get("font_weight"),
                anchor=str(title_style.get("anchor") or DEFAULT_TEXT_ANCHOR),
            )
        else:
            builder.add_title(str(title))
    panels = _parse_panels(params, canvas)
    t_start_ratio = _parse_ratio(params, "t_start_ratio")
    t_trigger_ratio = _parse_ratio(params, "t_trigger_ratio")
    if t_start_ratio >= t_trigger_ratio:
        raise Png2SvgError(
            code="E2113_RATIO_ORDER",
            message="t_start_ratio must be less than t_trigger_ratio.",
            hint="Adjust ratios so t_start is before t_trigger.",
        )

    curves_by_panel = params.get("curves_by_panel") if isinstance(params.get("curves_by_panel"), dict) else {}
    default_curves = params.get("curves") if isinstance(params.get("curves"), dict) else None
    thresholds_by_panel = params.get("thresholds_by_panel") if isinstance(params.get("thresholds_by_panel"), list) else []
    text_blocks = params.get("text_blocks") if isinstance(params.get("text_blocks"), list) else []
    extra_texts = params.get("texts") if isinstance(params.get("texts"), list) else []
    guide_arrows = params.get("guide_arrows") if isinstance(params.get("guide_arrows"), list) else []
    markers = params.get("markers") if isinstance(params.get("markers"), list) else []
    axis_labels = params.get("axis_labels") if isinstance(params.get("axis_labels"), dict) else None
    min_font_size = float(_style_value(style, "min_font_size", 0))
    curve_marker_mode = str(_style_value(style, "curve_markers", "auto")).lower()
    wrap_text_blocks = bool(_style_value(style, "wrap_text_blocks", False))

    # Get panel dicts for color information
    panel_dicts = params.get("panels") if isinstance(params.get("panels"), list) else []
    panel_dict_by_id = {str(p.get("id", "")): p for p in panel_dicts if isinstance(p, dict)}

    for panel in panels:
        t_start = panel.x + panel.width * t_start_ratio
        t_trigger = panel.x + panel.width * t_trigger_ratio

        # Get panel dict for color information
        panel_dict = panel_dict_by_id.get(panel.panel_id, {})

        # Draw background and threshold bands first (behind other elements)
        _draw_panel_background(builder, panel, panel_dict, style=style)
        _draw_threshold_bands(builder, panel, panel_dict, style=style)

        _draw_panel_shading(builder, panel, t_start, t_trigger, style=style)
        _draw_panel_guides(builder, panel, t_start, t_trigger, style=style)

        panel_curves = curves_by_panel.get(panel.panel_id) if curves_by_panel else None
        if not isinstance(panel_curves, dict):
            panel_curves = default_curves
        serving_points, serving_style = _parse_curve_entry(panel_curves, "serving")
        neighbor_points, neighbor_style = _parse_curve_entry(panel_curves, "neighbor")

        serving_stroke = serving_style.get("stroke") or "#2b6cb0"
        neighbor_stroke = neighbor_style.get("stroke") or "#dd6b20"

        _draw_curve(
            builder,
            panel,
            serving_points,
            str(serving_stroke),
            f"curve_serving_{panel.panel_id}",
            dashed=bool(serving_style.get("dashed")),
            dasharray=serving_style.get("dasharray"),
            show_markers=curve_marker_mode != "none",
        )
        _draw_curve(
            builder,
            panel,
            neighbor_points,
            str(neighbor_stroke),
            f"curve_neighbor_{panel.panel_id}",
            dashed=bool(neighbor_style.get("dashed", False)),
            dasharray=neighbor_style.get("dasharray"),
            show_markers=curve_marker_mode != "none",
        )

        for threshold in thresholds_by_panel:
            if not isinstance(threshold, dict):
                continue
            if str(threshold.get("panel_id") or "") != panel.panel_id:
                continue
            try:
                y_ratio = float(threshold.get("y_ratio", 0.0))
            except (TypeError, ValueError):
                continue
            y = panel.y + panel.height * (1.0 - y_ratio)
            dash = threshold.get("dasharray") or [4, 4]
            stroke = str(threshold.get("stroke") or "#888888")
            builder.groups["g_annotations"].add(
                builder.drawing.line(
                    start=(panel.x, y),
                    end=(panel.x + panel.width, y),
                    stroke=stroke,
                    stroke_width=1,
                    stroke_dasharray=",".join(str(val) for val in dash),
                )
            )
            label = threshold.get("label")
            if label:
                _add_text(
                    builder,
                    str(label),
                    panel.x + panel.width * 0.02,
                    y - 4,
                    f"txt_threshold_{panel.panel_id}_{label}",
                    font_size=_ensure_font_size(threshold.get("font_size", 12), min_font_size),
                    fill=stroke,
                )

        for arrow in guide_arrows:
            if not isinstance(arrow, dict):
                continue
            if str(arrow.get("panel_id") or "") != panel.panel_id:
                continue
            arrow_type = str(arrow.get("type") or "")
            stroke = str(arrow.get("stroke") or "#000000")
            arrow_size = float(arrow.get("arrow_size", 6.0))
            stroke_width = float(arrow.get("stroke_width", 1.0))
            if arrow_type == "ttt":
                y_val = arrow.get("y")
                if y_val is None:
                    try:
                        y_ratio = float(arrow.get("y_ratio", 0.2))
                        y_val = panel.y + panel.height * y_ratio
                    except (TypeError, ValueError):
                        y_val = panel.y + panel.height * 0.2
                _draw_double_arrow(
                    builder,
                    t_start,
                    float(y_val),
                    t_trigger,
                    float(y_val),
                    stroke,
                    stroke_width,
                    arrow_size,
                    f"arrow_ttt_{panel.panel_id}",
                )
                label = arrow.get("label") or "TTT"
                _add_text(
                    builder,
                    str(label),
                    (t_start + t_trigger) / 2.0,
                    float(y_val) - 6,
                    f"txt_ttt_{panel.panel_id}",
                    font_size=_ensure_font_size(arrow.get("font_size", 12), min_font_size),
                    fill=stroke,
                    text_anchor="middle",
                )
            elif arrow_type == "hys":
                x_val = arrow.get("x")
                if x_val is None:
                    try:
                        x_ratio = float(arrow.get("x_ratio", 0.5))
                        x_val = panel.x + panel.width * x_ratio
                    except (TypeError, ValueError):
                        x_val = panel.x + panel.width * 0.6
                length = float(arrow.get("length", panel.height * 0.12))
                y_center = arrow.get("y") or (panel.y + panel.height * 0.5)
                y1 = float(y_center) - length / 2
                y2 = float(y_center) + length / 2
                _draw_double_arrow(
                    builder,
                    float(x_val),
                    y1,
                    float(x_val),
                    y2,
                    stroke,
                    stroke_width,
                    arrow_size,
                    f"arrow_hys_{panel.panel_id}",
                )
                label = arrow.get("label") or "Hys"
                _add_text(
                    builder,
                    str(label),
                    float(x_val) + 6,
                    float(y_center),
                    f"txt_hys_{panel.panel_id}",
                    font_size=_ensure_font_size(arrow.get("font_size", 12), min_font_size),
                    fill=stroke,
                    text_anchor="start",
                )

        _add_text(
            builder,
            panel.label,
            panel.x + 6,
            panel.y + 16,
            f"txt_panel_{panel.panel_id}",
            font_size=_ensure_font_size(
                panel.label_font_size or _style_value(style, "panel_label_font_size", 12),
                min_font_size,
            ),
            fill=_style_value(style, "text_color", "#000000"),
            font_weight="bold",
        )

        show_default_labels = bool(_style_value(style, "show_curve_labels", panel.show_curve_labels))
        if show_default_labels:
            serving_label = _style_value(style, "curve_label_serving", "Serving")
            neighbor_label = _style_value(style, "curve_label_neighbor", "Neighbor")
            # Use the same stroke colors for labels to maintain consistency
            _add_text(
                builder,
                str(serving_label),
                panel.x + panel.width * 0.05,
                panel.y + panel.height * 0.35,
                f"txt_serving_{panel.panel_id}",
                font_size=_ensure_font_size(10, min_font_size),
                fill=str(serving_stroke),
            )
            _add_text(
                builder,
                str(neighbor_label),
                panel.x + panel.width * 0.05,
                panel.y + panel.height * 0.55,
                f"txt_neighbor_{panel.panel_id}",
                font_size=_ensure_font_size(10, min_font_size),
                fill=str(neighbor_stroke),
            )

        if axis_labels:
            axis_fill = str(axis_labels.get("fill") or "#000000")
            axis_font_size = _ensure_font_size(axis_labels.get("font_size", 12), min_font_size)
            x_label = axis_labels.get("x")
            y_label = axis_labels.get("y")
            x_offset = float(axis_labels.get("x_offset", 24))
            y_offset = float(axis_labels.get("y_offset", 32))
            if x_label:
                _add_text(
                    builder,
                    str(x_label),
                    panel.x + panel.width * 0.5,
                    panel.y + panel.height + x_offset,
                    f"txt_axis_x_{panel.panel_id}",
                    font_size=axis_font_size,
                    fill=axis_fill,
                    text_anchor="middle",
                )
            if y_label:
                _add_rotated_text(
                    builder,
                    str(y_label),
                    panel.x - y_offset,
                    panel.y + panel.height * 0.5,
                    -90,
                    f"txt_axis_y_{panel.panel_id}",
                    font_size=axis_font_size,
                    fill=axis_fill,
                    text_anchor="middle",
                    dominant_baseline="middle",
                )

    if text_blocks:
        for idx, block in enumerate(text_blocks):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id") or f"txt_block_{idx}")
            lines = block.get("lines") if isinstance(block.get("lines"), list) else []
            if not lines:
                text_value = str(block.get("text") or "")
                lines = [line.strip() for line in text_value.splitlines() if line.strip()]
            if not lines:
                continue
            x = block.get("x")
            y = block.get("y")
            if x is None or y is None:
                rx, ry = _resolve_panel_text_position(panels, block)
                if rx is None or ry is None:
                    continue
                x, y = rx, ry
            try:
                x_val = float(x)
                y_val = float(y)
            except (TypeError, ValueError):
                continue
            font_size = _ensure_font_size(block.get("font_size", 12), min_font_size)
            fill = str(block.get("fill") or "#000000")
            font_weight = block.get("font_weight")
            anchor = str(block.get("anchor") or DEFAULT_TEXT_ANCHOR)
            if wrap_text_blocks:
                panel_id = str(block.get("panel_id") or "")
                panel = next((p for p in panels if p.panel_id == panel_id), None)
                if panel is not None:
                    available = panel.width * 0.9
                    if anchor == "start":
                        available = max(panel.x + panel.width * 0.95 - x_val, panel.width * 0.5)
                    max_chars = int(available / max(font_size * 0.6, 1.0))
                    lines = _wrap_lines([str(line).strip() for line in lines if str(line).strip()], max_chars)
            _add_text_block(
                builder,
                block_id,
                [str(line).strip() for line in lines if str(line).strip()],
                x_val,
                y_val,
                font_size=font_size,
                fill=fill,
                font_weight=font_weight,
                text_anchor=anchor,
            )

    annotations = params.get("annotations")
    if isinstance(annotations, list):
        _render_text_items(builder, panels, annotations, min_font_size)
    if extra_texts:
        _render_text_items(builder, panels, extra_texts, min_font_size)

    if markers:
        for idx, marker in enumerate(markers):
            if not isinstance(marker, dict):
                continue
            try:
                x = float(marker.get("x", 0.0))
                y = float(marker.get("y", 0.0))
                r = float(marker.get("radius", 3.0))
            except (TypeError, ValueError):
                continue
            fill = str(marker.get("fill") or "#dd6b20")
            stroke = str(marker.get("stroke") or "#000000")
            stroke_width = float(marker.get("stroke_width", 2.0))
            builder.groups["g_markers"].add(
                builder.drawing.circle(
                    center=(x, y),
                    r=r,
                    fill=fill,
                    stroke=stroke,
                    stroke_width=stroke_width,
                    id=str(marker.get("id") or f"marker_{idx}"),
                )
            )
