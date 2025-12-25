from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.svg_builder import DEFAULT_FONT_FAMILY, SvgBuilder
from png2svg.errors import Png2SvgError


@dataclass(frozen=True)
class Panel:
    panel_id: str
    label: str
    x: float
    y: float
    width: float
    height: float


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
        panels.append(
            Panel(
                panel_id=panel_id,
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
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


def _parse_curve_points(params: dict[str, Any], key: str) -> list[tuple[float, float]]:
    curves = params.get("curves")
    if not isinstance(curves, dict):
        raise Png2SvgError(
            code="E2120_CURVES_MISSING",
            message="curves must be an object with serving and neighbor points.",
            hint="Provide curves.serving and curves.neighbor point arrays.",
        )
    raw_points = curves.get(key)
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise Png2SvgError(
            code="E2121_CURVES_POINTS",
            message=f"curves.{key} must include at least 3 points.",
            hint="Provide at least 3 {x,y} ratio points for the curve.",
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
    return points


def _curve_path(points: list[tuple[float, float]]) -> str:
    if len(points) < 3:
        raise ValueError("Need at least 3 points for cubic segments.")
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
) -> None:
    builder.groups["g_text"].add(
        builder.drawing.text(
            text,
            insert=(x, y),
            id=text_id,
            font_family=DEFAULT_FONT_FAMILY,
            font_size=font_size,
            fill=fill,
        )
    )


def _draw_panel_guides(
    builder: SvgBuilder,
    panel: Panel,
    t_start: float,
    t_trigger: float,
) -> None:
    axes_group = builder.groups["g_axes"]
    annotations_group = builder.groups["g_annotations"]
    markers_group = builder.groups["g_markers"]

    x0 = panel.x
    y0 = panel.y
    x1 = panel.x + panel.width
    y1 = panel.y + panel.height

    axes_group.add(
        builder.drawing.line(
            start=(x0, y1),
            end=(x1, y1),
            stroke="#000000",
            stroke_width=2,
        )
    )
    axes_group.add(
        builder.drawing.line(
            start=(x0, y1),
            end=(x0, y0),
            stroke="#000000",
            stroke_width=2,
        )
    )

    annotations_group.add(
        builder.drawing.line(
            start=(t_start, y1),
            end=(t_start, y0),
            stroke="#555555",
            stroke_width=1,
        )
    )
    annotations_group.add(
        builder.drawing.line(
            start=(t_trigger, y1),
            end=(t_trigger, y0),
            stroke="#555555",
            stroke_width=1,
        )
    )

    markers_group.add(
        builder.drawing.circle(center=(t_start, y0 + 6), r=3, fill="#555555")
    )
    markers_group.add(
        builder.drawing.circle(center=(t_trigger, y0 + 6), r=3, fill="#555555")
    )

    _add_text(
        builder,
        "t_start",
        t_start + 4,
        y0 + 12,
        f"txt_t_start_{panel.panel_id}",
        font_size=10,
        fill="#333333",
    )
    _add_text(
        builder,
        "t_trigger",
        t_trigger + 4,
        y0 + 12,
        f"txt_t_trigger_{panel.panel_id}",
        font_size=10,
        fill="#333333",
    )


def _draw_panel_shading(
    builder: SvgBuilder, panel: Panel, t_start: float, t_trigger: float
) -> None:
    annotations_group = builder.groups["g_annotations"]
    annotations_group.add(
        builder.drawing.rect(
            insert=(t_start, panel.y),
            size=(t_trigger - t_start, panel.height),
            fill="#d0d0d0",
            fill_opacity=0.3,
        )
    )


def _draw_curve(
    builder: SvgBuilder,
    panel: Panel,
    points: list[tuple[float, float]],
    stroke: str,
    curve_id: str,
) -> None:
    curve_points = [_panel_point(panel, point) for point in points[:3]]
    path = _curve_path(curve_points)
    builder.groups["g_curves"].add(
        builder.drawing.path(
            d=path,
            id=curve_id,
            fill="none",
            stroke=stroke,
            stroke_width=2,
        )
    )
    start = curve_points[0]
    end = curve_points[-1]
    markers = builder.groups["g_markers"]
    markers.add(builder.drawing.circle(center=start, r=3, fill=stroke))
    markers.add(builder.drawing.circle(center=end, r=3, fill=stroke))


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    title = params.get("title")
    if title:
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

    serving_points = _parse_curve_points(params, "serving")
    neighbor_points = _parse_curve_points(params, "neighbor")

    for panel in panels:
        t_start = panel.x + panel.width * t_start_ratio
        t_trigger = panel.x + panel.width * t_trigger_ratio

        _draw_panel_shading(builder, panel, t_start, t_trigger)
        _draw_panel_guides(builder, panel, t_start, t_trigger)
        _draw_curve(builder, panel, serving_points, "#2b6cb0", f"curve_serving_{panel.panel_id}")
        _draw_curve(
            builder, panel, neighbor_points, "#dd6b20", f"curve_neighbor_{panel.panel_id}"
        )

        _add_text(
            builder,
            panel.label,
            panel.x + 6,
            panel.y + 16,
            f"txt_panel_{panel.panel_id}",
            font_size=12,
            fill="#000000",
        )

        _add_text(
            builder,
            "Serving",
            panel.x + panel.width * 0.05,
            panel.y + panel.height * 0.35,
            f"txt_serving_{panel.panel_id}",
            font_size=10,
            fill="#2b6cb0",
        )
        _add_text(
            builder,
            "Neighbor",
            panel.x + panel.width * 0.05,
            panel.y + panel.height * 0.55,
            f"txt_neighbor_{panel.panel_id}",
            font_size=10,
            fill="#dd6b20",
        )
