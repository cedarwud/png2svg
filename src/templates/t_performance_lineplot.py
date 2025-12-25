from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from common.svg_builder import DEFAULT_FONT_FAMILY, SvgBuilder
from png2svg.errors import Png2SvgError


AXIS_STROKE = "#000000"
TICK_STROKE = "#000000"
LABEL_COLOR = "#000000"
TICK_LENGTH = 6
FONT_SIZE = 12
TICK_FONT_SIZE = 10


@dataclass(frozen=True)
class Tick:
    value: float
    label: str


@dataclass(frozen=True)
class Axis:
    label: str
    ticks: list[Tick]
    min_value: float
    max_value: float


@dataclass(frozen=True)
class PlotArea:
    origin_x: float
    origin_y: float
    width: float
    height: float
    axis_x: Axis
    axis_y: Axis


@dataclass(frozen=True)
class Series:
    series_id: str
    label: str
    color: str
    dashed: bool
    stroke_width: float
    points: list[tuple[float, float]]


def _require_dict(params: dict[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key)
    if not isinstance(value, dict):
        raise Png2SvgError(
            code="E2300_AXIS_MISSING",
            message=f"'{key}' must be an object.",
            hint=f"Provide an object for '{key}'.",
        )
    return value


def _parse_ticks(raw_ticks: Any, axis_name: str) -> list[Tick]:
    if raw_ticks is None:
        return []
    if not isinstance(raw_ticks, list):
        raise Png2SvgError(
            code="E2301_TICK_TYPE",
            message=f"{axis_name} ticks must be a list.",
            hint="Provide tick definitions as a list.",
        )
    ticks: list[Tick] = []
    for idx, raw in enumerate(raw_ticks):
        if isinstance(raw, dict):
            if "value" not in raw:
                raise Png2SvgError(
                    code="E2302_TICK_VALUE",
                    message=f"{axis_name} tick missing value.",
                    hint="Provide tick value for each tick.",
                )
            try:
                value = float(raw["value"])
            except (TypeError, ValueError) as exc:
                raise Png2SvgError(
                    code="E2303_TICK_VALUE",
                    message=f"{axis_name} tick value must be numeric.",
                    hint="Use numeric values for ticks.",
                ) from exc
            label = str(raw.get("label", raw["value"]))
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise Png2SvgError(
                    code="E2304_TICK_VALUE",
                    message=f"{axis_name} tick value must be numeric.",
                    hint="Use numeric values for ticks.",
                ) from exc
            label = str(raw)
        ticks.append(Tick(value=value, label=label))
    if not ticks:
        raise Png2SvgError(
            code="E2305_TICK_EMPTY",
            message=f"{axis_name} ticks must not be empty.",
            hint="Provide at least one tick per axis.",
        )
    return ticks


def _resolve_range(values: list[float], explicit_min: Any, explicit_max: Any, axis: str) -> tuple[float, float]:
    if explicit_min is not None:
        try:
            min_value = float(explicit_min)
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2306_AXIS_RANGE",
                message=f"{axis} min must be numeric.",
                hint="Provide numeric axis min/max values.",
            ) from exc
    else:
        min_value = min(values)
    if explicit_max is not None:
        try:
            max_value = float(explicit_max)
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2307_AXIS_RANGE",
                message=f"{axis} max must be numeric.",
                hint="Provide numeric axis min/max values.",
            ) from exc
    else:
        max_value = max(values)
    if math.isclose(max_value, min_value):
        raise Png2SvgError(
            code="E2308_AXIS_RANGE",
            message=f"{axis} range must be non-zero.",
            hint="Provide distinct min and max values.",
        )
    return min_value, max_value


def _parse_axes(params: dict[str, Any], series_points: list[tuple[float, float]]) -> PlotArea:
    axes = _require_dict(params, "axes")
    plot = _require_dict(axes, "plot")

    try:
        origin_x = float(plot["x"])
        origin_y = float(plot["y"])
        width = float(plot["width"])
        height = float(plot["height"])
    except KeyError as exc:
        raise Png2SvgError(
            code="E2310_PLOT_GEOMETRY",
            message=f"Plot area missing field: {exc}",
            hint="Provide plot.x, plot.y, plot.width, plot.height.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise Png2SvgError(
            code="E2311_PLOT_GEOMETRY",
            message="Plot area geometry must be numeric.",
            hint="Use numeric values for plot geometry.",
        ) from exc

    axis_x_raw = _require_dict(axes, "x")
    axis_y_raw = _require_dict(axes, "y")

    ticks_x = _parse_ticks(axis_x_raw.get("ticks"), "x")
    ticks_y = _parse_ticks(axis_y_raw.get("ticks"), "y")

    x_values = [point[0] for point in series_points] + [tick.value for tick in ticks_x]
    y_values = [point[1] for point in series_points] + [tick.value for tick in ticks_y]
    if not x_values or not y_values:
        raise Png2SvgError(
            code="E2312_AXIS_VALUES",
            message="Series points required to define axis ranges.",
            hint="Provide series points or explicit axis min/max values.",
        )

    x_min, x_max = _resolve_range(x_values, axis_x_raw.get("min"), axis_x_raw.get("max"), "x")
    y_min, y_max = _resolve_range(y_values, axis_y_raw.get("min"), axis_y_raw.get("max"), "y")

    axis_x = Axis(label=str(axis_x_raw.get("label", "")), ticks=ticks_x, min_value=x_min, max_value=x_max)
    axis_y = Axis(label=str(axis_y_raw.get("label", "")), ticks=ticks_y, min_value=y_min, max_value=y_max)

    return PlotArea(
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
        axis_x=axis_x,
        axis_y=axis_y,
    )


def _parse_series(params: dict[str, Any]) -> list[Series]:
    series_raw = params.get("series")
    if not isinstance(series_raw, list) or not series_raw:
        raise Png2SvgError(
            code="E2320_SERIES_MISSING",
            message="series must be a non-empty list.",
            hint="Provide series definitions with points.",
        )
    series_list: list[Series] = []
    for raw in series_raw:
        if not isinstance(raw, dict):
            raise Png2SvgError(
                code="E2321_SERIES_TYPE",
                message="Each series must be an object.",
                hint="Use {id,label,points} for each series.",
            )
        series_id = str(raw.get("id", "")).strip()
        if not series_id:
            raise Png2SvgError(
                code="E2322_SERIES_ID",
                message="Series id is required.",
                hint="Provide a unique series id.",
            )
        label = str(raw.get("label", series_id))
        color = str(raw.get("color", "#1f77b4"))
        dashed = bool(raw.get("dashed", False))
        stroke_width = float(raw.get("stroke_width", 2))
        points_raw = raw.get("points")
        if not isinstance(points_raw, list) or len(points_raw) < 2:
            raise Png2SvgError(
                code="E2323_SERIES_POINTS",
                message="Series points must include at least 2 points.",
                hint="Provide at least two {x,y} points per series.",
            )
        points: list[tuple[float, float]] = []
        for point in points_raw:
            if not isinstance(point, dict):
                raise Png2SvgError(
                    code="E2324_SERIES_POINTS",
                    message="Series point must be an object with x/y.",
                    hint="Provide {x, y} for each point.",
                )
            try:
                x = float(point["x"])
                y = float(point["y"])
            except KeyError as exc:
                raise Png2SvgError(
                    code="E2325_SERIES_POINTS",
                    message=f"Series point missing field: {exc}",
                    hint="Provide x and y for each point.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise Png2SvgError(
                    code="E2326_SERIES_POINTS",
                    message="Series point values must be numeric.",
                    hint="Provide numeric x/y values.",
                ) from exc
            points.append((x, y))
        series_list.append(
            Series(
                series_id=series_id,
                label=label,
                color=color,
                dashed=dashed,
                stroke_width=stroke_width,
                points=points,
            )
        )
    return series_list


def _map_point(plot: PlotArea, point: tuple[float, float]) -> tuple[float, float]:
    x = plot.origin_x + (point[0] - plot.axis_x.min_value) / (
        plot.axis_x.max_value - plot.axis_x.min_value
    ) * plot.width
    y = plot.origin_y + plot.height - (point[1] - plot.axis_y.min_value) / (
        plot.axis_y.max_value - plot.axis_y.min_value
    ) * plot.height
    return (x, y)


def _draw_axes(builder: SvgBuilder, plot: PlotArea) -> None:
    axes = builder.groups["g_axes"]
    x0 = plot.origin_x
    y0 = plot.origin_y
    x1 = plot.origin_x + plot.width
    y1 = plot.origin_y + plot.height
    axes.add(
        builder.drawing.line(
            start=(x0, y1), end=(x1, y1), stroke=AXIS_STROKE, stroke_width=2
        )
    )
    axes.add(
        builder.drawing.line(
            start=(x0, y1), end=(x0, y0), stroke=AXIS_STROKE, stroke_width=2
        )
    )


def _draw_ticks_and_labels(builder: SvgBuilder, plot: PlotArea) -> None:
    axes = builder.groups["g_axes"]
    text_group = builder.groups["g_text"]
    x_axis_y = plot.origin_y + plot.height
    x_tick_y = x_axis_y
    x_label_y = x_axis_y + TICK_LENGTH + 12

    for idx, tick in enumerate(plot.axis_x.ticks):
        x = plot.origin_x + (tick.value - plot.axis_x.min_value) / (
            plot.axis_x.max_value - plot.axis_x.min_value
        ) * plot.width
        axes.add(
            builder.drawing.line(
                start=(x, x_tick_y),
                end=(x, x_tick_y + TICK_LENGTH),
                stroke=TICK_STROKE,
                stroke_width=1,
            )
        )
        text_group.add(
            builder.drawing.text(
                tick.label,
                insert=(x, x_label_y),
                id=f"txt_tick_x_{idx}",
                font_family=DEFAULT_FONT_FAMILY,
                font_size=TICK_FONT_SIZE,
                fill=LABEL_COLOR,
                text_anchor="middle",
            )
        )

    y_axis_x = plot.origin_x
    y_label_x = y_axis_x - TICK_LENGTH - 4
    for idx, tick in enumerate(plot.axis_y.ticks):
        y = plot.origin_y + plot.height - (tick.value - plot.axis_y.min_value) / (
            plot.axis_y.max_value - plot.axis_y.min_value
        ) * plot.height
        axes.add(
            builder.drawing.line(
                start=(y_axis_x - TICK_LENGTH, y),
                end=(y_axis_x, y),
                stroke=TICK_STROKE,
                stroke_width=1,
            )
        )
        text_group.add(
            builder.drawing.text(
                tick.label,
                insert=(y_label_x, y + 3),
                id=f"txt_tick_y_{idx}",
                font_family=DEFAULT_FONT_FAMILY,
                font_size=TICK_FONT_SIZE,
                fill=LABEL_COLOR,
                text_anchor="end",
            )
        )


def _draw_axis_labels(builder: SvgBuilder, plot: PlotArea) -> None:
    text_group = builder.groups["g_text"]
    if plot.axis_x.label:
        text_group.add(
            builder.drawing.text(
                plot.axis_x.label,
                insert=(plot.origin_x + plot.width / 2, plot.origin_y + plot.height + 32),
                id="txt_axis_x",
                font_family=DEFAULT_FONT_FAMILY,
                font_size=FONT_SIZE,
                fill=LABEL_COLOR,
                text_anchor="middle",
            )
        )
    if plot.axis_y.label:
        text_group.add(
            builder.drawing.text(
                plot.axis_y.label,
                insert=(plot.origin_x - 40, plot.origin_y + plot.height / 2),
                id="txt_axis_y",
                font_family=DEFAULT_FONT_FAMILY,
                font_size=FONT_SIZE,
                fill=LABEL_COLOR,
                text_anchor="middle",
            )
        )


def _draw_series(builder: SvgBuilder, plot: PlotArea, series_list: list[Series]) -> None:
    curves = builder.groups["g_curves"]
    for series in series_list:
        points = [_map_point(plot, point) for point in series.points]
        poly = builder.drawing.polyline(
            points=points,
            fill="none",
            stroke=series.color,
            stroke_width=series.stroke_width,
            id=f"series_{series.series_id}",
        )
        if series.dashed:
            poly.update({"stroke-dasharray": "6,4"})
        curves.add(poly)


def _draw_legend(builder: SvgBuilder, plot: PlotArea, series_list: list[Series]) -> None:
    annotations = builder.groups["g_annotations"]
    legend = builder.drawing.g(id="legend")
    legend_x = plot.origin_x + plot.width - 140
    legend_y = plot.origin_y + 10
    spacing = 18

    for idx, series in enumerate(series_list):
        y = legend_y + idx * spacing
        legend.add(
            builder.drawing.line(
                start=(legend_x, y),
                end=(legend_x + 24, y),
                stroke=series.color,
                stroke_width=series.stroke_width,
            )
        )
        if series.dashed:
            legend.elements[-1].update({"stroke-dasharray": "6,4"})
        legend.add(
            builder.drawing.text(
                series.label,
                insert=(legend_x + 30, y + 4),
                id=f"txt_legend_{series.series_id}",
                font_family=DEFAULT_FONT_FAMILY,
                font_size=TICK_FONT_SIZE,
                fill=LABEL_COLOR,
            )
        )
    annotations.add(legend)


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    title = params.get("title")
    if title:
        builder.add_title(str(title))

    series_list = _parse_series(params)
    all_points = [point for series in series_list for point in series.points]
    plot = _parse_axes(params, all_points)

    _draw_axes(builder, plot)
    _draw_ticks_and_labels(builder, plot)
    _draw_axis_labels(builder, plot)
    _draw_series(builder, plot, series_list)
    _draw_legend(builder, plot, series_list)
