from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.svg_builder import DEFAULT_FONT_FAMILY, DEFAULT_TEXT_ANCHOR, SvgBuilder
from png2svg.errors import Png2SvgError

STROKE = "#000000"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


def _split_lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _layout(width: int, height: int, rows: int, cols: int) -> dict[str, Any]:
    margin_x = max(int(width * 0.06), 36)
    margin_y = max(int(height * 0.06), 28)
    header_height = max(int(height * 0.12), 70)
    gap_x = max(int(width * 0.04), 18)
    gap_y = max(int(height * 0.05), 20)
    panel_width = (width - 2 * margin_x - gap_x * (cols - 1)) / cols
    panel_height = (height - margin_y - header_height - gap_y * (rows - 1)) / rows
    panel_rects: list[Rect] = []
    start_y = margin_y + header_height
    for row in range(rows):
        y = start_y + row * (panel_height + gap_y)
        for col in range(cols):
            x = margin_x + col * (panel_width + gap_x)
            panel_rects.append(Rect(x=x, y=y, width=panel_width, height=panel_height))
    return {
        "margin_x": margin_x,
        "margin_y": margin_y,
        "header_height": header_height,
        "panel_rects": panel_rects,
    }


def _parse_layout(params: dict[str, Any]) -> tuple[int, int]:
    layout = params.get("layout")
    if isinstance(layout, str):
        if layout == "2x2":
            return 2, 2
        if layout == "1x3":
            return 1, 3
    grid = params.get("grid")
    if isinstance(grid, dict):
        try:
            rows = int(grid.get("rows", 2))
            cols = int(grid.get("cols", 2))
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E2501_GRID_INVALID",
                message=f"Invalid grid rows/cols: {exc}",
                hint="Provide numeric grid.rows and grid.cols.",
            ) from exc
        return max(rows, 1), max(cols, 1)
    return 2, 2


def _panel_title(panel: dict[str, Any], fallback: str) -> str:
    title = panel.get("title") or panel.get("label") or fallback
    return str(title).strip()


def _series_defaults() -> list[dict[str, Any]]:
    return [
        {
            "id": "series_0",
            "points": [{"x": 0.0, "y": 0.2}, {"x": 1.0, "y": 0.8}],
            "stroke": "#1f77b4",
            "dashed": False,
        },
        {
            "id": "series_1",
            "points": [{"x": 0.0, "y": 0.7}, {"x": 1.0, "y": 0.3}],
            "stroke": "#ff7f0e",
            "dashed": True,
            "dasharray": [6, 4],
        },
    ]


def _panel_plot(rect: Rect, title_font: float) -> Rect:
    padding = max(int(rect.height * 0.08), 10)
    title_height = title_font * 1.4
    plot_x = rect.x + padding
    plot_y = rect.y + padding + title_height
    plot_width = rect.width - padding * 2
    plot_height = rect.height - padding * 2 - title_height
    return Rect(x=plot_x, y=plot_y, width=plot_width, height=plot_height)


def _add_multiline_text(
    group: Any,
    drawing: Any,
    text_id: str,
    lines: list[str],
    x: float,
    y: float,
    font_size: float,
    anchor: str = DEFAULT_TEXT_ANCHOR,
    font_weight: str | None = None,
) -> None:
    if not lines:
        return
    if len(lines) == 1:
        kwargs = {
            "insert": (x, y),
            "id": text_id,
            "font_family": DEFAULT_FONT_FAMILY,
            "font_size": float(font_size),
            "text_anchor": anchor,
            "fill": STROKE,
        }
        if font_weight:
            kwargs["font_weight"] = font_weight
        group.add(drawing.text(lines[0], **kwargs))
        return
    kwargs = {
        "insert": (x, y),
        "id": text_id,
        "font_family": DEFAULT_FONT_FAMILY,
        "font_size": float(font_size),
        "text_anchor": anchor,
        "fill": STROKE,
    }
    if font_weight:
        kwargs["font_weight"] = font_weight
    text = drawing.text("", **kwargs)
    line_height = float(font_size) * 1.25
    for idx, line in enumerate(lines):
        if idx == 0:
            tspan = drawing.tspan(line, x=[x], y=[y], id=f"{text_id}_line{idx}")
        else:
            tspan = drawing.tspan(line, x=[x], dy=[line_height], id=f"{text_id}_line{idx}")
        text.add(tspan)
    group.add(text)


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    width, height = canvas
    rows, cols = _parse_layout(params)
    if rows * cols <= 0:
        raise Png2SvgError(
            code="E2500_GRID_EMPTY",
            message="Grid rows/cols must be positive.",
            hint="Set layout to 2x2 or 1x3.",
        )

    layout = _layout(width, height, rows, cols)
    panel_rects: list[Rect] = layout["panel_rects"]

    title = str(params.get("title") or "Performance Grid").strip()
    panels = params.get("panels")
    if not isinstance(panels, list) or not panels:
        panels = [
            {"id": f"P{idx+1}", "title": f"Panel {idx+1}", "series": _series_defaults()}
            for idx in range(len(panel_rects))
        ]

    annotations = builder.groups["g_annotations"]
    axes_group = builder.groups["g_axes"]
    curves_group = builder.groups["g_curves"]
    text_group = builder.groups["g_text"]

    g_boxes = builder.drawing.g(id="g_boxes")
    annotations.add(g_boxes)
    g_legend = builder.drawing.g(id="g_legend")
    text_group.add(g_legend)
    g_title = builder.drawing.g(id="g_title")
    text_group.add(g_title)

    title_font = max(int(height * 0.04), 18)
    panel_title_font = max(int(height * 0.022), 12)
    label_font = max(int(height * 0.018), 10)

    if title:
        _add_multiline_text(
            g_title,
            builder.drawing,
            "txt_title",
            _split_lines(title),
            layout["margin_x"],
            layout["margin_y"] + title_font,
            title_font,
            anchor="start",
            font_weight="bold",
        )

    series_defaults = _series_defaults()
    if rows > 1 or cols > 1:
        min_x = min(rect.x for rect in panel_rects)
        max_x = max(rect.right for rect in panel_rects)
        min_y = min(rect.y for rect in panel_rects)
        max_y = max(rect.bottom for rect in panel_rects)
        for col in range(cols - 1):
            left_rect = panel_rects[col]
            right_rect = panel_rects[col + 1]
            x_sep = (left_rect.right + right_rect.x) / 2.0
            axes_group.add(
                builder.drawing.line(
                    start=(x_sep, min_y),
                    end=(x_sep, max_y),
                    stroke=STROKE,
                    stroke_width=1,
                    id=f"grid_sep_v_{col}",
                )
            )
        for row in range(rows - 1):
            top_rect = panel_rects[row * cols]
            bottom_rect = panel_rects[(row + 1) * cols]
            y_sep = (top_rect.bottom + bottom_rect.y) / 2.0
            axes_group.add(
                builder.drawing.line(
                    start=(min_x, y_sep),
                    end=(max_x, y_sep),
                    stroke=STROKE,
                    stroke_width=1,
                    id=f"grid_sep_h_{row}",
                )
            )

    for idx, rect in enumerate(panel_rects):
        panel = panels[idx] if idx < len(panels) else {"id": f"P{idx+1}"}
        panel_id = str(panel.get("id") or f"P{idx+1}")
        panel_title = _panel_title(panel, f"Panel {idx+1}")
        plot = _panel_plot(rect, panel_title_font)

        g_boxes.add(
            builder.drawing.rect(
                insert=(rect.x, rect.y),
                size=(rect.width, rect.height),
                rx=8,
                ry=8,
                fill="none",
                stroke=STROKE,
                stroke_width=1,
                id=f"panel_{panel_id}",
            )
        )

        axes_group.add(
            builder.drawing.line(
                start=(plot.x, plot.bottom),
                end=(plot.right, plot.bottom),
                stroke=STROKE,
                stroke_width=1,
                id=f"axis_x_{panel_id}",
            )
        )
        axes_group.add(
            builder.drawing.line(
                start=(plot.x, plot.bottom),
                end=(plot.x, plot.y),
                stroke=STROKE,
                stroke_width=1,
                id=f"axis_y_{panel_id}",
            )
        )

        ticks = [0.0, 0.5, 1.0]
        tick_size = max(int(plot.height * 0.04), 4)
        for t_idx, tick in enumerate(ticks):
            tx = plot.x + plot.width * tick
            ty = plot.bottom
            axes_group.add(
                builder.drawing.line(
                    start=(tx, ty),
                    end=(tx, ty + tick_size),
                    stroke=STROKE,
                    stroke_width=1,
                    id=f"tick_x_{panel_id}_{t_idx}",
                )
            )
            ty = plot.bottom - plot.height * tick
            axes_group.add(
                builder.drawing.line(
                    start=(plot.x - tick_size, ty),
                    end=(plot.x, ty),
                    stroke=STROKE,
                    stroke_width=1,
                    id=f"tick_y_{panel_id}_{t_idx}",
                )
            )

        _add_multiline_text(
            text_group,
            builder.drawing,
            f"txt_panel_{panel_id}_title",
            _split_lines(panel_title),
            rect.x + max(int(rect.width * 0.06), 12),
            rect.y + max(int(rect.height * 0.08), 12) + panel_title_font,
            panel_title_font,
            anchor="start",
            font_weight="bold",
        )

        series = panel.get("series")
        if not isinstance(series, list) or not series:
            series = series_defaults
        for s_idx, series_entry in enumerate(series):
            if not isinstance(series_entry, dict):
                continue
            points = series_entry.get("points")
            if not isinstance(points, list) or len(points) < 2:
                points = series_defaults[s_idx % len(series_defaults)]["points"]
            coords: list[tuple[float, float]] = []
            for point in points:
                try:
                    rx = float(point.get("x", 0.0))
                    ry = float(point.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                rx = max(0.0, min(1.0, rx))
                ry = max(0.0, min(1.0, ry))
                x = plot.x + plot.width * rx
                y = plot.y + plot.height * (1.0 - ry)
                coords.append((x, y))
            if len(coords) < 2:
                continue
            stroke = series_entry.get("stroke") or series_defaults[s_idx % len(series_defaults)]["stroke"]
            dashed = bool(series_entry.get("dashed"))
            dasharray = series_entry.get("dasharray")
            series_id = str(series_entry.get("id") or f"series_{s_idx}")
            poly_kwargs = {
                "points": coords,
                "fill": "none",
                "stroke": stroke,
                "stroke_width": float(series_entry.get("stroke_width", 2)),
                "id": f"curve_{panel_id}_{series_id}",
            }
            if dashed:
                dasharray = dasharray or [6, 4]
                poly_kwargs["stroke_dasharray"] = ",".join(str(v) for v in dasharray)
                poly_kwargs["stroke_linecap"] = "round"
            curves_group.add(builder.drawing.polyline(**poly_kwargs))

    legend_items = params.get("legend")
    if isinstance(legend_items, list) and legend_items:
        legend_x = width - max(int(width * 0.24), 180)
        legend_y = layout["margin_y"] + title_font * 0.2
        legend_gap = max(int(title_font * 0.8), 14)
        for idx, entry in enumerate(legend_items):
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label") or entry.get("id") or f"Series {idx+1}").strip()
            stroke = entry.get("stroke") or series_defaults[idx % len(series_defaults)]["stroke"]
            dashed = bool(entry.get("dashed"))
            y = legend_y + idx * legend_gap
            line = builder.drawing.line(
                start=(legend_x, y),
                end=(legend_x + 30, y),
                stroke=stroke,
                stroke_width=2,
                id=f"legend_line_{idx}",
            )
            if dashed:
                line["stroke-dasharray"] = "6,4"
                line["stroke-linecap"] = "round"
            curves_group.add(line)
            _add_multiline_text(
                g_legend,
                builder.drawing,
                f"txt_legend_{idx}",
                _split_lines(label),
                legend_x + 36,
                y + label_font * 0.3,
                label_font,
                anchor="start",
            )


__all__ = ["render"]
