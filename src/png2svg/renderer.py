from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from common.svg_builder import DEFAULT_FONT_FAMILY, DEFAULT_TEXT_ANCHOR, SvgBuilder
from png2svg.errors import Png2SvgError
from templates.t_3gpp_events_3panel import render as render_3gpp_events_3panel
from templates.t_procedure_flow import render as render_procedure_flow
from templates.t_performance_lineplot import render as render_performance_lineplot
from templates.t_project_architecture_v1 import render as render_project_architecture_v1


def _load_params(params_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(params_path.read_text())
    except json.JSONDecodeError as exc:
        raise Png2SvgError(
            code="E1101_PARAMS_INVALID",
            message=f"Failed to parse params JSON: {exc}",
            hint="Ensure params.json is valid JSON.",
        ) from exc
    if not isinstance(data, dict):
        raise Png2SvgError(
            code="E1102_PARAMS_TYPE",
            message="params.json must contain a JSON object at the top level.",
            hint="Wrap parameters in a JSON object with keys like template/canvas.",
        )
    return data


def _resolve_canvas_size(input_png: Path, params: dict[str, Any]) -> tuple[int, int]:
    canvas = params.get("canvas")
    if canvas is not None and not isinstance(canvas, dict):
        raise Png2SvgError(
            code="E1106_CANVAS_TYPE",
            message="canvas must be an object with width and height.",
            hint="Provide canvas as an object or omit it to use input.png size.",
        )
    if isinstance(canvas, dict):
        width = canvas.get("width")
        height = canvas.get("height")
        if width is None or height is None:
            raise Png2SvgError(
                code="E1103_CANVAS_MISSING",
                message="canvas requires width and height.",
                hint="Set canvas.width and canvas.height or omit canvas to read input.png.",
            )
        try:
            width_int = int(width)
            height_int = int(height)
        except (TypeError, ValueError) as exc:
            raise Png2SvgError(
                code="E1104_CANVAS_INVALID",
                message="canvas width/height must be numeric.",
                hint="Provide numeric canvas dimensions.",
            ) from exc
        if width_int <= 0 or height_int <= 0:
            raise Png2SvgError(
                code="E1107_CANVAS_RANGE",
                message="canvas width/height must be positive.",
                hint="Provide positive canvas dimensions.",
            )
        return width_int, height_int
    with Image.open(input_png) as image:
        width, height = image.size
    return int(width), int(height)


def _render_dummy(builder: SvgBuilder, params: dict[str, Any]) -> None:
    builder.add_axes_placeholder()
    title = params.get("title") or "Untitled"
    builder.add_title(str(title))


def _dispatch_template(
    template: str, builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]
) -> None:
    if template in {"dummy", "t_dummy"}:
        _render_dummy(builder, params)
        return
    if template == "t_3gpp_events_3panel":
        render_3gpp_events_3panel(builder, params, canvas)
        return
    if template == "t_procedure_flow":
        render_procedure_flow(builder, params, canvas)
        return
    if template == "t_performance_lineplot":
        render_performance_lineplot(builder, params, canvas)
        return
    if template in {"t_project_architecture_v1", "project_architecture_v1"}:
        render_project_architecture_v1(builder, params, canvas)
        return
    raise Png2SvgError(
        code="E1105_TEMPLATE_UNKNOWN",
        message=f"Unknown template '{template}'.",
        hint="Use template 'dummy' until specific templates are implemented.",
    )


def _text_items_from_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    items = params.get("texts")
    if isinstance(items, list) and items:
        return items
    extracted = params.get("extracted")
    if isinstance(extracted, dict):
        items = extracted.get("text_items")
        if isinstance(items, list) and items:
            return items
    return []


def _add_text_items(builder: SvgBuilder, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    allowed_anchors = {"start", "middle", "end"}
    sorted_items = sorted(
        items,
        key=lambda item: (
            float(item.get("y", 0.0) or 0.0),
            float(item.get("x", 0.0) or 0.0),
            str(item.get("content") or item.get("text") or ""),
        ),
    )
    for idx, item in enumerate(sorted_items):
        if item.get("render") is False:
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        content = item.get("content")
        if content is None:
            content = item.get("text")
        if content is None:
            content = "Unknown"
        anchor = str(item.get("anchor") or DEFAULT_TEXT_ANCHOR).lower()
        if anchor not in allowed_anchors:
            anchor = DEFAULT_TEXT_ANCHOR
        font_family = item.get("font_family") or DEFAULT_FONT_FAMILY
        text_id = item.get("id") or f"txt_text_{idx:02d}"
        text_kwargs = {
            "insert": (x, y),
            "id": str(text_id),
            "font_family": str(font_family),
            "font_size": float(item.get("font_size", 10)),
            "text_anchor": anchor,
            "fill": str(item.get("fill", "#000000")),
        }
        font_weight = item.get("font_weight")
        if isinstance(font_weight, str) and font_weight:
            text_kwargs["font_weight"] = font_weight
        dominant = item.get("dominant_baseline")
        if isinstance(dominant, str) and dominant:
            text_kwargs["dominant_baseline"] = dominant
        content_str = str(content)
        if "\n" in content_str:
            lines = [line.strip() for line in content_str.splitlines() if line.strip()]
            text = builder.drawing.text("", **text_kwargs)
            line_height = float(text_kwargs.get("font_size", 10)) * 1.3
            for line_idx, line in enumerate(lines):
                if line_idx == 0:
                    tspan = builder.drawing.tspan(
                        line, x=[x], y=[y], id=f"{text_id}_line{line_idx}"
                    )
                else:
                    tspan = builder.drawing.tspan(
                        line, x=[x], dy=[line_height], id=f"{text_id}_line{line_idx}"
                    )
                text.add(tspan)
            builder.groups["g_text"].add(text)
        else:
            builder.groups["g_text"].add(builder.drawing.text(content_str, **text_kwargs))


def _add_geometry(builder: SvgBuilder, params: dict[str, Any]) -> None:
    geometry = params.get("geometry")
    if not isinstance(geometry, dict):
        return
    lines = geometry.get("lines", [])
    rects = geometry.get("rects", [])
    markers = geometry.get("markers", [])

    if isinstance(lines, list):
        sorted_lines = sorted(
            [line for line in lines if isinstance(line, dict)],
            key=lambda line: (
                str(line.get("role", "")),
                float(line.get("y1", 0.0) or 0.0),
                float(line.get("x1", 0.0) or 0.0),
                float(line.get("y2", 0.0) or 0.0),
                float(line.get("x2", 0.0) or 0.0),
            ),
        )
        for idx, line in enumerate(sorted_lines):
            try:
                x1 = float(line.get("x1"))
                y1 = float(line.get("y1"))
                x2 = float(line.get("x2"))
                y2 = float(line.get("y2"))
            except (TypeError, ValueError):
                continue
            stroke = line.get("stroke") or "#555555"
            stroke_width = float(line.get("stroke_width", 1))
            line_kwargs = {
                "start": (x1, y1),
                "end": (x2, y2),
                "stroke": stroke,
                "stroke_width": stroke_width,
                "id": line.get("id") or f"geom_line_{idx:02d}",
            }
            if line.get("dashed"):
                dasharray = line.get("dasharray") or [6, 4]
                if isinstance(dasharray, list):
                    dasharray = ",".join(str(value) for value in dasharray)
                line_kwargs["stroke_dasharray"] = dasharray
                line_kwargs["class_"] = "dashed"
            builder.groups["g_annotations"].add(builder.drawing.line(**line_kwargs))

    if isinstance(rects, list):
        sorted_rects = sorted(
            [rect for rect in rects if isinstance(rect, dict)],
            key=lambda rect: (
                str(rect.get("role", "")),
                float(rect.get("y", 0.0) or 0.0),
                float(rect.get("x", 0.0) or 0.0),
            ),
        )
        for idx, rect in enumerate(sorted_rects):
            try:
                x = float(rect.get("x"))
                y = float(rect.get("y"))
                width = float(rect.get("width"))
                height = float(rect.get("height"))
            except (TypeError, ValueError):
                continue
            rect_kwargs = {
                "insert": (x, y),
                "size": (width, height),
                "fill": rect.get("fill") or "none",
                "stroke": rect.get("stroke") or "#555555",
                "stroke_width": float(rect.get("stroke_width", 1)),
                "id": rect.get("id") or f"geom_rect_{idx:02d}",
            }
            builder.groups["g_annotations"].add(builder.drawing.rect(**rect_kwargs))

    if isinstance(markers, list):
        sorted_markers = sorted(
            [marker for marker in markers if isinstance(marker, dict)],
            key=lambda marker: (
                str(marker.get("role", "")),
                float(marker.get("y", 0.0) or 0.0),
                float(marker.get("x", 0.0) or 0.0),
            ),
        )
        for idx, marker in enumerate(sorted_markers):
            try:
                x = float(marker.get("x"))
                y = float(marker.get("y"))
            except (TypeError, ValueError):
                continue
            radius = float(marker.get("radius", 3))
            builder.groups["g_markers"].add(
                builder.drawing.circle(
                    center=(x, y),
                    r=radius,
                    fill=marker.get("fill") or "#555555",
                    id=marker.get("id") or f"geom_marker_{idx:02d}",
                )
            )


def render_svg(input_png: Path, params_path: Path, output_svg: Path) -> None:
    params = _load_params(params_path)
    template = params.get("template")
    if not template:
        raise Png2SvgError(
            code="E1100_TEMPLATE_MISSING",
            message="params.json missing required 'template' field.",
            hint="Set template to 'dummy' for now.",
        )
    width, height = _resolve_canvas_size(input_png, params)
    builder = SvgBuilder.create(width=width, height=height)
    _dispatch_template(str(template), builder, params, (width, height))
    _add_geometry(builder, params)
    _add_text_items(builder, _text_items_from_params(params))
    builder.save(output_svg)
