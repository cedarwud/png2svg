from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from common.svg_builder import SvgBuilder
from png2svg.errors import Png2SvgError
from templates.t_3gpp_events_3panel import render as render_3gpp_events_3panel
from templates.t_procedure_flow import render as render_procedure_flow
from templates.t_performance_lineplot import render as render_performance_lineplot


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
    raise Png2SvgError(
        code="E1105_TEMPLATE_UNKNOWN",
        message=f"Unknown template '{template}'.",
        hint="Use template 'dummy' until specific templates are implemented.",
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
    builder.save(output_svg)
