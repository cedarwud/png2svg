from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from png2svg.classifier import classify_png
from png2svg.errors import Png2SvgError
from png2svg.normalize import normalize_params
from png2svg.ocr import has_pytesseract, has_tesseract, ocr_image

from png2svg.extractor_config import _effective_config, _load_adaptive_config
from png2svg.extractor_constants import TEMPLATE_ALIASES
from png2svg.extractor_debug import _write_debug_artifacts
from png2svg.extractor_geometry import _detect_panels
from png2svg.extractor_preprocess import (
    _ink_mask,
    _load_image,
    _pad_roi,
    _prepare_ocr_image,
    _preprocess_image,
)
from png2svg.extractor_templates import (
    _default_panels,
    _default_plot,
    _extract_flow,
    _extract_lineplot,
    _extract_project_architecture_v1,
    _project_architecture_rois,
    _finalize_3gpp_v1_metadata,
    extract_3gpp_events_3panel_v1,
)
from png2svg.extractor_text import (
    _count_renderable_texts,
    _detect_text_boxes,
    _filter_text_items,
    _text_items_from_ocr,
)
from png2svg.extractor_types import ExtractIssue


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
            hint=(
                "Use one of: t_3gpp_events_3panel, t_procedure_flow, "
                "t_performance_lineplot, t_project_architecture_v1, or auto."
            ),
        )

    skip_ocr = False
    rgba, width, height = _load_image(input_png)
    adaptive_config = _load_adaptive_config()
    effective_config = _effective_config(adaptive_config, width, height)
    preprocessed = _preprocess_image(rgba, effective_config)
    mask = _ink_mask(rgba, effective_config)
    text_boxes = [] if skip_ocr else _detect_text_boxes(rgba, effective_config)
    warnings: list[ExtractIssue] = []
    errors: list[ExtractIssue] = []
    ocr_backend = os.environ.get("PNG2SVG_OCR_BACKEND", "auto")
    if skip_ocr:
        ocr_backend = "none"

    ocr_rois: list[dict[str, int]] | None = None
    if template_id == "t_3gpp_events_3panel":
        panels = _detect_panels(mask, width, height) or _default_panels(width, height)
        ocr_rois = [
            {"id": "title", "x": 0, "y": 0, "width": width, "height": int(height * 0.18)},
        ]
        for panel in panels:
            panel_id = str(panel.get("id") or "")
            ocr_rois.append(
                {
                    "id": f"panel_top_{panel_id}",
                    "x": int(panel["x"]),
                    "y": int(panel["y"]),
                    "width": int(panel["width"]),
                    "height": int(panel["height"] * 0.25),
                }
            )
            ocr_rois.append(
                {
                    "id": f"panel_mid_{panel_id}",
                    "x": int(panel["x"]),
                    "y": int(panel["y"] + panel["height"] * 0.3),
                    "width": int(panel["width"]),
                    "height": int(panel["height"] * 0.4),
                }
            )
            ocr_rois.append(
                {
                    "id": f"panel_bottom_{panel_id}",
                    "x": int(panel["x"]),
                    "y": int(panel["y"] + panel["height"] * 0.75),
                    "width": int(panel["width"]),
                    "height": int(panel["height"] * 0.25),
                }
            )
    elif template_id == "t_performance_lineplot":
        plot = _default_plot(width, height)
        ocr_rois = [
            {"id": "title", "x": 0, "y": 0, "width": width, "height": int(height * 0.18)},
            {
                "id": "axis_x",
                "x": int(plot["x"]),
                "y": int(plot["y"] + plot["height"]),
                "width": int(plot["width"]),
                "height": int(height * 0.2),
            },
            {
                "id": "axis_y",
                "x": 0,
                "y": int(plot["y"]),
                "width": int(plot["x"]),
                "height": int(plot["height"]),
            },
        ]
    elif template_id == "t_project_architecture_v1":
        ocr_rois = _project_architecture_rois(width, height)

    if ocr_rois:
        pad_px = int(effective_config.get("ocr", {}).get("roi_pad_px", 0))
        ocr_rois = [_pad_roi(roi, pad_px, width, height) for roi in ocr_rois]

    backend_value = ocr_backend.lower()
    if skip_ocr:
        ocr_available = False
    elif backend_value == "pytesseract":
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

    if not skip_ocr:
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
    elif template_id == "t_project_architecture_v1":
        params, overlay = _extract_project_architecture_v1(
            width, height, text_items, warnings, adaptive=effective_config
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
    if ocr_results:
        renderable_count = _count_renderable_texts(text_items)
        extracted["texts_detected"] = renderable_count if renderable_count > 0 else 1
    else:
        extracted["texts_detected"] = 0
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


__all__ = ["extract_skeleton", "ExtractIssue"]
