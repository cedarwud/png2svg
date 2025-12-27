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
    _clean_text,
    _enforce_curve_label_colors_3gpp,
    _merge_stacked_text_items,
    _normalize_curve_labels_3gpp,
    _normalize_panel_mid_text,
    _panel_label_from_text,
    _text_bbox_center,
    _text_bbox_width,
)
from png2svg.extractor_types import ExtractIssue
from png2svg.text_normalize import (
    canonical_text_for_template,
    lexicon_for_template,
    normalize_lines,
    normalize_text_items,
    text_sanity,
)


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
    text_items: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
    text_mode: str = "hybrid",
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout = _project_architecture_layout(width, height)
    defaults = _project_architecture_defaults()
    title = defaults["title"]
    subtitle = defaults["subtitle"]
    panels = defaults["panels"]
    work_packages = defaults["work_packages"]

    text_mode_value = str(text_mode or "hybrid").lower()
    lexicon = lexicon_for_template("t_project_architecture_v1")
    ocr_cfg = adaptive.get("ocr") if adaptive else None
    min_conf = float(ocr_cfg.get("min_conf", 0.6)) if isinstance(ocr_cfg, dict) else 0.6
    corrections = 0

    by_roi: dict[str, list[dict[str, Any]]] = {}
    for item in text_items:
        roi_id = item.get("roi_id")
        if not isinstance(roi_id, str) or not roi_id:
            continue
        by_roi.setdefault(roi_id, []).append(item)

    def _roi_lines(roi_id: str) -> tuple[list[str], float]:
        nonlocal corrections
        items = by_roi.get(roi_id, [])
        if not items:
            return [], 0.0
        items.sort(
            key=lambda entry: (
                float(entry.get("bbox", {}).get("y", 0.0)),
                float(entry.get("bbox", {}).get("x", 0.0)),
            )
        )
        lines: list[str] = []
        conf_values: list[float] = []
        for entry in items:
            text = _clean_text(str(entry.get("content") or entry.get("text") or ""))
            if text:
                lines.append(text)
            try:
                conf_values.append(float(entry.get("conf", 0.0)))
            except (TypeError, ValueError):
                conf_values.append(0.0)
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        if lexicon and lines:
            normalized, corrected = normalize_lines(
                lines, lexicon, avg_conf, min_conf=min_conf
            )
            if normalized:
                lines = normalized
            corrections += corrected
        return lines, avg_conf

    def _join_lines(lines: list[str]) -> str:
        return _clean_text(" ".join(lines))

    def _parse_bullets(lines: list[str]) -> list[str]:
        bullets: list[str] = []
        for line in lines:
            trimmed = line.lstrip("•-–· ").strip()
            if trimmed:
                bullets.append(trimmed)
        return bullets

    def _parse_goal_output(lines: list[str]) -> tuple[str | None, str | None]:
        goal = None
        output = None
        remaining: list[str] = []
        for line in lines:
            lower = line.lower()
            if "goal" in lower:
                goal = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            elif "output" in lower:
                output = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            else:
                remaining.append(line.strip())
        if goal is None and remaining:
            goal = remaining.pop(0)
        if output is None and remaining:
            output = remaining.pop(0)
        return goal, output

    fields_from_ocr: list[str] = []
    fallback_fields: list[str] = []
    if text_mode_value != "template_text":
        title_lines, title_conf = _roi_lines("title")
        if title_lines and (text_mode_value == "ocr_text" or title_conf >= min_conf):
            title = _join_lines(title_lines)
            if text_mode_value == "ocr_text" or text_sanity(title):
                fields_from_ocr.append("title")
            else:
                fallback_fields.append("title")
        subtitle_lines, subtitle_conf = _roi_lines("subtitle")
        if subtitle_lines and (text_mode_value == "ocr_text" or subtitle_conf >= min_conf):
            subtitle = _join_lines(subtitle_lines)
            if text_mode_value == "ocr_text" or text_sanity(subtitle):
                fields_from_ocr.append("subtitle")
            else:
                fallback_fields.append("subtitle")

    if text_mode_value != "template_text":
        for panel in panels:
            panel_id = str(panel.get("id") or "")
            title_lines, title_conf = _roi_lines(f"panel_{panel_id}_title")
            if title_lines and (text_mode_value == "ocr_text" or title_conf >= min_conf):
                candidate = _join_lines(title_lines)
                if text_mode_value == "ocr_text" or text_sanity(candidate):
                    panel["title"] = candidate
                    fields_from_ocr.append(f"panel_{panel_id}_title")
                else:
                    fallback_fields.append(f"panel_{panel_id}_title")
            bullet_lines, bullet_conf = _roi_lines(f"panel_{panel_id}_bullets")
            if bullet_lines and (text_mode_value == "ocr_text" or bullet_conf >= min_conf):
                bullet_items = _parse_bullets(bullet_lines)
                if bullet_items:
                    panel["bullets"] = bullet_items
                    fields_from_ocr.append(f"panel_{panel_id}_bullets")
                else:
                    fallback_fields.append(f"panel_{panel_id}_bullets")

    if text_mode_value != "template_text":
        for wp in work_packages:
            wp_id = str(wp.get("id") or "")
            title_lines, title_conf = _roi_lines(f"wp_{wp_id}_title")
            if title_lines and (text_mode_value == "ocr_text" or title_conf >= min_conf):
                candidate = _join_lines(title_lines)
                if text_mode_value == "ocr_text" or text_sanity(candidate):
                    wp["title"] = candidate
                    fields_from_ocr.append(f"wp_{wp_id}_title")
                else:
                    fallback_fields.append(f"wp_{wp_id}_title")
            body_lines, body_conf = _roi_lines(f"wp_{wp_id}_body")
            if body_lines and (text_mode_value == "ocr_text" or body_conf >= min_conf):
                goal, output = _parse_goal_output(body_lines)
                if goal:
                    wp["goal"] = goal
                    fields_from_ocr.append(f"wp_{wp_id}_goal")
                if output:
                    wp["output"] = output
                    fields_from_ocr.append(f"wp_{wp_id}_output")
                if not (goal or output):
                    fallback_fields.append(f"wp_{wp_id}_body")

    if text_mode_value != "template_text" and not fields_from_ocr:
        warnings.append(
            ExtractIssue(
                code="W4500_PROJECT_ARCH_FALLBACK",
                message="Project architecture OCR yielded no usable text; using defaults.",
                hint="Ensure OCR is available or edit params.json manually.",
            )
        )
    if fallback_fields:
        warnings.append(
            ExtractIssue(
                code="W4501_PROJECT_ARCH_TEXT_FALLBACK",
                message="Some OCR fields were low confidence; template defaults were used.",
                hint="Increase OCR confidence or edit params.json text fields manually.",
                context={"fields": sorted(set(fallback_fields))},
            )
        )

    params: dict[str, Any] = {
        "template": "t_project_architecture_v1",
        "canvas": {"width": width, "height": height},
        "title": title,
        "subtitle": subtitle,
        "panels": panels,
        "work_packages": work_packages,
        "extracted": {
            "project_architecture": {
                "ocr_used": bool(fields_from_ocr),
                "fields_from_ocr": fields_from_ocr,
            }
        },
    }
    extracted = params.get("extracted")
    if isinstance(extracted, dict):
        extracted.setdefault("ocr_stats", {})
        if isinstance(extracted["ocr_stats"], dict):
            extracted["ocr_stats"].update(
                {
                    "corrected_tokens": corrections,
                    "text_mode": text_mode_value,
                }
            )
    overlay = {
        "panels": list(layout["panel_rects"].values()),
        "text_boxes": [
            {"bbox": roi} for roi in _project_architecture_rois(width, height)
        ],
    }
    return params, overlay


def _project_architecture_defaults() -> dict[str, Any]:
    return {
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


def _project_architecture_layout(width: int, height: int) -> dict[str, Any]:
    margin_x = max(int(width * 0.04), 40)
    margin_y = max(int(height * 0.04), 32)
    header_height = max(int(height * 0.16), 140)
    top_height = max(int(height * 0.28), 260)
    gap_y = max(int(height * 0.04), 32)
    top_y = margin_y + header_height
    bottom_y = top_y + top_height + gap_y
    bottom_height = height - margin_y - bottom_y
    min_bottom = int(height * 0.25)
    if bottom_height < min_bottom:
        bottom_height = max(height - margin_y - bottom_y, min_bottom)
    gap_x = max(int(width * 0.02), 24)
    panel_width = (width - 2 * margin_x - 2 * gap_x) / 3.0
    panel_rects: dict[str, dict[str, float]] = {}
    x = float(margin_x)
    for panel_id in ("A", "B", "C"):
        panel_rects[panel_id] = {
            "x": x,
            "y": float(top_y),
            "width": float(panel_width),
            "height": float(top_height),
        }
        x += panel_width + gap_x
    container = {
        "x": float(margin_x),
        "y": float(bottom_y),
        "width": float(width - 2 * margin_x),
        "height": float(bottom_height),
    }
    padding = max(int(height * 0.02), 18)
    label_height = max(int(height * 0.035), 28)
    wp_y = container["y"] + padding + label_height
    wp_height = max(container["height"] - padding * 2 - label_height, label_height)
    wp_gap = max(int(width * 0.015), 18)
    wp_width = (container["width"] - padding * 2 - wp_gap * 3) / 4.0
    wp_rects: dict[str, dict[str, float]] = {}
    x = container["x"] + padding
    for wp_id in ("WP1", "WP2", "WP3", "WP4"):
        wp_rects[wp_id] = {
            "x": x,
            "y": float(wp_y),
            "width": float(wp_width),
            "height": float(wp_height),
        }
        x += wp_width + wp_gap
    return {
        "margin_x": float(margin_x),
        "margin_y": float(margin_y),
        "header_height": float(header_height),
        "panel_rects": panel_rects,
        "container": container,
        "wp_rects": wp_rects,
        "container_padding": float(padding),
        "container_label_height": float(label_height),
    }


def _project_architecture_rois(width: int, height: int) -> list[dict[str, int]]:
    layout = _project_architecture_layout(width, height)
    rois: list[dict[str, int]] = []

    def _roi(roi_id: str, x: float, y: float, w: float, h: float) -> None:
        ix = max(int(round(x)), 0)
        iy = max(int(round(y)), 0)
        iw = max(int(round(w)), 1)
        ih = max(int(round(h)), 1)
        if ix + iw > width:
            iw = max(width - ix, 1)
        if iy + ih > height:
            ih = max(height - iy, 1)
        rois.append({"id": roi_id, "x": ix, "y": iy, "width": iw, "height": ih})

    header_height = layout["header_height"]
    margin_x = layout["margin_x"]
    margin_y = layout["margin_y"]
    _roi("title", margin_x, margin_y, width - 2 * margin_x, header_height * 0.5)
    _roi(
        "subtitle",
        margin_x,
        margin_y + header_height * 0.45,
        width - 2 * margin_x,
        header_height * 0.5,
    )

    pad = max(int(height * 0.01), 10)
    for panel_id, rect in layout["panel_rects"].items():
        _roi(
            f"panel_{panel_id}_title",
            rect["x"] + pad,
            rect["y"] + pad,
            rect["width"] - pad * 2,
            rect["height"] * 0.25,
        )
        _roi(
            f"panel_{panel_id}_bullets",
            rect["x"] + pad,
            rect["y"] + rect["height"] * 0.25,
            rect["width"] - pad * 2,
            rect["height"] * 0.65,
        )

    container = layout["container"]
    padding = layout["container_padding"]
    label_height = layout["container_label_height"]
    _roi(
        "wp_container_label",
        container["x"] + padding,
        container["y"] + padding * 0.3,
        container["width"] - padding * 2,
        label_height * 1.2,
    )

    for wp_id, rect in layout["wp_rects"].items():
        title_height = label_height * 1.2
        _roi(
            f"wp_{wp_id}_title",
            rect["x"] + padding,
            rect["y"] + padding * 0.4,
            rect["width"] - padding * 2,
            title_height,
        )
        body_y = rect["y"] + padding * 0.4 + title_height
        _roi(
            f"wp_{wp_id}_body",
            rect["x"] + padding,
            body_y,
            rect["width"] - padding * 2,
            rect["height"] - (body_y - rect["y"]) - padding,
        )

    return rois


def _rl_agent_loop_layout(width: int, height: int) -> dict[str, dict[str, float]]:
    margin_x = max(int(width * 0.08), 40)
    margin_y = max(int(height * 0.06), 30)
    header_height = max(int(height * 0.14), 90)
    box_width = width * 0.28
    box_height = height * 0.2
    gap_y = max(int(height * 0.08), 50)
    main_y = margin_y + header_height
    agent = {
        "x": float(margin_x),
        "y": float(main_y),
        "width": float(box_width),
        "height": float(box_height),
    }
    env = {
        "x": float(width - margin_x - box_width),
        "y": float(main_y),
        "width": float(box_width),
        "height": float(box_height),
    }
    constraint = {
        "x": float((width - box_width * 0.7) / 2.0),
        "y": float(max(margin_y + header_height * 0.4, main_y - box_height * 0.7)),
        "width": float(box_width * 0.7),
        "height": float(box_height * 0.45),
    }
    buffer = {
        "x": float((width - box_width * 0.75) / 2.0),
        "y": float(main_y + box_height + gap_y),
        "width": float(box_width * 0.75),
        "height": float(box_height * 0.45),
    }
    return {
        "agent": agent,
        "environment": env,
        "constraint": constraint,
        "buffer": buffer,
        "margin_x": float(margin_x),
        "margin_y": float(margin_y),
        "header_height": float(header_height),
    }


def _rl_agent_loop_rois(width: int, height: int) -> list[dict[str, int]]:
    layout = _rl_agent_loop_layout(width, height)
    rois: list[dict[str, int]] = []

    def _roi(roi_id: str, x: float, y: float, w: float, h: float) -> None:
        ix = max(int(round(x)), 0)
        iy = max(int(round(y)), 0)
        iw = max(int(round(w)), 1)
        ih = max(int(round(h)), 1)
        if ix + iw > width:
            iw = max(width - ix, 1)
        if iy + ih > height:
            ih = max(height - iy, 1)
        rois.append({"id": roi_id, "x": ix, "y": iy, "width": iw, "height": ih})

    header_height = layout["header_height"]
    margin_x = layout["margin_x"]
    margin_y = layout["margin_y"]
    _roi("title", margin_x, margin_y, width - 2 * margin_x, header_height * 0.7)

    for key in ("agent", "environment", "constraint", "buffer"):
        rect = layout[key]
        pad = max(int(rect["height"] * 0.15), 8)
        _roi(
            f"box_{key}",
            rect["x"] + pad,
            rect["y"] + pad,
            rect["width"] - pad * 2,
            rect["height"] - pad * 2,
        )

    agent = layout["agent"]
    env = layout["environment"]
    action_y = agent["y"] + agent["height"] * 0.7
    feedback_y = agent["y"] + agent["height"] * 0.3
    signal_width = env["x"] - (agent["x"] + agent["width"])
    _roi(
        "signal_action",
        agent["x"] + agent["width"],
        action_y - agent["height"] * 0.2,
        signal_width,
        agent["height"] * 0.4,
    )
    _roi(
        "signal_feedback",
        agent["x"] + agent["width"],
        feedback_y - agent["height"] * 0.2,
        signal_width,
        agent["height"] * 0.4,
    )
    return rois


def _performance_grid_layout(
    width: int,
    height: int,
    rows: int,
    cols: int,
) -> dict[str, Any]:
    margin_x = max(int(width * 0.06), 36)
    margin_y = max(int(height * 0.06), 28)
    header_height = max(int(height * 0.12), 70)
    gap_x = max(int(width * 0.04), 18)
    gap_y = max(int(height * 0.05), 20)
    panel_width = (width - 2 * margin_x - gap_x * (cols - 1)) / cols
    panel_height = (height - margin_y - header_height - gap_y * (rows - 1)) / rows
    panels: list[dict[str, float]] = []
    start_y = margin_y + header_height
    for row in range(rows):
        y = start_y + row * (panel_height + gap_y)
        for col in range(cols):
            x = margin_x + col * (panel_width + gap_x)
            panels.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "width": float(panel_width),
                    "height": float(panel_height),
                }
            )
    return {
        "rows": rows,
        "cols": cols,
        "margin_x": float(margin_x),
        "margin_y": float(margin_y),
        "header_height": float(header_height),
        "panels": panels,
    }


def _detect_performance_grid_layout(mask: np.ndarray, width: int, height: int) -> tuple[str, dict[str, Any]]:
    min_v = max(int(height * 0.6), 1)
    min_h = max(int(width * 0.6), 1)
    v_lines = _long_line_positions(mask, axis=0, min_len=min_v)
    h_lines = _long_line_positions(mask, axis=1, min_len=min_h)
    v_lines = [x for x in v_lines if width * 0.15 < x < width * 0.85]
    h_lines = [y for y in h_lines if height * 0.2 < y < height * 0.85]

    layout_id = None
    if len(v_lines) >= 2 and not h_lines:
        layout_id = "1x3"
        layout = _performance_grid_layout(width, height, rows=1, cols=3)
    elif len(v_lines) >= 1 and len(h_lines) >= 1:
        layout_id = "2x2"
        layout = _performance_grid_layout(width, height, rows=2, cols=2)
    else:
        layout_id = "1x3" if width >= height * 1.5 else "2x2"
        rows, cols = (1, 3) if layout_id == "1x3" else (2, 2)
        layout = _performance_grid_layout(width, height, rows=rows, cols=cols)
    layout["separators"] = {"vertical": v_lines, "horizontal": h_lines}
    return layout_id, layout


def _performance_grid_rois(width: int, height: int, layout: dict[str, Any]) -> list[dict[str, int]]:
    rois: list[dict[str, int]] = []

    def _roi(roi_id: str, x: float, y: float, w: float, h: float) -> None:
        ix = max(int(round(x)), 0)
        iy = max(int(round(y)), 0)
        iw = max(int(round(w)), 1)
        ih = max(int(round(h)), 1)
        if ix + iw > width:
            iw = max(width - ix, 1)
        if iy + ih > height:
            ih = max(height - iy, 1)
        rois.append({"id": roi_id, "x": ix, "y": iy, "width": iw, "height": ih})

    margin_x = layout["margin_x"]
    margin_y = layout["margin_y"]
    header_height = layout["header_height"]
    _roi("title", margin_x, margin_y, width - 2 * margin_x, header_height * 0.7)

    for idx, panel in enumerate(layout["panels"]):
        pid = f"P{idx + 1}"
        pad = max(int(panel["height"] * 0.08), 10)
        _roi(
            f"panel_{pid}_title",
            panel["x"] + pad,
            panel["y"] + pad,
            panel["width"] - pad * 2,
            panel["height"] * 0.25,
        )
    return rois


def _extract_rl_agent_loop_v1(
    width: int,
    height: int,
    text_items: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = {
        "title": "RL Agent Loop",
        "agent": "Agent",
        "environment": "Environment",
        "constraint": "Constraints",
        "buffer": "Replay Buffer",
        "action": "Action",
        "feedback": "State / Reward",
    }
    by_roi: dict[str, list[dict[str, Any]]] = {}
    for item in text_items:
        roi_id = item.get("roi_id")
        if isinstance(roi_id, str) and roi_id:
            by_roi.setdefault(roi_id, []).append(item)

    def _roi_lines(roi_id: str) -> list[str]:
        items = by_roi.get(roi_id, [])
        if not items:
            return []
        items.sort(
            key=lambda entry: (
                float(entry.get("bbox", {}).get("y", 0.0)),
                float(entry.get("bbox", {}).get("x", 0.0)),
            )
        )
        lines: list[str] = []
        for entry in items:
            text = _clean_text(str(entry.get("content") or entry.get("text") or ""))
            if text:
                lines.append(text)
        return lines

    def _pick_text(roi_id: str, fallback: str) -> str:
        lines = _roi_lines(roi_id)
        if not lines:
            return fallback
        return _clean_text(" ".join(lines))

    fields_from_ocr: list[str] = []
    title = _pick_text("title", defaults["title"])
    if title != defaults["title"]:
        fields_from_ocr.append("title")
    agent = _pick_text("box_agent", defaults["agent"])
    if agent != defaults["agent"]:
        fields_from_ocr.append("agent")
    env = _pick_text("box_environment", defaults["environment"])
    if env != defaults["environment"]:
        fields_from_ocr.append("environment")
    constraint = _pick_text("box_constraint", defaults["constraint"])
    if constraint != defaults["constraint"]:
        fields_from_ocr.append("constraint")
    buffer = _pick_text("box_buffer", defaults["buffer"])
    if buffer != defaults["buffer"]:
        fields_from_ocr.append("buffer")
    action = _pick_text("signal_action", defaults["action"])
    if action != defaults["action"]:
        fields_from_ocr.append("action")
    feedback = _pick_text("signal_feedback", defaults["feedback"])
    if feedback != defaults["feedback"]:
        fields_from_ocr.append("feedback")

    if not fields_from_ocr:
        warnings.append(
            ExtractIssue(
                code="W4501_RL_FALLBACK",
                message="RL agent loop OCR yielded no usable text; using defaults.",
                hint="Ensure OCR is available or edit params.json manually.",
            )
        )

    params: dict[str, Any] = {
        "template": "t_rl_agent_loop_v1",
        "canvas": {"width": width, "height": height},
        "title": title,
        "boxes": [
            {"id": "agent", "role": "agent", "label": agent},
            {"id": "environment", "role": "environment", "label": env},
            {"id": "constraint", "role": "constraint", "label": constraint, "enabled": True},
            {"id": "buffer", "role": "buffer", "label": buffer, "enabled": True},
        ],
        "signals": {"action": action, "feedback": feedback},
        "extracted": {
            "texts_detected": len(fields_from_ocr) if fields_from_ocr else 0,
            "rl_agent_loop": {"ocr_used": bool(fields_from_ocr), "fields_from_ocr": fields_from_ocr},
        },
    }
    overlay = {
        "text_boxes": [{"bbox": roi} for roi in _rl_agent_loop_rois(width, height)],
    }
    return params, overlay


def _grid_panel_plot(panel: dict[str, Any], title_font: float) -> dict[str, float]:
    padding = max(int(panel["height"] * 0.08), 10)
    title_height = title_font * 1.4
    return {
        "x": float(panel["x"] + padding),
        "y": float(panel["y"] + padding + title_height),
        "width": float(panel["width"] - padding * 2),
        "height": float(panel["height"] - padding * 2 - title_height),
    }


def _extract_performance_grid_v1(
    width: int,
    height: int,
    mask: np.ndarray,
    rgba: np.ndarray,
    text_items: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout_id, layout = _detect_performance_grid_layout(mask, width, height)
    panels_layout = layout["panels"]
    panel_titles = [f"Panel {idx+1}" for idx in range(len(panels_layout))]
    by_roi: dict[str, list[dict[str, Any]]] = {}
    for item in text_items:
        roi_id = item.get("roi_id")
        if isinstance(roi_id, str) and roi_id:
            by_roi.setdefault(roi_id, []).append(item)

    def _roi_text(roi_id: str) -> str | None:
        items = by_roi.get(roi_id, [])
        if not items:
            return None
        items.sort(
            key=lambda entry: (
                float(entry.get("bbox", {}).get("y", 0.0)),
                float(entry.get("bbox", {}).get("x", 0.0)),
            )
        )
        lines: list[str] = []
        for entry in items:
            text = _clean_text(str(entry.get("content") or entry.get("text") or ""))
            if text:
                lines.append(text)
        if not lines:
            return None
        return _clean_text(" ".join(lines))

    title = _roi_text("title") or "Performance Grid"
    panels: list[dict[str, Any]] = []
    fields_from_ocr: list[str] = []
    if title != "Performance Grid":
        fields_from_ocr.append("title")

    curve_cfg = adaptive.get("curves", {}) if adaptive else {}
    series_colors = DEFAULT_SERIES_COLORS
    curve_points: list[dict[str, Any]] = []
    for idx, panel in enumerate(panels_layout):
        panel_id = f"P{idx + 1}"
        roi_title = _roi_text(f"panel_{panel_id}_title")
        panel_title = roi_title or panel_titles[idx]
        if roi_title:
            fields_from_ocr.append(f"panel_{panel_id}_title")
        title_font = max(int(height * 0.022), 12)
        plot = _grid_panel_plot(panel, title_font)

        series: list[dict[str, Any]] = []
        x0 = int(panel["x"])
        y0 = int(panel["y"])
        x1 = int(panel["x"] + panel["width"])
        y1 = int(panel["y"] + panel["height"])
        sub = rgba[y0:y1, x0:x1]
        for s_idx, color in enumerate(series_colors):
            hue = [220.0, 30.0, 120.0, 0.0][s_idx % 4]
            mask_color = _curve_color_mask(sub, hue, curve_cfg)
            points = _curve_centerline_points(mask_color, curve_cfg)
            if not points:
                continue
            full_points = [(x + x0, y + y0) for x, y in points]
            curve_points.append(
                {
                    "panel_id": panel_id,
                    "curve_id": f"series_{s_idx}",
                    "points": [{"x": x, "y": y} for x, y in full_points],
                }
            )
            series.append(
                {
                    "id": f"series_{s_idx}",
                    "points": _points_to_ratio(full_points, plot),
                    "stroke": color,
                    "dashed": s_idx % 2 == 1,
                    "dasharray": DEFAULT_DASHARRAY if s_idx % 2 == 1 else None,
                }
            )
        if not series:
            warnings.append(
                ExtractIssue(
                    code="W4503_GRID_CURVES_FALLBACK",
                    message=f"No curve points detected for {panel_id}; using defaults.",
                    hint="Increase line contrast or adjust curve detection settings.",
                )
            )
            series = [
                {
                    "id": "series_0",
                    "points": [{"x": 0.0, "y": 0.2}, {"x": 1.0, "y": 0.8}],
                    "stroke": series_colors[0],
                    "dashed": False,
                }
            ]
        panels.append({"id": panel_id, "title": panel_title, "series": series})

    if not fields_from_ocr:
        warnings.append(
            ExtractIssue(
                code="W4502_GRID_FALLBACK",
                message="Performance grid OCR yielded no usable text; using defaults.",
                hint="Ensure OCR is available or edit params.json manually.",
            )
        )

    params = {
        "template": "t_performance_grid_v1",
        "canvas": {"width": width, "height": height},
        "title": title,
        "layout": layout_id,
        "panels": panels,
        "legend": [
            {"id": "series_0", "label": "Baseline", "stroke": "#1f77b4", "dashed": False},
            {"id": "series_1", "label": "Proposed", "stroke": "#ff7f0e", "dashed": True},
        ],
        "geometry": {"lines": [], "rects": [], "markers": []},
        "extracted": {
            "texts_detected": len(fields_from_ocr) if fields_from_ocr else 0,
            "panel_bounds": panels_layout,
            "grid_layout": layout_id,
            "curve_points": curve_points,
            "text_blocks": text_boxes,
            "grid_meta": {"ocr_used": bool(fields_from_ocr), "fields_from_ocr": fields_from_ocr},
        },
    }
    overlay = {
        "panels": panels_layout,
        "text_boxes": text_boxes,
    }
    return params, overlay


def _extract_3gpp(
    width: int,
    height: int,
    mask: np.ndarray,
    rgba: np.ndarray,
    text_items: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    warnings: list[ExtractIssue],
    adaptive: dict[str, Any] | None = None,
    text_mode: str = "hybrid",
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
    text_mode_value = str(text_mode or "hybrid").lower()
    ocr_cfg = adaptive.get("ocr") if adaptive else None
    min_conf = float(ocr_cfg.get("min_conf", 0.6)) if isinstance(ocr_cfg, dict) else 0.6
    corrections = 0
    if text_mode_value != "template_text":
        corrections = normalize_text_items(text_items, lexicon_for_template("t_3gpp_events_3panel"), min_conf=min_conf)

    title, title_style, text_items = _assign_roles_3gpp(text_items, panels, width, height)
    canonical = canonical_text_for_template("t_3gpp_events_3panel")
    canonical_title = canonical.get("title")
    if not title and canonical_title:
        title = canonical_title
        if text_mode_value != "template_text":
            warnings.append(
                ExtractIssue(
                    code="W4006_TITLE_FALLBACK",
                    message="3GPP title missing; using canonical template title.",
                    hint="Provide a clearer title region or edit params.json manually.",
                )
            )
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

    annotations: list[dict[str, Any]] = []
    seen_annotations: set[tuple[str, str]] = set()

    def _panel_for_item(item: dict[str, Any]) -> str | None:
        cx, cy = _text_bbox_center(item)
        for panel in panels:
            if (
                panel["x"] <= cx <= panel["x"] + panel["width"]
                and panel["y"] <= cy <= panel["y"] + panel["height"]
            ):
                return str(panel.get("id") or "")
        return None

    def _maybe_add_annotation(item: dict[str, Any], label: str, slug: str) -> None:
        panel_id = _panel_for_item(item) or "global"
        key = (panel_id, slug)
        if key in seen_annotations:
            return
        try:
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
        except (TypeError, ValueError):
            return
        if x <= 0 or y <= 0:
            return
        font_size = item.get("font_size")
        try:
            font_size_value = int(float(font_size)) if font_size is not None else 10
        except (TypeError, ValueError):
            font_size_value = 10
        anchor = str(item.get("anchor") or "start")
        fill = str(item.get("fill") or "#000000")
        annotations.append(
            {
                "id": f"txt_ann_{slug}_{panel_id}",
                "text": label,
                "x": x,
                "y": y,
                "anchor": anchor,
                "font_size": font_size_value,
                "fill": fill,
            }
        )
        seen_annotations.add(key)

    if text_mode_value != "template_text":
        for item in text_items:
            if item.get("render") is False:
                continue
            text_value = str(item.get("content") or item.get("text") or "")
            if not text_value:
                continue
            try:
                conf = float(item.get("conf", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if text_mode_value == "hybrid" and conf < min_conf and not text_sanity(text_value):
                continue
            normalized = text_value.strip()
            lowered = normalized.lower()
            if item.get("role") == "curve_label_serving":
                _maybe_add_annotation(item, normalized, "serving")
                continue
            if item.get("role") == "curve_label_neighbor":
                _maybe_add_annotation(item, normalized, "neighbor")
                continue
            if "ttt" in lowered:
                _maybe_add_annotation(item, "TTT", "ttt")
                continue
            if "hys" in lowered:
                _maybe_add_annotation(item, "Hys", "hys")
                continue
            if "trigger" in lowered:
                _maybe_add_annotation(item, "triggered", "triggered")
                continue

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
        "annotations": annotations,
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
    if text_mode_value == "template_text":
        serving_label = canonical.get("curve_label_serving")
        neighbor_label = canonical.get("curve_label_neighbor")
        if serving_label:
            params["style"]["curve_label_serving"] = serving_label
        if neighbor_label:
            params["style"]["curve_label_neighbor"] = neighbor_label
    extracted = params.get("extracted")
    if isinstance(extracted, dict):
        extracted.setdefault("ocr_stats", {})
        if isinstance(extracted["ocr_stats"], dict):
            extracted["ocr_stats"].update(
                {
                    "corrected_tokens": corrections,
                    "text_mode": text_mode_value,
                }
            )

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
    text_mode: str = "hybrid",
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
        text_mode=text_mode,
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
