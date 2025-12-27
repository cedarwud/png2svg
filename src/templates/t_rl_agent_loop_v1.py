from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.svg_builder import DEFAULT_FONT_FAMILY, DEFAULT_TEXT_ANCHOR, SvgBuilder
from png2svg.errors import Png2SvgError

STROKE = "#000000"
AGENT_FILL = "#e7f1ff"
ENV_FILL = "#fff1e0"
BOX_FILL = "#f2f2f2"


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

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


def _split_lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _layout(width: int, height: int) -> dict[str, Rect]:
    margin_x = max(int(width * 0.08), 40)
    margin_y = max(int(height * 0.06), 30)
    header_height = max(int(height * 0.14), 90)
    box_width = width * 0.28
    box_height = height * 0.2
    gap_y = max(int(height * 0.08), 50)
    main_y = margin_y + header_height
    agent = Rect(
        x=margin_x,
        y=main_y,
        width=box_width,
        height=box_height,
    )
    env = Rect(
        x=width - margin_x - box_width,
        y=main_y,
        width=box_width,
        height=box_height,
    )
    constraint = Rect(
        x=(width - box_width * 0.7) / 2.0,
        y=max(margin_y + header_height * 0.4, main_y - box_height * 0.7),
        width=box_width * 0.7,
        height=box_height * 0.45,
    )
    buffer = Rect(
        x=(width - box_width * 0.75) / 2.0,
        y=main_y + box_height + gap_y,
        width=box_width * 0.75,
        height=box_height * 0.45,
    )
    return {
        "agent": agent,
        "environment": env,
        "constraint": constraint,
        "buffer": buffer,
        "margin_x": Rect(x=margin_x, y=margin_y, width=0, height=0),
    }


def _parse_box_label(params: dict[str, Any], role: str, default: str) -> tuple[str, bool]:
    boxes = params.get("boxes")
    if isinstance(boxes, list):
        for entry in boxes:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("role") or entry.get("id") or "").lower() != role:
                continue
            label = entry.get("label") or entry.get("text") or default
            enabled = entry.get("enabled")
            return str(label).strip(), False if enabled is False else True
    label = params.get(role) or default
    return str(label).strip(), True


def _parse_signal(params: dict[str, Any], key: str, default: str) -> str:
    signals = params.get("signals")
    if isinstance(signals, dict) and key in signals:
        return str(signals.get(key) or default).strip()
    return str(params.get(key) or default).strip()


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


def _arrow_head(x: float, y: float, direction: str, size: float = 8) -> list[tuple[float, float]]:
    if direction == "right":
        return [(x - size, y - size * 0.6), (x - size, y + size * 0.6), (x, y)]
    if direction == "left":
        return [(x + size, y - size * 0.6), (x + size, y + size * 0.6), (x, y)]
    if direction == "up":
        return [(x - size * 0.6, y + size), (x + size * 0.6, y + size), (x, y)]
    return [(x - size * 0.6, y - size), (x + size * 0.6, y - size), (x, y)]


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    width, height = canvas
    if width <= 0 or height <= 0:
        raise Png2SvgError(
            code="E2400_CANVAS_RANGE",
            message="Canvas size must be positive.",
            hint="Provide a positive canvas width/height.",
        )

    layout = _layout(width, height)
    agent_label, _ = _parse_box_label(params, "agent", "Agent")
    env_label, _ = _parse_box_label(params, "environment", "Environment")
    constraint_label, constraint_enabled = _parse_box_label(params, "constraint", "Constraints")
    buffer_label, buffer_enabled = _parse_box_label(params, "buffer", "Replay Buffer")

    title = str(params.get("title") or "RL Agent Loop").strip()
    action_label = _parse_signal(params, "action", "Action")
    feedback_label = _parse_signal(params, "feedback", "").strip()
    if not feedback_label:
        state_label = _parse_signal(params, "state", "State")
        reward_label = _parse_signal(params, "reward", "Reward")
        feedback_label = f"{state_label} / {reward_label}" if reward_label else state_label

    annotations = builder.groups["g_annotations"]
    g_boxes = builder.drawing.g(id="g_boxes")
    g_arrows = builder.drawing.g(id="g_arrows")
    annotations.add(g_boxes)
    annotations.add(g_arrows)
    markers = builder.groups["g_markers"]

    text_group = builder.groups["g_text"]
    g_title = builder.drawing.g(id="g_title")
    g_legend = builder.drawing.g(id="g_legend")
    text_group.add(g_title)
    text_group.add(g_legend)

    title_font = max(int(height * 0.04), 18)
    label_font = max(int(height * 0.025), 12)
    arrow_font = max(int(height * 0.022), 11)

    if title:
        _add_multiline_text(
            g_title,
            builder.drawing,
            "txt_title",
            _split_lines(title),
            layout["margin_x"].x,
            layout["margin_x"].y + title_font,
            title_font,
            anchor="start",
            font_weight="bold",
        )

    for role in ("agent", "environment"):
        rect = layout[role]
        fill = AGENT_FILL if role == "agent" else ENV_FILL
        g_boxes.add(
            builder.drawing.rect(
                insert=(rect.x, rect.y),
                size=(rect.width, rect.height),
                rx=12,
                ry=12,
                fill=fill,
                stroke=STROKE,
                stroke_width=2,
                id=f"box_{role}",
            )
        )

    if constraint_enabled:
        rect = layout["constraint"]
        g_boxes.add(
            builder.drawing.rect(
                insert=(rect.x, rect.y),
                size=(rect.width, rect.height),
                rx=10,
                ry=10,
                fill="#ffffff",
                stroke=STROKE,
                stroke_width=1,
                id="box_constraint",
            )
        )
    if buffer_enabled:
        rect = layout["buffer"]
        g_boxes.add(
            builder.drawing.rect(
                insert=(rect.x, rect.y),
                size=(rect.width, rect.height),
                rx=10,
                ry=10,
                fill="#ffffff",
                stroke=STROKE,
                stroke_width=1,
                id="box_buffer",
            )
        )

    agent = layout["agent"]
    env = layout["environment"]
    action_y = agent.center_y + agent.height * 0.25
    feedback_y = agent.center_y - agent.height * 0.25

    g_arrows.add(
        builder.drawing.line(
            start=(agent.right, action_y),
            end=(env.x, action_y),
            stroke=STROKE,
            stroke_width=1,
            id="arrow_action",
        )
    )
    markers.add(
        builder.drawing.polygon(
            points=_arrow_head(env.x, action_y, "right"),
            fill=STROKE,
            id="arrowhead_action",
        )
    )
    g_arrows.add(
        builder.drawing.line(
            start=(env.x, feedback_y),
            end=(agent.right, feedback_y),
            stroke=STROKE,
            stroke_width=1,
            id="arrow_feedback",
        )
    )
    markers.add(
        builder.drawing.polygon(
            points=_arrow_head(agent.right, feedback_y, "left"),
            fill=STROKE,
            id="arrowhead_feedback",
        )
    )

    if constraint_enabled:
        rect = layout["constraint"]
        g_arrows.add(
            builder.drawing.line(
                start=(rect.center_x, rect.bottom),
                end=(agent.center_x, agent.y),
                stroke=STROKE,
                stroke_width=1,
                id="arrow_constraint",
            )
        )
        markers.add(
            builder.drawing.polygon(
                points=_arrow_head(agent.center_x, agent.y, "down"),
                fill=STROKE,
                id="arrowhead_constraint",
            )
        )
    if buffer_enabled:
        rect = layout["buffer"]
        g_arrows.add(
            builder.drawing.line(
                start=(agent.center_x, agent.bottom),
                end=(rect.center_x, rect.y),
                stroke=STROKE,
                stroke_width=1,
                id="arrow_buffer",
            )
        )
        markers.add(
            builder.drawing.polygon(
                points=_arrow_head(rect.center_x, rect.y, "down"),
                fill=STROKE,
                id="arrowhead_buffer",
            )
        )

    _add_multiline_text(
        text_group,
        builder.drawing,
        "txt_agent",
        _split_lines(agent_label),
        agent.center_x,
        agent.center_y + label_font * 0.4,
        label_font,
        anchor="middle",
        font_weight="bold",
    )
    _add_multiline_text(
        text_group,
        builder.drawing,
        "txt_environment",
        _split_lines(env_label),
        env.center_x,
        env.center_y + label_font * 0.4,
        label_font,
        anchor="middle",
        font_weight="bold",
    )
    if constraint_enabled:
        rect = layout["constraint"]
        _add_multiline_text(
            text_group,
            builder.drawing,
            "txt_constraint",
            _split_lines(constraint_label),
            rect.center_x,
            rect.center_y + label_font * 0.3,
            label_font * 0.85,
            anchor="middle",
        )
    if buffer_enabled:
        rect = layout["buffer"]
        _add_multiline_text(
            text_group,
            builder.drawing,
            "txt_buffer",
            _split_lines(buffer_label),
            rect.center_x,
            rect.center_y + label_font * 0.3,
            label_font * 0.85,
            anchor="middle",
        )

    _add_multiline_text(
        text_group,
        builder.drawing,
        "txt_action",
        _split_lines(action_label),
        (agent.right + env.x) / 2.0,
        action_y - arrow_font * 0.6,
        arrow_font,
        anchor="middle",
    )
    _add_multiline_text(
        text_group,
        builder.drawing,
        "txt_feedback",
        _split_lines(feedback_label),
        (agent.right + env.x) / 2.0,
        feedback_y - arrow_font * 0.6,
        arrow_font,
        anchor="middle",
    )


__all__ = ["render"]
