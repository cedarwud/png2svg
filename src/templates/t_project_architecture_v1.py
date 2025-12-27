from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.svg_builder import DEFAULT_FONT_FAMILY, DEFAULT_TEXT_ANCHOR, SvgBuilder
from png2svg.errors import Png2SvgError

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

PANEL_IDS = ("A", "B", "C")
WP_IDS = ("WP1", "WP2", "WP3", "WP4")

STROKE = "#000000"
PANEL_FILL = "#f2f2f2"
CONTAINER_FILL = "#f2f2f2"
WP_FILL = "#ffffff"


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


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    title: str
    bullets: list[str]


@dataclass(frozen=True)
class WorkPackageSpec:
    wp_id: str
    title: str
    goal: str
    output: str


@dataclass(frozen=True)
class Layout:
    margin_x: float
    margin_y: float
    header_height: float
    panel_rects: dict[str, Rect]
    container: Rect
    wp_rects: dict[str, Rect]
    panel_gap: float
    container_padding: float
    container_label_height: float


def _clamp_font(value: float, min_value: int, max_value: int) -> int:
    return int(max(min_value, min(max_value, round(value))))


def _normalize_lines(value: Any) -> list[str]:
    if isinstance(value, list):
        lines = [str(item).strip() for item in value if str(item).strip()]
    else:
        lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    return lines


def _normalize_text_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _wrap_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _truncate_lines(lines: list[str], max_lines: int, max_chars: int) -> list[str]:
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    truncated = lines[:max_lines]
    suffix = "..."
    last = truncated[-1]
    if max_chars > len(suffix) + 1:
        limit = max_chars - len(suffix)
        if len(last) > limit:
            last = last[:limit].rstrip()
        truncated[-1] = f"{last}{suffix}"
    return truncated


def _max_chars_for_width(width: float, font_size: float) -> int:
    if font_size <= 0:
        return 4
    avg_char = font_size * 0.55
    return max(int(width / avg_char), 4)


def _bullet_lines(
    items: list[str], max_width: float, font_size: float, max_lines: int
) -> list[str]:
    max_chars = _max_chars_for_width(max_width, font_size)
    lines: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        per_line = max(max_chars - 2, 4)
        wrapped = _wrap_words(text, per_line)
        for idx, line in enumerate(wrapped):
            prefix = "- " if idx == 0 else "  "
            lines.append(f"{prefix}{line}")
    return _truncate_lines(lines, max_lines, max_chars)


def _panel_defaults() -> dict[str, PanelSpec]:
    return {
        "A": PanelSpec(
            panel_id="A",
            title="Panel A: Core Platform",
            bullets=["Common services", "Interfaces and APIs", "Scalable runtime"],
        ),
        "B": PanelSpec(
            panel_id="B",
            title="Panel B: Data and Analytics",
            bullets=["Ingestion and storage", "Analytics pipelines", "Dashboards"],
        ),
        "C": PanelSpec(
            panel_id="C",
            title="Panel C: Integration",
            bullets=["External systems", "Security and compliance", "Deployment ops"],
        ),
    }


def _work_package_defaults() -> dict[str, WorkPackageSpec]:
    return {
        "WP1": WorkPackageSpec(
            wp_id="WP1",
            title="WP1",
            goal="Requirements and scope",
            output="Architecture brief",
        ),
        "WP2": WorkPackageSpec(
            wp_id="WP2",
            title="WP2",
            goal="Core platform build",
            output="MVP services",
        ),
        "WP3": WorkPackageSpec(
            wp_id="WP3",
            title="WP3",
            goal="Data pipeline and UI",
            output="Reports and dashboards",
        ),
        "WP4": WorkPackageSpec(
            wp_id="WP4",
            title="WP4",
            goal="Integration and rollout",
            output="Release package",
        ),
    }


def _parse_panels(params: dict[str, Any]) -> list[PanelSpec]:
    defaults = _panel_defaults()
    raw = params.get("panels")
    if raw is None:
        return [defaults[panel_id] for panel_id in PANEL_IDS]
    if not isinstance(raw, list):
        raise Png2SvgError(
            code="E2301_PANELS_TYPE",
            message="panels must be a list.",
            hint="Provide panels as a list with ids A, B, and C.",
        )
    if len(raw) != len(PANEL_IDS):
        raise Png2SvgError(
            code="E2302_PANELS_COUNT",
            message="panels must contain exactly three entries (A, B, C).",
            hint="Provide panels for A, B, and C only.",
        )
    panels: dict[str, PanelSpec] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise Png2SvgError(
                code="E2303_PANEL_ENTRY",
                message="Each panel must be an object.",
                hint="Use {id, title, bullets} for each panel.",
            )
        panel_id = str(entry.get("id", "")).strip().upper()
        if panel_id not in PANEL_IDS:
            raise Png2SvgError(
                code="E2304_PANEL_ID",
                message=f"Unsupported panel id '{panel_id}'.",
                hint="Use panel ids A, B, and C.",
            )
        title = _normalize_text_value(
            entry.get("title") or entry.get("label") or defaults[panel_id].title
        )
        bullets_raw = entry.get("bullets") if "bullets" in entry else entry.get("items")
        if bullets_raw is None:
            bullets = defaults[panel_id].bullets
        else:
            bullets = _normalize_lines(bullets_raw)
        if not bullets:
            bullets = defaults[panel_id].bullets
        panels[panel_id] = PanelSpec(panel_id=panel_id, title=title, bullets=bullets)
    return [panels[panel_id] for panel_id in PANEL_IDS]


def _parse_work_packages(params: dict[str, Any]) -> list[WorkPackageSpec]:
    defaults = _work_package_defaults()
    raw = params.get("work_packages")
    if raw is None:
        return [defaults[wp_id] for wp_id in WP_IDS]
    if not isinstance(raw, list):
        raise Png2SvgError(
            code="E2305_WORKPACKAGES_TYPE",
            message="work_packages must be a list.",
            hint="Provide work_packages as a list with ids WP1-WP4.",
        )
    if len(raw) != len(WP_IDS):
        raise Png2SvgError(
            code="E2306_WORKPACKAGES_COUNT",
            message="work_packages must contain exactly four entries (WP1-WP4).",
            hint="Provide work_packages for WP1, WP2, WP3, and WP4 only.",
        )
    packages: dict[str, WorkPackageSpec] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise Png2SvgError(
                code="E2307_WORKPACKAGE_ENTRY",
                message="Each work package must be an object.",
                hint="Use {id, title, goal, output} for each work package.",
            )
        wp_id = str(entry.get("id", "")).strip().upper()
        if wp_id not in WP_IDS:
            raise Png2SvgError(
                code="E2308_WORKPACKAGE_ID",
                message=f"Unsupported work package id '{wp_id}'.",
                hint="Use work package ids WP1, WP2, WP3, and WP4.",
            )
        title = _normalize_text_value(entry.get("title") or defaults[wp_id].title)
        goal = _normalize_text_value(entry.get("goal") or defaults[wp_id].goal)
        output = _normalize_text_value(entry.get("output") or defaults[wp_id].output)
        packages[wp_id] = WorkPackageSpec(
            wp_id=wp_id,
            title=title,
            goal=goal,
            output=output,
        )
    return [packages[wp_id] for wp_id in WP_IDS]


def _layout(width: int, height: int) -> Layout:
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
    panel_rects: dict[str, Rect] = {}
    x = margin_x
    for panel_id in PANEL_IDS:
        panel_rects[panel_id] = Rect(x=x, y=top_y, width=panel_width, height=top_height)
        x += panel_width + gap_x
    container = Rect(
        x=margin_x,
        y=bottom_y,
        width=width - 2 * margin_x,
        height=bottom_height,
    )
    padding = max(int(height * 0.02), 18)
    label_height = max(int(height * 0.035), 28)
    wp_y = container.y + padding + label_height
    wp_height = max(container.height - padding * 2 - label_height, label_height)
    wp_gap = max(int(width * 0.015), 18)
    wp_width = (container.width - padding * 2 - wp_gap * 3) / 4.0
    wp_rects: dict[str, Rect] = {}
    x = container.x + padding
    for wp_id in WP_IDS:
        wp_rects[wp_id] = Rect(x=x, y=wp_y, width=wp_width, height=wp_height)
        x += wp_width + wp_gap
    return Layout(
        margin_x=margin_x,
        margin_y=margin_y,
        header_height=header_height,
        panel_rects=panel_rects,
        container=container,
        wp_rects=wp_rects,
        panel_gap=gap_x,
        container_padding=padding,
        container_label_height=label_height,
    )


def _add_text(
    builder: SvgBuilder,
    text_id: str,
    text: str,
    x: float,
    y: float,
    font_size: float,
    font_weight: str | None = None,
    anchor: str | None = None,
    fill: str = STROKE,
) -> None:
    text_group = builder.groups["g_text"]
    text_anchor = anchor or DEFAULT_TEXT_ANCHOR
    kwargs = {
        "insert": (x, y),
        "id": text_id,
        "font_family": DEFAULT_FONT_FAMILY,
        "font_size": float(font_size),
        "text_anchor": text_anchor,
        "fill": fill,
    }
    if font_weight:
        kwargs["font_weight"] = font_weight
    text_group.add(builder.drawing.text(text, **kwargs))


def _add_multiline_text(
    builder: SvgBuilder,
    text_id: str,
    lines: list[str],
    x: float,
    y: float,
    font_size: float,
    anchor: str | None = None,
    fill: str = STROKE,
    font_weight: str | None = None,
) -> None:
    if not lines:
        return
    if len(lines) == 1:
        _add_text(builder, text_id, lines[0], x, y, font_size, font_weight, anchor, fill)
        return
    text_group = builder.groups["g_text"]
    text_anchor = anchor or DEFAULT_TEXT_ANCHOR
    kwargs = {
        "insert": (x, y),
        "id": text_id,
        "font_family": DEFAULT_FONT_FAMILY,
        "font_size": float(font_size),
        "text_anchor": text_anchor,
        "fill": fill,
    }
    if font_weight:
        kwargs["font_weight"] = font_weight
    text = builder.drawing.text("", **kwargs)
    line_height = float(font_size) * 1.3
    for idx, line in enumerate(lines):
        if idx == 0:
            tspan = builder.drawing.tspan(line, x=[x], y=[y], id=f"{text_id}_line{idx}")
        else:
            tspan = builder.drawing.tspan(line, x=[x], dy=[line_height], id=f"{text_id}_line{idx}")
        text.add(tspan)
    text_group.add(text)


def _arrow_down(x: float, y: float, size: float) -> list[tuple[float, float]]:
    return [(x - size, y), (x + size, y), (x, y + size)]


def render(builder: SvgBuilder, params: dict[str, Any], canvas: tuple[int, int]) -> None:
    width, height = canvas
    allow_any_size = bool(params.get("allow_any_size", False))
    if not allow_any_size and (width, height) != (CANVAS_WIDTH, CANVAS_HEIGHT):
        raise Png2SvgError(
            code="E2300_CANVAS_SIZE",
            message=(
                f"t_project_architecture_v1 expects {CANVAS_WIDTH}x{CANVAS_HEIGHT} canvas, "
                f"got {width}x{height}."
            ),
            hint="Set canvas.width/height to 1920x1080 or pass allow_any_size=true.",
        )

    panels = _parse_panels(params)
    work_packages = _parse_work_packages(params)
    layout = _layout(width, height)

    title = _normalize_text_value(params.get("title") or "Project Architecture")
    subtitle = _normalize_text_value(params.get("subtitle") or "")

    title_font = _clamp_font(height * 0.028, 22, 34)
    subtitle_font = _clamp_font(height * 0.018, 12, 20)
    panel_title_font = _clamp_font(height * 0.017, 12, 20)
    panel_body_font = _clamp_font(height * 0.014, 10, 16)
    wp_title_font = _clamp_font(height * 0.016, 12, 18)
    wp_body_font = _clamp_font(height * 0.013, 10, 16)

    text_x = layout.margin_x
    title_y = layout.margin_y + title_font
    builder.add_title(title, x=text_x, y=title_y, font_size=title_font, font_weight="bold")
    if subtitle:
        subtitle_y = title_y + title_font * 0.9 + 6
        _add_text(builder, "txt_subtitle", subtitle, text_x, subtitle_y, subtitle_font)

    annotations = builder.groups["g_annotations"]
    markers = builder.groups["g_markers"]

    for panel in panels:
        rect = layout.panel_rects[panel.panel_id]
        annotations.add(
            builder.drawing.rect(
                insert=(rect.x, rect.y),
                size=(rect.width, rect.height),
                rx=12,
                ry=12,
                fill=PANEL_FILL,
                stroke=STROKE,
                stroke_width=2,
                id=f"panel_{panel.panel_id}",
            )
        )

    annotations.add(
        builder.drawing.rect(
            insert=(layout.container.x, layout.container.y),
            size=(layout.container.width, layout.container.height),
            rx=14,
            ry=14,
            fill=CONTAINER_FILL,
            stroke=STROKE,
            stroke_width=2,
            id="wp_container",
        )
    )

    for wp in work_packages:
        rect = layout.wp_rects[wp.wp_id]
        annotations.add(
            builder.drawing.rect(
                insert=(rect.x, rect.y),
                size=(rect.width, rect.height),
                rx=10,
                ry=10,
                fill=WP_FILL,
                stroke=STROKE,
                stroke_width=1,
                id=f"wp_box_{wp.wp_id}",
            )
        )

    arrow_size = 8
    for panel_id in PANEL_IDS:
        panel_rect = layout.panel_rects[panel_id]
        line_end = layout.container.y - arrow_size
        annotations.add(
            builder.drawing.line(
                start=(panel_rect.center_x, panel_rect.bottom),
                end=(panel_rect.center_x, line_end),
                stroke=STROKE,
                stroke_width=1,
                id=f"connector_{panel_id}",
            )
        )
        arrow_points = _arrow_down(panel_rect.center_x, line_end, arrow_size)
        markers.add(
            builder.drawing.polygon(points=arrow_points, fill=STROKE, id=f"arrow_{panel_id}")
        )

    for panel in panels:
        rect = layout.panel_rects[panel.panel_id]
        padding = layout.container_padding
        title_x = rect.x + padding
        title_y = rect.y + padding + panel_title_font
        _add_text(
            builder,
            f"txt_panel_{panel.panel_id}_title",
            panel.title,
            title_x,
            title_y,
            panel_title_font,
            font_weight="bold",
        )
        bullet_y = title_y + panel_title_font * 0.6 + 8
        max_bullet_height = rect.bottom - bullet_y - padding
        line_height = panel_body_font * 1.3
        max_lines = max(int(max_bullet_height / line_height), 1)
        bullets = _bullet_lines(panel.bullets, rect.width - padding * 2, panel_body_font, max_lines)
        _add_multiline_text(
            builder,
            f"txt_panel_{panel.panel_id}_bullets",
            bullets,
            title_x,
            bullet_y,
            panel_body_font,
        )

    container_label_x = layout.container.x + layout.container_padding
    container_label_y = layout.container.y + layout.container_padding + wp_title_font
    _add_text(
        builder,
        "txt_wp_container_label",
        "Work Packages",
        container_label_x,
        container_label_y,
        wp_title_font,
        font_weight="bold",
    )

    for wp in work_packages:
        rect = layout.wp_rects[wp.wp_id]
        padding = layout.container_padding
        title_x = rect.x + padding
        title_y = rect.y + padding + wp_title_font
        _add_text(
            builder,
            f"txt_{wp.wp_id}_title",
            wp.title,
            title_x,
            title_y,
            wp_title_font,
            font_weight="bold",
        )
        body_y = title_y + wp_title_font * 0.6 + 6
        max_body_height = rect.bottom - body_y - padding
        line_height = wp_body_font * 1.3
        max_lines = max(int(max_body_height / line_height), 1)
        goal_lines = _wrap_words(
            f"Goal: {wp.goal}",
            _max_chars_for_width(rect.width - padding * 2, wp_body_font),
        )
        output_lines = _wrap_words(
            f"Output: {wp.output}",
            _max_chars_for_width(rect.width - padding * 2, wp_body_font),
        )
        body_lines = goal_lines + output_lines
        body_lines = _truncate_lines(
            body_lines,
            max_lines,
            _max_chars_for_width(rect.width - padding * 2, wp_body_font),
        )
        _add_multiline_text(
            builder,
            f"txt_{wp.wp_id}_details",
            body_lines,
            title_x,
            body_y,
            wp_body_font,
        )


__all__ = ["render"]
