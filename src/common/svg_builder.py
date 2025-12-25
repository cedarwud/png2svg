from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import svgwrite

REQUIRED_GROUP_IDS = [
    "figure_root",
    "g_axes",
    "g_curves",
    "g_annotations",
    "g_text",
    "g_markers",
]

DEFAULT_FONT_FAMILY = "Arial"


@dataclass
class SvgBuilder:
    drawing: svgwrite.Drawing
    root: svgwrite.container.Group
    groups: dict[str, svgwrite.container.Group]
    width: int
    height: int

    @classmethod
    def create(cls, width: int, height: int) -> "SvgBuilder":
        drawing = svgwrite.Drawing(size=(width, height), profile="full")
        root = drawing.g(id="figure_root")
        drawing.add(root)

        groups: dict[str, svgwrite.container.Group] = {}
        for group_id in REQUIRED_GROUP_IDS:
            if group_id == "figure_root":
                continue
            group = drawing.g(id=group_id)
            root.add(group)
            groups[group_id] = group

        return cls(
            drawing=drawing,
            root=root,
            groups=groups,
            width=int(width),
            height=int(height),
        )

    def add_title(self, title: str, x: int = 10, y: int = 20) -> None:
        text_group = self.groups["g_text"]
        text_group.add(
            self.drawing.text(
                title,
                insert=(x, y),
                id="txt_title",
                font_family=DEFAULT_FONT_FAMILY,
                fill="#000000",
            )
        )

    def add_axes_placeholder(self, margin: int = 10) -> None:
        axes_group = self.groups["g_axes"]
        stroke = "#000000"
        stroke_width = 2
        axes_group.add(
            self.drawing.line(
                start=(margin, self.height - margin),
                end=(self.width - margin, self.height - margin),
                stroke=stroke,
                stroke_width=stroke_width,
            )
        )
        axes_group.add(
            self.drawing.line(
                start=(margin, self.height - margin),
                end=(margin, margin),
                stroke=stroke,
                stroke_width=stroke_width,
            )
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.drawing.saveas(str(path))
