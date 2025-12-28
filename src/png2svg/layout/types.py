from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["ROOT", "ROW", "COL", "PANEL", "HEADER", "FOOTER", "LEAF"]


@dataclass
class LayoutNode:
    """A node in the layout hierarchy tree.

    Attributes:
        id: Unique identifier for the node
        type: Semantic type of the region (ROOT, ROW, COL, etc.)
        bbox: Bounding box (x, y, width, height)
        children: Child nodes
        content: Dictionary containing assigned content (e.g., extracted text, curves)
        debug_info: Optional debug metadata (e.g., separator confidence)
    """

    id: str
    type: NodeType
    bbox: tuple[int, int, int, int]
    children: list[LayoutNode] = field(default_factory=list)
    content: dict[str, Any] = field(default_factory=dict)
    debug_info: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def x2(self) -> int:
        return self.bbox[0] + self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[1] + self.bbox[3]

    def add_child(self, node: LayoutNode) -> None:
        self.children.append(node)

    def to_dict(self) -> dict[str, Any]:
        """Recursive serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "bbox": self.bbox,
            "children": [child.to_dict() for child in self.children],
            "content": self.content,
        }
