from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .svg_checks import local_name


def _parse_svg(svg_path: Path | str) -> ET.Element:
    tree = ET.parse(svg_path)
    return tree.getroot()


def _element_context(node: ET.Element) -> dict[str, str]:
    context: dict[str, str] = {"tag": local_name(node.tag)}
    node_id = node.get("id")
    if node_id:
        context["id"] = node_id
    return context


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}
