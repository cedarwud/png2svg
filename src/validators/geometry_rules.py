from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .config import FigureContract, GeometryThresholds
from .report import ValidationIssue
from .svg_checks import (
    count_path_commands,
    extract_property_values,
    iter_with_style,
    local_name,
    parse_color,
    parse_number,
    parse_style,
)
from .validate_constants import (
    E2001_FORBIDDEN_ELEMENT,
    E2002_FORBIDDEN_PREFIX,
    E2003_MISSING_GROUP,
    E2005_TOO_MANY_COLORS,
    E2006_BAD_STROKE_WIDTH,
    E2007_PATH_TOO_COMPLEX,
    E2013_DASHED_MISSING_DASHARRAY,
    W2101_LINE_NOT_SNAPPED,
    W2102_DASHED_SIMULATED,
    W2104_POLYLINE_TOO_COMPLEX,
    W2105_POLYLINE_NOT_SNAPPED,
)
from .xml_utils import _element_context


def _has_dasharray(node: ET.Element, style: dict[str, str]) -> bool:
    values = extract_property_values(node, style, "stroke-dasharray")
    for value in values:
        lowered = value.strip().lower()
        if lowered and lowered not in {"none", "0"}:
            return True
    return False


def _is_marked_dashed(node: ET.Element) -> bool:
    class_attr = (node.get("class") or "").lower()
    if any(token in {"dash", "dashed"} for token in class_attr.split()):
        return True
    data_dashed = (node.get("data-dashed") or "").lower()
    return data_dashed in {"1", "true", "yes"}


def _check_forbidden_elements(
    elements: list[ET.Element], contract: FigureContract
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    forbidden_set = set(contract.forbid_elements)
    prefix_set = {prefix.lower() for prefix in contract.forbid_element_prefixes}
    for node in elements:
        name = local_name(node.tag)
        if name in forbidden_set:
            issues.append(
                ValidationIssue(
                    code=E2001_FORBIDDEN_ELEMENT,
                    message=f"Forbidden element <{name}> found.",
                    hint="Remove the element and replace it with vector primitives (<line>, <rect>, <path>, <text>).",
                    context={"element": name, **_element_context(node)},
                )
            )
        lowered = name.lower()
        if any(lowered.startswith(prefix) for prefix in prefix_set):
            issues.append(
                ValidationIssue(
                    code=E2002_FORBIDDEN_PREFIX,
                    message=f"Forbidden element prefix in <{name}>.",
                    hint="Remove fe* filter primitives and any filter references.",
                    context={"element": name, **_element_context(node)},
                )
            )
    return issues


def _check_required_groups(
    elements: list[ET.Element], contract: FigureContract
) -> list[ValidationIssue]:
    group_ids = {
        node.get("id")
        for node in elements
        if local_name(node.tag) == "g" and node.get("id")
    }
    issues: list[ValidationIssue] = []
    for required in contract.required_groups:
        if required not in group_ids:
            issues.append(
                ValidationIssue(
                    code=E2003_MISSING_GROUP,
                    message=f"Required group id '{required}' is missing.",
                    hint=f"Add <g id=\"{required}\"> and place the expected elements inside it.",
                    context={"tag": "g", "id": required},
                )
            )
    return issues


def _parse_polyline_points(points: str) -> list[tuple[float, float]]:
    coords: list[float] = []
    for token in points.replace(",", " ").split():
        try:
            coords.append(float(token))
        except ValueError:
            return []
    if len(coords) < 2 or len(coords) % 2 != 0:
        return []
    return list(zip(coords[0::2], coords[1::2]))


def _check_polyline_complexity(
    elements: list[ET.Element], thresholds: GeometryThresholds
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    max_points = thresholds.polyline_max_points
    for node in elements:
        if local_name(node.tag) != "polyline":
            continue
        points_attr = node.get("points", "")
        points = _parse_polyline_points(points_attr)
        if not points:
            continue
        if len(points) > max_points:
            warnings.append(
                ValidationIssue(
                    code=W2104_POLYLINE_TOO_COMPLEX,
                    message=f"Polyline has {len(points)} points (limit {max_points}).",
                    hint="Simplify polylines to fewer points to avoid traced look.",
                    context={"point_count": len(points), "max_points": max_points, **_element_context(node)},
                )
            )
    return warnings


def _check_polyline_snapping(
    elements: list[ET.Element], thresholds: GeometryThresholds
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    tolerance = thresholds.snap_tolerance
    for node in elements:
        if local_name(node.tag) != "polyline":
            continue
        points_attr = node.get("points", "")
        points = _parse_polyline_points(points_attr)
        if len(points) < 2:
            continue
        near_horizontal = False
        near_vertical = False
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            dx = x2 - x1
            dy = y2 - y1
            if 0 < abs(dy) <= tolerance:
                near_horizontal = True
            if 0 < abs(dx) <= tolerance:
                near_vertical = True
        if near_horizontal:
            warnings.append(
                ValidationIssue(
                    code=W2105_POLYLINE_NOT_SNAPPED,
                    message="Polyline has a near-horizontal segment that is not snapped.",
                    hint="Snap polyline segment coordinates so y-values match exactly.",
                    context={"axis": "horizontal", **_element_context(node)},
                )
            )
        if near_vertical:
            warnings.append(
                ValidationIssue(
                    code=W2105_POLYLINE_NOT_SNAPPED,
                    message="Polyline has a near-vertical segment that is not snapped.",
                    hint="Snap polyline segment coordinates so x-values match exactly.",
                    context={"axis": "vertical", **_element_context(node)},
                )
            )
    return warnings


def _check_colors(
    elements: list[ET.Element], contract: FigureContract
) -> tuple[list[ValidationIssue], int]:
    max_colors = contract.max_colors
    colors: set[object] = set()
    example_context: dict[str, str] | None = None
    for node, style in iter_with_style(elements):
        for prop in ("fill", "stroke"):
            for value in extract_property_values(node, style, prop):
                parsed = parse_color(value)
                if parsed is not None:
                    colors.add(parsed)
                    if example_context is None:
                        example_context = _element_context(node)
    color_count = len(colors)
    if max_colors is None or color_count <= max_colors:
        return [], color_count
    context = {
        "color_count": color_count,
        "max_colors": max_colors,
        "colors": [str(color) for color in sorted(colors, key=str)[:12]],
    }
    if example_context:
        context.update(example_context)
    else:
        context["tag"] = "svg"
    issue = ValidationIssue(
        code=E2005_TOO_MANY_COLORS,
        message=f"Color count {color_count} exceeds max_colors {max_colors}.",
        hint="Reduce the palette by reusing fill/stroke colors across elements.",
        context=context,
    )
    return [issue], color_count


def _check_stroke_widths(
    elements: list[ET.Element], contract: FigureContract
) -> list[ValidationIssue]:
    allowed = contract.allowed_stroke_widths
    if not allowed:
        return []
    tolerance = contract.stroke_width_tolerance
    seen_invalid: set[str] = set()
    issues: list[ValidationIssue] = []
    for node, style in iter_with_style(elements):
        stroke_values = extract_property_values(node, style, "stroke")
        stroke_is_none = any(value.strip().lower() == "none" for value in stroke_values)
        if stroke_is_none:
            continue
        for value in extract_property_values(node, style, "stroke-width"):
            parsed = parse_number(value)
            if parsed is None:
                if value in seen_invalid:
                    continue
                seen_invalid.add(value)
                issues.append(
                    ValidationIssue(
                        code=E2006_BAD_STROKE_WIDTH,
                        message=f"Unparseable stroke-width '{value}'.",
                        hint="Use numeric stroke widths that match the allowed set (e.g., 1 or 2).",
                        context={"stroke_width": value, **_element_context(node)},
                    )
                )
                continue
            if any(abs(parsed - allowed_value) <= tolerance for allowed_value in allowed):
                continue
            key = f"{parsed:.6g}"
            if key in seen_invalid:
                continue
            seen_invalid.add(key)
            issues.append(
                ValidationIssue(
                    code=E2006_BAD_STROKE_WIDTH,
                    message=f"Stroke-width {parsed} is not within tolerance.",
                    hint="Use allowed stroke widths within tolerance (e.g., 1 or 2).",
                    context={
                        "stroke_width": parsed,
                        "allowed": allowed,
                        "tolerance": tolerance,
                        **_element_context(node),
                    },
                )
            )
    return issues


def _check_path_complexity(
    elements: list[ET.Element], contract: FigureContract
) -> tuple[list[ValidationIssue], int]:
    max_commands = contract.max_path_commands
    if max_commands is None:
        return [], 0
    issues: list[ValidationIssue] = []
    max_seen = 0
    for node in elements:
        if local_name(node.tag) != "path":
            continue
        d = node.get("d")
        if not d:
            continue
        command_count = count_path_commands(d)
        max_seen = max(max_seen, command_count)
        if command_count <= max_commands:
            continue
        issues.append(
            ValidationIssue(
                code=E2007_PATH_TOO_COMPLEX,
                message=f"Path command count {command_count} exceeds limit {max_commands}.",
                hint="Reduce path complexity by using fewer segments or simplifying curves.",
                context={
                    "command_count": command_count,
                    "max_commands": max_commands,
                    **_element_context(node),
                },
            )
        )
    return issues, max_seen


def _check_dasharray_required(elements: list[ET.Element]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for node, style in iter_with_style(elements):
        tag = local_name(node.tag)
        if tag not in {"line", "polyline", "path"}:
            continue
        if _is_marked_dashed(node) and not _has_dasharray(node, style):
            issues.append(
                ValidationIssue(
                    code=E2013_DASHED_MISSING_DASHARRAY,
                    message="Dashed element is missing stroke-dasharray.",
                    hint="Use stroke-dasharray on dashed lines instead of custom segments.",
                    context=_element_context(node),
                )
            )
    return issues


def _check_line_snapping(
    elements: list[ET.Element], thresholds: GeometryThresholds
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    tolerance = thresholds.snap_tolerance
    for node in elements:
        if local_name(node.tag) != "line":
            continue
        x1 = parse_number(node.get("x1", ""))
        y1 = parse_number(node.get("y1", ""))
        x2 = parse_number(node.get("x2", ""))
        y2 = parse_number(node.get("y2", ""))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        dx = x2 - x1
        dy = y2 - y1
        if 0 < abs(dy) <= tolerance:
            warnings.append(
                ValidationIssue(
                    code=W2101_LINE_NOT_SNAPPED,
                    message="Line is nearly horizontal but not snapped.",
                    hint="Snap the line to a perfect horizontal by equalizing y values.",
                    context={"axis": "horizontal", **_element_context(node)},
                )
            )
        if 0 < abs(dx) <= tolerance:
            warnings.append(
                ValidationIssue(
                    code=W2101_LINE_NOT_SNAPPED,
                    message="Line is nearly vertical but not snapped.",
                    hint="Snap the line to a perfect vertical by equalizing x values.",
                    context={"axis": "vertical", **_element_context(node)},
                )
            )
    return warnings


def _check_dashed_simulation(
    elements: list[ET.Element], thresholds: GeometryThresholds
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    tolerance = thresholds.snap_tolerance
    min_count = thresholds.dash_segment_min_count
    if min_count <= 0:
        return warnings
    groups: dict[tuple[str, float, str, float], list[ET.Element]] = {}
    for node in elements:
        if local_name(node.tag) != "line":
            continue
        x1 = parse_number(node.get("x1", ""))
        y1 = parse_number(node.get("y1", ""))
        x2 = parse_number(node.get("x2", ""))
        y2 = parse_number(node.get("y2", ""))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        dx = x2 - x1
        dy = y2 - y1
        axis = None
        coord = None
        if abs(dy) <= tolerance and abs(dx) > 0:
            axis = "h"
            coord = (y1 + y2) / 2.0
        elif abs(dx) <= tolerance and abs(dy) > 0:
            axis = "v"
            coord = (x1 + x2) / 2.0
        if axis is None or coord is None:
            continue
        style = parse_style(node.get("style", ""))
        stroke_values = extract_property_values(node, style, "stroke")
        stroke = stroke_values[0] if stroke_values else ""
        stroke_width_value = extract_property_values(node, style, "stroke-width")
        stroke_width = parse_number(stroke_width_value[0]) if stroke_width_value else 0.0
        key = (axis, coord, stroke, stroke_width or 0.0)
        groups.setdefault(key, []).append(node)

    for key, nodes in groups.items():
        if len(nodes) < min_count:
            continue
        axis, coord, stroke, stroke_width = key
        context = {
            "axis": "horizontal" if axis == "h" else "vertical",
            "coord": coord,
            "segment_count": len(nodes),
            "stroke": stroke,
            "stroke_width": stroke_width,
            **_element_context(nodes[0]),
        }
        warnings.append(
            ValidationIssue(
                code=W2102_DASHED_SIMULATED,
                message="Multiple short segments suggest a dashed line built from segments.",
                hint="Use a single line with stroke-dasharray for dashed styles.",
                context=context,
            )
        )
    return warnings
