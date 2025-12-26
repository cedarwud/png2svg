from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .config import (
    FigureContract,
    GeometryThresholds,
    load_contract,
    load_geometry_thresholds,
    load_thresholds,
    load_visual_diff_thresholds,
)
from .report import ValidationIssue, ValidationReport
from .svg_checks import (
    count_path_commands,
    extract_property_values,
    iter_with_style,
    local_name,
    parse_color,
    parse_number,
    parse_style,
)
from .visual_diff import DiffError, RasterizeError, compute_visual_diff, rasterize_svg_to_png

E1000_PARSE_ERROR = "E1000_PARSE_ERROR"
E1001_CONFIG_ERROR = "E1001_CONFIG_ERROR"
E1002_THRESHOLDS_ERROR = "E1002_THRESHOLDS_ERROR"
E2001_FORBIDDEN_ELEMENT = "E2001_FORBIDDEN_ELEMENT"
E2002_FORBIDDEN_PREFIX = "E2002_FORBIDDEN_PREFIX"
E2003_MISSING_GROUP = "E2003_MISSING_GROUP"
E2004_BAD_FONT_FAMILY = "E2004_BAD_FONT_FAMILY"
E2005_TOO_MANY_COLORS = "E2005_TOO_MANY_COLORS"
E2006_BAD_STROKE_WIDTH = "E2006_BAD_STROKE_WIDTH"
E2007_PATH_TOO_COMPLEX = "E2007_PATH_TOO_COMPLEX"
E2008_TEXT_MISSING = "E2008_TEXT_MISSING"
E2009_TEXT_ID_MISSING = "E2009_TEXT_ID_MISSING"
E2010_TEXT_AS_PATH = "E2010_TEXT_AS_PATH"
E2011_TSPAN_ID_MISSING = "E2011_TSPAN_ID_MISSING"
E2012_MULTILINE_TSPAN_MISSING = "E2012_MULTILINE_TSPAN_MISSING"
E2013_DASHED_MISSING_DASHARRAY = "E2013_DASHED_MISSING_DASHARRAY"
E2014_TEXT_COUNT_LOW = "E2014_TEXT_COUNT_LOW"
E2015_TEXT_FALLBACK_MISSING = "E2015_TEXT_FALLBACK_MISSING"
E2016_TEXT_ANCHOR_INVALID = "E2016_TEXT_ANCHOR_INVALID"
E2017_TEXT_OUTLINE_DETECTED = "E2017_TEXT_OUTLINE_DETECTED"
E3001_RASTERIZE_FAILED = "E3001_RASTERIZE_FAILED"
E3002_DIFF_FAILED = "E3002_DIFF_FAILED"
E3003_VISUAL_THRESHOLD_EXCEEDED = "E3003_VISUAL_THRESHOLD_EXCEEDED"
W2101_LINE_NOT_SNAPPED = "W2101_LINE_NOT_SNAPPED"
W2102_DASHED_SIMULATED = "W2102_DASHED_SIMULATED"
W2103_TEXT_OUTSIDE_TEXT_GROUP = "W2103_TEXT_OUTSIDE_TEXT_GROUP"
W2104_POLYLINE_TOO_COMPLEX = "W2104_POLYLINE_TOO_COMPLEX"
W2105_POLYLINE_NOT_SNAPPED = "W2105_POLYLINE_NOT_SNAPPED"
W2106_TEXT_BASELINE_NOT_ALIGNED = "W2106_TEXT_BASELINE_NOT_ALIGNED"


@dataclass(frozen=True)
class TextThresholds:
    min_keep_ratio: float
    max_missing: int
    min_keep_absolute: int
    outline_path_min: int
    baseline_tolerance: float


def _font_family_ok(value: str, allowed: list[str]) -> bool:
    if not value:
        return False
    allowed_set = {item.strip().strip("'\"").lower() for item in allowed}
    candidates = [
        item.strip().strip("'\"").lower() for item in value.split(",") if item.strip()
    ]
    return any(candidate in allowed_set for candidate in candidates)


def _font_family_candidates(value: str) -> list[str]:
    return [item.strip().strip("'\"").lower() for item in value.split(",") if item.strip()]


def _font_family_has_generic(value: str) -> bool:
    candidates = _font_family_candidates(value)
    return any(candidate in {"sans-serif", "serif"} for candidate in candidates)


def _load_text_thresholds(thresholds: dict | None) -> TextThresholds:
    data = thresholds.get("text", {}) if isinstance(thresholds, dict) else {}
    return TextThresholds(
        min_keep_ratio=float(data.get("min_keep_ratio", 0.7)),
        max_missing=int(data.get("max_missing", 1)),
        min_keep_absolute=int(data.get("min_keep_absolute", 1)),
        outline_path_min=int(data.get("outline_path_min", 8)),
        baseline_tolerance=float(data.get("baseline_tolerance", 1.0)),
    )


def _resolve_property(
    node: ET.Element,
    style: dict[str, str],
    parent_map: dict[ET.Element, ET.Element],
    prop: str,
) -> str:
    values = extract_property_values(node, style, prop)
    if values:
        return values[0]
    current = parent_map.get(node)
    while current is not None:
        parent_style = parse_style(current.get("style", ""))
        values = extract_property_values(current, parent_style, prop)
        if values:
            return values[0]
        current = parent_map.get(current)
    return ""


def _parse_svg(svg_path: Path) -> ET.Element:
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


def _has_ancestor_group(node: ET.Element, parent_map: dict[ET.Element, ET.Element], group_id: str) -> bool:
    current = parent_map.get(node)
    while current is not None:
        if local_name(current.tag) == "g" and current.get("id") == group_id:
            return True
        current = parent_map.get(current)
    return False


def _text_has_newline(node: ET.Element) -> bool:
    if node.text and "\n" in node.text:
        return True
    for child in node:
        if child.tail and "\n" in child.tail:
            return True
    return False


def _tspan_elements(node: ET.Element) -> list[ET.Element]:
    return [child for child in node if local_name(child.tag) == "tspan"]


def _looks_like_text_path(node: ET.Element) -> bool:
    node_id = (node.get("id") or "").lower()
    if node_id.startswith(("txt_", "text_", "glyph_")):
        return True
    class_attr = (node.get("class") or "").lower()
    return any(token in class_attr for token in ("text", "glyph", "font"))


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


def _check_font_families(
    elements: list[ET.Element],
    contract: FigureContract,
    parent_map: dict[ET.Element, ET.Element],
) -> list[ValidationIssue]:
    allowed = contract.allowed_font_families
    if not allowed:
        return []
    text_elements = [node for node in elements if local_name(node.tag) == "text"]
    if not text_elements:
        return []
    issues: list[ValidationIssue] = []
    for node, style in iter_with_style(text_elements):
        value = _resolve_property(node, style, parent_map, "font-family")
        if not value:
            issues.append(
                ValidationIssue(
                    code=E2004_BAD_FONT_FAMILY,
                    message="Missing font-family on text element.",
                    hint=f"Set font-family to include one of: {', '.join(allowed)}.",
                    context=_element_context(node),
                )
            )
            continue
        if not _font_family_ok(value, allowed):
            issues.append(
                ValidationIssue(
                    code=E2004_BAD_FONT_FAMILY,
                    message=f"Font-family '{value}' is not allowed.",
                    hint=f"Use one of: {', '.join(allowed)}.",
                    context={"font_family": value, **_element_context(node)},
                )
            )
            continue
        if not _font_family_has_generic(value):
            issues.append(
                ValidationIssue(
                    code=E2015_TEXT_FALLBACK_MISSING,
                    message="Font-family must include a generic fallback (sans-serif or serif).",
                    hint="Append a generic fallback, e.g., 'Arial, sans-serif'.",
                    context={"font_family": value, **_element_context(node)},
                )
            )
    return issues


def _check_text_requirements(
    elements: list[ET.Element], contract: FigureContract
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    text_elements = [node for node in elements if local_name(node.tag) == "text"]
    if contract.require_text_elements and not text_elements:
        issues.append(
            ValidationIssue(
                code=E2008_TEXT_MISSING,
                message="No <text> elements found, but text is required.",
                hint="Add editable text using <text> elements (avoid converting text to paths).",
                context={"tag": "text"},
            )
        )
        return issues
    for node in text_elements:
        if contract.require_text_ids and not node.get("id"):
            context = _element_context(node)
            text_value = (node.text or "").strip()
            if text_value:
                context["text"] = text_value
            issues.append(
                ValidationIssue(
                    code=E2009_TEXT_ID_MISSING,
                    message="Text element is missing a stable id.",
                    hint="Give every <text> element an id (e.g., txt_title, txt_axis_x).",
                    context=context,
                )
            )
        tspans = _tspan_elements(node)
        if tspans:
            for tspan in tspans:
                if not tspan.get("id"):
                    issues.append(
                        ValidationIssue(
                            code=E2011_TSPAN_ID_MISSING,
                            message="Multiline text <tspan> is missing a stable id.",
                            hint="Add ids to each <tspan> (e.g., txt_title_line0).",
                            context={
                                "tag": "tspan",
                                "text_id": node.get("id"),
                                **_element_context(tspan),
                            },
                        )
                    )
        elif _text_has_newline(node):
            issues.append(
                ValidationIssue(
                    code=E2012_MULTILINE_TSPAN_MISSING,
                    message="Multiline <text> must use <tspan> elements.",
                    hint="Split lines into <tspan id=\"...\"> elements with dy offsets.",
                    context=_element_context(node),
                )
            )
    return issues


def _check_text_grouping(
    elements: list[ET.Element], parent_map: dict[ET.Element, ET.Element]
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    for node in elements:
        if local_name(node.tag) != "text":
            continue
        if not _has_ancestor_group(node, parent_map, "g_text"):
            warnings.append(
                ValidationIssue(
                    code=W2103_TEXT_OUTSIDE_TEXT_GROUP,
                    message="Text element is outside g_text group.",
                    hint="Place editable text elements under the g_text group.",
                    context=_element_context(node),
                )
            )
    return warnings


def _text_baseline_value(node: ET.Element) -> float | None:
    y_attr = node.get("y")
    if y_attr:
        value = parse_number(y_attr)
        if value is not None:
            return value
    for tspan in _tspan_elements(node):
        tspan_y = tspan.get("y")
        if not tspan_y:
            continue
        value = parse_number(tspan_y)
        if value is not None:
            return value
    return None


def _check_text_anchor(
    elements: list[ET.Element], parent_map: dict[ET.Element, ET.Element]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    allowed = {"start", "middle", "end"}
    text_elements = [node for node in elements if local_name(node.tag) == "text"]
    for node, style in iter_with_style(text_elements):
        value = _resolve_property(node, style, parent_map, "text-anchor").strip().lower()
        if not value:
            issues.append(
                ValidationIssue(
                    code=E2016_TEXT_ANCHOR_INVALID,
                    message="Text element is missing text-anchor.",
                    hint="Set text-anchor to start, middle, or end to stabilize positioning.",
                    context=_element_context(node),
                )
            )
            continue
        if value not in allowed:
            issues.append(
                ValidationIssue(
                    code=E2016_TEXT_ANCHOR_INVALID,
                    message=f"Text-anchor '{value}' is invalid.",
                    hint="Use text-anchor start, middle, or end.",
                    context={"text_anchor": value, **_element_context(node)},
                )
            )
    return issues


def _check_text_baseline_alignment(
    elements: list[ET.Element],
    parent_map: dict[ET.Element, ET.Element],
    thresholds: TextThresholds,
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    text_elements = [node for node in elements if local_name(node.tag) == "text"]
    baselines: list[tuple[ET.Element, float]] = []
    for node in text_elements:
        value = _text_baseline_value(node)
        if value is not None:
            baselines.append((node, value))
    if len(baselines) < 2:
        return warnings
    baselines.sort(key=lambda item: item[1])
    cluster: list[tuple[ET.Element, float]] = [baselines[0]]
    tolerance = thresholds.baseline_tolerance

    def _flush(group: list[tuple[ET.Element, float]]) -> None:
        if len(group) < 2:
            return
        values = [value for _, value in group]
        if max(values) - min(values) <= 1e-3:
            return
        dominant_values = set()
        for node, _ in group:
            style = parse_style(node.get("style", ""))
            dominant = _resolve_property(node, style, parent_map, "dominant-baseline").strip()
            if dominant:
                dominant_values.add(dominant)
        if len(dominant_values) == 1:
            return
        warnings.append(
            ValidationIssue(
                code=W2106_TEXT_BASELINE_NOT_ALIGNED,
                message="Text baselines within a line are not aligned.",
                hint="Snap text y positions or set a consistent dominant-baseline.",
                context=_element_context(group[0][0]),
            )
        )

    for item in baselines[1:]:
        if abs(item[1] - cluster[-1][1]) <= tolerance:
            cluster.append(item)
        else:
            _flush(cluster)
            cluster = [item]
    _flush(cluster)
    return warnings


def _required_text_count(texts_detected: int, thresholds: TextThresholds) -> int:
    if texts_detected <= 0:
        return 0
    ratio_required = texts_detected * thresholds.min_keep_ratio
    missing_allowed = texts_detected - thresholds.max_missing
    required = min(ratio_required, missing_allowed)
    required = max(required, thresholds.min_keep_absolute, 0.0)
    return int(math.ceil(required))


def _check_text_expectations(
    elements: list[ET.Element],
    text_expectations: dict[str, object] | None,
    thresholds: TextThresholds,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not text_expectations:
        return issues
    detected = text_expectations.get("texts_detected")
    if detected is None:
        return issues
    try:
        texts_detected = int(detected)
    except (TypeError, ValueError):
        return issues
    if texts_detected <= 0:
        return issues
    text_elements = [node for node in elements if local_name(node.tag) == "text"]
    text_count = len(text_elements)
    required = _required_text_count(texts_detected, thresholds)
    if text_count < required:
        issues.append(
            ValidationIssue(
                code=E2014_TEXT_COUNT_LOW,
                message="Too few text elements compared to detected text.",
                hint="Preserve detected text using <text> elements (avoid converting to paths).",
                context={
                    "texts_detected": texts_detected,
                    "text_count": text_count,
                    "required_min": required,
                    "tag": "text",
                },
            )
        )
    if text_count == 0:
        path_count = sum(1 for node in elements if local_name(node.tag) == "path")
        issues.append(
            ValidationIssue(
                code=E2017_TEXT_OUTLINE_DETECTED,
                message="Detected text is missing from <text> elements.",
                hint="Preserve detected text using editable <text> elements (avoid outlining text into paths).",
                context={
                    "texts_detected": texts_detected,
                    "path_count": path_count,
                    "outline_path_min": thresholds.outline_path_min,
                    "tag": "text",
                },
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


def _check_text_as_path(
    elements: list[ET.Element], parent_map: dict[ET.Element, ET.Element]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for node in elements:
        if local_name(node.tag) != "path":
            continue
        if _has_ancestor_group(node, parent_map, "g_text") or _looks_like_text_path(node):
            issues.append(
                ValidationIssue(
                    code=E2010_TEXT_AS_PATH,
                    message="Text appears to be converted to <path>.",
                    hint="Keep editable text as <text> elements (do not outline text).",
                    context=_element_context(node),
                )
            )
    return issues


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
        if None in (x1, y1, x2, y2):
            continue
        dx = x2 - x1
        dy = y2 - y1
        if 0 < abs(dy) <= tolerance:
            warnings.append(
                ValidationIssue(
                    code=W2101_LINE_NOT_SNAPPED,
                    message="Line is nearly horizontal but not snapped.",
                    hint="Snap line coordinates so y1 equals y2 exactly.",
                    context={"axis": "horizontal", **_element_context(node)},
                )
            )
        if 0 < abs(dx) <= tolerance:
            warnings.append(
                ValidationIssue(
                    code=W2101_LINE_NOT_SNAPPED,
                    message="Line is nearly vertical but not snapped.",
                    hint="Snap line coordinates so x1 equals x2 exactly.",
                    context={"axis": "vertical", **_element_context(node)},
                )
            )
    return warnings


def _check_dashed_simulation(
    elements: list[ET.Element], thresholds: GeometryThresholds
) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    tolerance = thresholds.snap_tolerance
    max_length = thresholds.dash_segment_length_max
    min_count = thresholds.dash_segment_min_count
    groups: dict[tuple[str, float, str, float], list[ET.Element]] = {}

    for node, style in iter_with_style(elements):
        if local_name(node.tag) != "line":
            continue
        if _has_dasharray(node, style):
            continue
        x1 = parse_number(node.get("x1", ""))
        y1 = parse_number(node.get("y1", ""))
        x2 = parse_number(node.get("x2", ""))
        y2 = parse_number(node.get("y2", ""))
        if None in (x1, y1, x2, y2):
            continue
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) <= tolerance and abs(dx) > tolerance:
            length = abs(dx)
            axis = "h"
            coord = round(y1, 3)
        elif abs(dx) <= tolerance and abs(dy) > tolerance:
            length = abs(dy)
            axis = "v"
            coord = round(x1, 3)
        else:
            continue
        if length > max_length:
            continue
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


def validate_svg(
    svg_path: Path | str,
    contract_path: Path | str,
    thresholds_path: Path | str | None = None,
    expected_png: Path | str | None = None,
    actual_png_path: Path | str | None = None,
    diff_png_path: Path | str | None = None,
    text_expectations: dict[str, object] | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    stats: dict[str, object] = {}

    try:
        contract = load_contract(Path(contract_path))
    except Exception as exc:  # noqa: BLE001
        issues.append(
            ValidationIssue(
                code=E1001_CONFIG_ERROR,
                message=f"Failed to load contract: {exc}",
                hint="Check that the contract YAML is present and valid.",
                context={"path": str(contract_path), "tag": "config"},
            )
        )
        return ValidationReport(status="fail", errors=issues, stats=stats)

    thresholds = None
    if thresholds_path is not None:
        try:
            thresholds = load_thresholds(Path(thresholds_path))
        except Exception as exc:  # noqa: BLE001
            issues.append(
                ValidationIssue(
                    code=E1002_THRESHOLDS_ERROR,
                    message=f"Failed to load thresholds: {exc}",
                    hint="Check that the thresholds YAML is present and valid.",
                    context={"path": str(thresholds_path), "tag": "config"},
                )
            )
            return ValidationReport(status="fail", errors=issues, stats=stats)
    elif expected_png is not None:
        issues.append(
            ValidationIssue(
                code=E1002_THRESHOLDS_ERROR,
                message="Thresholds are required when running visual diff.",
                hint="Provide the validator thresholds YAML.",
                context={"path": str(thresholds_path) if thresholds_path else None, "tag": "config"},
            )
        )
        return ValidationReport(status="fail", errors=issues, stats=stats)

    try:
        root = _parse_svg(Path(svg_path))
    except Exception as exc:  # noqa: BLE001
        issues.append(
            ValidationIssue(
                code=E1000_PARSE_ERROR,
                message=f"Failed to parse SVG: {exc}",
                hint="Ensure the SVG is well-formed XML.",
                context={"path": str(svg_path), "tag": "svg"},
            )
        )
        return ValidationReport(status="fail", errors=issues, stats=stats)

    elements = list(root.iter())
    parent_map = _parent_map(root)
    geometry_thresholds = load_geometry_thresholds(thresholds or {})
    text_thresholds = _load_text_thresholds(thresholds or {})

    issues.extend(_check_forbidden_elements(elements, contract))
    issues.extend(_check_required_groups(elements, contract))
    issues.extend(_check_text_requirements(elements, contract))
    warnings.extend(_check_text_grouping(elements, parent_map))
    issues.extend(_check_font_families(elements, contract, parent_map))
    issues.extend(_check_text_anchor(elements, parent_map))
    issues.extend(_check_text_expectations(elements, text_expectations, text_thresholds))
    issues.extend(_check_text_as_path(elements, parent_map))
    color_issues, color_count = _check_colors(elements, contract)
    issues.extend(color_issues)
    issues.extend(_check_stroke_widths(elements, contract))
    issues.extend(_check_dasharray_required(elements))
    path_issues, max_path_commands = _check_path_complexity(elements, contract)
    issues.extend(path_issues)
    warnings.extend(_check_line_snapping(elements, geometry_thresholds))
    warnings.extend(_check_polyline_snapping(elements, geometry_thresholds))
    warnings.extend(_check_dashed_simulation(elements, geometry_thresholds))
    warnings.extend(_check_polyline_complexity(elements, geometry_thresholds))
    warnings.extend(_check_text_baseline_alignment(elements, parent_map, text_thresholds))

    text_count = sum(1 for node in elements if local_name(node.tag) == "text")
    stats.update(
        {
            "color_count": color_count,
            "max_path_commands": max_path_commands,
            "text_count": text_count,
        }
    )
    if text_expectations and "texts_detected" in text_expectations:
        stats["texts_detected"] = text_expectations.get("texts_detected")

    if expected_png is not None and thresholds is not None:
        try:
            visual_thresholds = load_visual_diff_thresholds(thresholds)
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    code=E1002_THRESHOLDS_ERROR,
                    message=f"Invalid visual_diff thresholds: {exc}",
                    hint="Check visual_diff thresholds in the YAML config.",
                    context={"tag": "config"},
                )
            )
            return ValidationReport(status="fail", errors=issues, stats=stats)
        try:
            if actual_png_path is None:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    actual_png = Path(tmp_dir) / "rendered.png"
                    backend = rasterize_svg_to_png(Path(svg_path), actual_png)
                    metrics = compute_visual_diff(
                        actual_png, Path(expected_png), visual_thresholds.pixel_tolerance
                    )
            else:
                actual_png = Path(actual_png_path)
                actual_png.parent.mkdir(parents=True, exist_ok=True)
                backend = rasterize_svg_to_png(Path(svg_path), actual_png)
                metrics = compute_visual_diff(
                    actual_png, Path(expected_png), visual_thresholds.pixel_tolerance
                )
        except RasterizeError as exc:
            issues.append(
                ValidationIssue(
                    code=E3001_RASTERIZE_FAILED,
                    message=f"Rasterization failed: {exc}",
                    hint="Install resvg or cairosvg and ensure the SVG is valid.",
                    context={"svg": str(svg_path), "tag": "svg"},
                )
            )
        except DiffError as exc:
            issues.append(
                ValidationIssue(
                    code=E3002_DIFF_FAILED,
                    message=f"Visual diff failed: {exc}",
                    hint="Ensure expected.png has the same dimensions as the rasterized SVG.",
                    context={
                        "expected_png": str(expected_png),
                        "actual_png": str(actual_png_path) if actual_png_path else None,
                        "tag": "svg",
                    },
                )
            )
        else:
            stats["visual_diff"] = {
                "backend": backend,
                "rmse": metrics.rmse,
                "bad_pixel_ratio": metrics.bad_pixel_ratio,
                "pixel_tolerance": visual_thresholds.pixel_tolerance,
                "rmse_max": visual_thresholds.rmse_max,
                "bad_pixel_ratio_max": visual_thresholds.bad_pixel_ratio_max,
            }
            if (
                metrics.rmse > visual_thresholds.rmse_max
                or metrics.bad_pixel_ratio > visual_thresholds.bad_pixel_ratio_max
            ):
                if diff_png_path is not None:
                    try:
                        from .visual_diff import write_diff_image

                        write_diff_image(
                            Path(actual_png),
                            Path(expected_png),
                            Path(diff_png_path),
                            visual_thresholds.pixel_tolerance,
                        )
                    except Exception:
                        pass
                issues.append(
                    ValidationIssue(
                        code=E3003_VISUAL_THRESHOLD_EXCEEDED,
                        message="Visual diff metrics exceed thresholds.",
                        hint="Reduce visual differences or adjust thresholds.",
                        context={
                            "rmse": metrics.rmse,
                            "rmse_max": visual_thresholds.rmse_max,
                            "bad_pixel_ratio": metrics.bad_pixel_ratio,
                            "bad_pixel_ratio_max": visual_thresholds.bad_pixel_ratio_max,
                            "tag": "svg",
                        },
                    )
                )

    status = "pass" if not issues else "fail"
    return ValidationReport(status=status, errors=issues, warnings=warnings, stats=stats)
