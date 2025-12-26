from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .config import FigureContract
from .report import ValidationIssue
from .svg_checks import (
    extract_property_values,
    iter_with_style,
    local_name,
    parse_number,
    parse_style,
)
from .validate_constants import (
    E2004_BAD_FONT_FAMILY,
    E2008_TEXT_MISSING,
    E2009_TEXT_ID_MISSING,
    E2010_TEXT_AS_PATH,
    E2011_TSPAN_ID_MISSING,
    E2012_MULTILINE_TSPAN_MISSING,
    E2014_TEXT_COUNT_LOW,
    E2015_TEXT_FALLBACK_MISSING,
    E2016_TEXT_ANCHOR_INVALID,
    E2017_TEXT_OUTLINE_DETECTED,
    W2103_TEXT_OUTSIDE_TEXT_GROUP,
    W2106_TEXT_BASELINE_NOT_ALIGNED,
)
from .xml_utils import _element_context


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
