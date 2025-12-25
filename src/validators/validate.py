from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import (
    FigureContract,
    load_contract,
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
E3001_RASTERIZE_FAILED = "E3001_RASTERIZE_FAILED"
E3002_DIFF_FAILED = "E3002_DIFF_FAILED"
E3003_VISUAL_THRESHOLD_EXCEEDED = "E3003_VISUAL_THRESHOLD_EXCEEDED"


def _font_family_ok(value: str, allowed: list[str]) -> bool:
    if not value:
        return False
    allowed_set = {item.strip().strip("'\"").lower() for item in allowed}
    candidates = [
        item.strip().strip("'\"").lower() for item in value.split(",") if item.strip()
    ]
    return any(candidate in allowed_set for candidate in candidates)


def _parse_svg(svg_path: Path) -> ET.Element:
    tree = ET.parse(svg_path)
    return tree.getroot()


def _element_context(node: ET.Element) -> dict[str, str]:
    context: dict[str, str] = {"tag": local_name(node.tag)}
    node_id = node.get("id")
    if node_id:
        context["id"] = node_id
    return context


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
                    hint="Remove the forbidden SVG elements from the output.",
                    context={"element": name, **_element_context(node)},
                )
            )
        lowered = name.lower()
        if any(lowered.startswith(prefix) for prefix in prefix_set):
            issues.append(
                ValidationIssue(
                    code=E2002_FORBIDDEN_PREFIX,
                    message=f"Forbidden element prefix in <{name}>.",
                    hint="Remove fe* filter primitive elements; filters are not permitted.",
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
                    hint=f"Add a <g> element with id '{required}'.",
                    context={"group": required},
                )
            )
    return issues


def _check_font_families(
    elements: list[ET.Element], contract: FigureContract
) -> list[ValidationIssue]:
    allowed = contract.allowed_font_families
    if not allowed:
        return []
    text_elements = [node for node in elements if local_name(node.tag) == "text"]
    if not text_elements:
        return []
    issues: list[ValidationIssue] = []
    for node, style in iter_with_style(text_elements):
        values = extract_property_values(node, style, "font-family")
        if not values:
            issues.append(
                ValidationIssue(
                    code=E2004_BAD_FONT_FAMILY,
                    message="Missing font-family on text element.",
                    hint=f"Set font-family to include one of: {', '.join(allowed)}.",
                    context=_element_context(node),
                )
            )
            continue
        value = values[0]
        if not _font_family_ok(value, allowed):
            issues.append(
                ValidationIssue(
                    code=E2004_BAD_FONT_FAMILY,
                    message=f"Font-family '{value}' is not allowed.",
                    hint=f"Use one of: {', '.join(allowed)}.",
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
                hint="Add at least one <text> element to the SVG.",
            )
        )
        return issues
    if contract.require_text_ids:
        for node in text_elements:
            if not node.get("id"):
                context = _element_context(node)
                text_value = (node.text or "").strip()
                if text_value:
                    context["text"] = text_value
                issues.append(
                    ValidationIssue(
                        code=E2009_TEXT_ID_MISSING,
                        message="Text element is missing a stable id.",
                        hint="Assign an id attribute to each editable text element.",
                        context=context,
                    )
                )
    return issues


def _check_colors(
    elements: list[ET.Element], contract: FigureContract
) -> tuple[list[ValidationIssue], int]:
    max_colors = contract.max_colors
    colors: set[object] = set()
    for node, style in iter_with_style(elements):
        for prop in ("fill", "stroke"):
            for value in extract_property_values(node, style, prop):
                parsed = parse_color(value)
                if parsed is not None:
                    colors.add(parsed)
    color_count = len(colors)
    if max_colors is None or color_count <= max_colors:
        return [], color_count
    issue = ValidationIssue(
        code=E2005_TOO_MANY_COLORS,
        message=f"Color count {color_count} exceeds max_colors {max_colors}.",
        hint="Reduce the palette by consolidating fill/stroke colors.",
        context={
            "color_count": color_count,
            "max_colors": max_colors,
            "colors": [str(color) for color in sorted(colors, key=str)[:12]],
        },
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
                        hint="Use numeric stroke widths matching the allowed set.",
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
                    hint="Use allowed stroke widths within tolerance.",
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
                hint="Simplify the path to use fewer commands.",
                context={
                    "command_count": command_count,
                    "max_commands": max_commands,
                    **_element_context(node),
                },
            )
        )
    return issues, max_seen


def validate_svg(
    svg_path: Path | str,
    contract_path: Path | str,
    thresholds_path: Path | str | None = None,
    expected_png: Path | str | None = None,
    actual_png_path: Path | str | None = None,
    diff_png_path: Path | str | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    stats: dict[str, object] = {}

    try:
        contract = load_contract(Path(contract_path))
    except Exception as exc:  # noqa: BLE001
        issues.append(
            ValidationIssue(
                code=E1001_CONFIG_ERROR,
                message=f"Failed to load contract: {exc}",
                hint="Check that the contract YAML is present and valid.",
                context={"path": str(contract_path)},
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
                    context={"path": str(thresholds_path)},
                )
            )
            return ValidationReport(status="fail", errors=issues, stats=stats)
    elif expected_png is not None:
        issues.append(
            ValidationIssue(
                code=E1002_THRESHOLDS_ERROR,
                message="Thresholds are required when running visual diff.",
                hint="Provide the validator thresholds YAML.",
                context={"path": str(thresholds_path) if thresholds_path else None},
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
                context={"path": str(svg_path)},
            )
        )
        return ValidationReport(status="fail", errors=issues, stats=stats)

    elements = list(root.iter())

    issues.extend(_check_forbidden_elements(elements, contract))
    issues.extend(_check_required_groups(elements, contract))
    issues.extend(_check_text_requirements(elements, contract))
    issues.extend(_check_font_families(elements, contract))
    color_issues, color_count = _check_colors(elements, contract)
    issues.extend(color_issues)
    issues.extend(_check_stroke_widths(elements, contract))
    path_issues, max_path_commands = _check_path_complexity(elements, contract)
    issues.extend(path_issues)

    stats.update(
        {
            "color_count": color_count,
            "max_path_commands": max_path_commands,
        }
    )

    if expected_png is not None and thresholds is not None:
        try:
            visual_thresholds = load_visual_diff_thresholds(thresholds)
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    code=E1002_THRESHOLDS_ERROR,
                    message=f"Invalid visual_diff thresholds: {exc}",
                    hint="Check visual_diff thresholds in the YAML config.",
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
                    context={"svg": str(svg_path)},
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
                        },
                    )
                )

    status = "pass" if not issues else "fail"
    return ValidationReport(status=status, errors=issues, stats=stats)
