from __future__ import annotations

import tempfile
from pathlib import Path

from .config import (
    load_contract,
    load_geometry_thresholds,
    load_thresholds,
    load_visual_diff_thresholds,
)
from .geometry_rules import (
    _check_colors,
    _check_dasharray_required,
    _check_dashed_simulation,
    _check_forbidden_elements,
    _check_line_snapping,
    _check_path_complexity,
    _check_polyline_complexity,
    _check_polyline_snapping,
    _check_required_groups,
    _check_stroke_widths,
)
from .report import ValidationIssue, ValidationReport
from .svg_checks import local_name
from .text_rules import (
    _check_font_families,
    _check_text_anchor,
    _check_text_as_path,
    _check_text_baseline_alignment,
    _check_text_expectations,
    _check_text_grouping,
    _check_text_requirements,
    _load_text_thresholds,
)
from .validate_constants import (
    E1000_PARSE_ERROR,
    E1001_CONFIG_ERROR,
    E1002_THRESHOLDS_ERROR,
    E2001_FORBIDDEN_ELEMENT,
    E2002_FORBIDDEN_PREFIX,
    E2003_MISSING_GROUP,
    E2004_BAD_FONT_FAMILY,
    E2005_TOO_MANY_COLORS,
    E2006_BAD_STROKE_WIDTH,
    E2007_PATH_TOO_COMPLEX,
    E2008_TEXT_MISSING,
    E2009_TEXT_ID_MISSING,
    E2010_TEXT_AS_PATH,
    E2011_TSPAN_ID_MISSING,
    E2012_MULTILINE_TSPAN_MISSING,
    E2013_DASHED_MISSING_DASHARRAY,
    E2014_TEXT_COUNT_LOW,
    E2015_TEXT_FALLBACK_MISSING,
    E2016_TEXT_ANCHOR_INVALID,
    E2017_TEXT_OUTLINE_DETECTED,
    E3001_RASTERIZE_FAILED,
    E3002_DIFF_FAILED,
    E3003_VISUAL_THRESHOLD_EXCEEDED,
    W2101_LINE_NOT_SNAPPED,
    W2102_DASHED_SIMULATED,
    W2103_TEXT_OUTSIDE_TEXT_GROUP,
    W2104_POLYLINE_TOO_COMPLEX,
    W2105_POLYLINE_NOT_SNAPPED,
    W2106_TEXT_BASELINE_NOT_ALIGNED,
)
from .visual_diff import DiffError, RasterizeError, compute_visual_diff, rasterize_svg_to_png
from .xml_utils import _parent_map, _parse_svg


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


__all__ = [
    "validate_svg",
    "E1000_PARSE_ERROR",
    "E1001_CONFIG_ERROR",
    "E1002_THRESHOLDS_ERROR",
    "E2001_FORBIDDEN_ELEMENT",
    "E2002_FORBIDDEN_PREFIX",
    "E2003_MISSING_GROUP",
    "E2004_BAD_FONT_FAMILY",
    "E2005_TOO_MANY_COLORS",
    "E2006_BAD_STROKE_WIDTH",
    "E2007_PATH_TOO_COMPLEX",
    "E2008_TEXT_MISSING",
    "E2009_TEXT_ID_MISSING",
    "E2010_TEXT_AS_PATH",
    "E2011_TSPAN_ID_MISSING",
    "E2012_MULTILINE_TSPAN_MISSING",
    "E2013_DASHED_MISSING_DASHARRAY",
    "E2014_TEXT_COUNT_LOW",
    "E2015_TEXT_FALLBACK_MISSING",
    "E2016_TEXT_ANCHOR_INVALID",
    "E2017_TEXT_OUTLINE_DETECTED",
    "E3001_RASTERIZE_FAILED",
    "E3002_DIFF_FAILED",
    "E3003_VISUAL_THRESHOLD_EXCEEDED",
    "W2101_LINE_NOT_SNAPPED",
    "W2102_DASHED_SIMULATED",
    "W2103_TEXT_OUTSIDE_TEXT_GROUP",
    "W2104_POLYLINE_TOO_COMPLEX",
    "W2105_POLYLINE_NOT_SNAPPED",
    "W2106_TEXT_BASELINE_NOT_ALIGNED",
]
