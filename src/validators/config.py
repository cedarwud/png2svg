from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FigureContract:
    forbid_elements: list[str]
    forbid_element_prefixes: list[str]
    required_groups: list[str]
    require_text_elements: bool
    require_text_ids: bool
    allowed_font_families: list[str]
    max_colors: int | None
    allowed_stroke_widths: list[float]
    stroke_width_tolerance: float
    max_path_commands: int | None


@dataclass(frozen=True)
class VisualDiffThresholds:
    pixel_tolerance: int
    rmse_max: float
    bad_pixel_ratio_max: float


@dataclass(frozen=True)
class GeometryThresholds:
    snap_tolerance: float
    dash_segment_length_max: float
    dash_segment_min_count: int


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of YAML: {path}")
    return data


def load_contract(path: Path) -> FigureContract:
    data = _load_yaml(path)
    svg = data.get("svg", {}) or {}
    text = data.get("text", {}) or {}
    typography = data.get("typography", {}) or {}
    style = data.get("style", {}) or {}
    complexity = data.get("complexity", {}) or {}

    forbid_elements = list(svg.get("forbid_elements", []) or [])
    forbid_element_prefixes = list(svg.get("forbid_element_prefixes", []) or [])
    required_groups = list(svg.get("required_groups", []) or [])
    require_text_elements = bool(text.get("require_text_elements", False))
    require_text_ids = bool(text.get("require_text_ids", False))
    allowed_font_families = list(typography.get("allowed_font_families_any_of", []) or [])

    max_colors_raw = style.get("max_colors")
    max_colors = int(max_colors_raw) if max_colors_raw is not None else None

    allowed_stroke_widths = [float(x) for x in (style.get("allowed_stroke_widths", []) or [])]
    stroke_width_tolerance = float(style.get("stroke_width_tolerance", 0.0))

    max_path_commands_raw = complexity.get("max_path_commands")
    max_path_commands = (
        int(max_path_commands_raw) if max_path_commands_raw is not None else None
    )

    return FigureContract(
        forbid_elements=forbid_elements,
        forbid_element_prefixes=forbid_element_prefixes,
        required_groups=required_groups,
        require_text_elements=require_text_elements,
        require_text_ids=require_text_ids,
        allowed_font_families=allowed_font_families,
        max_colors=max_colors,
        allowed_stroke_widths=allowed_stroke_widths,
        stroke_width_tolerance=stroke_width_tolerance,
        max_path_commands=max_path_commands,
    )


def load_thresholds(path: Path) -> dict[str, Any]:
    return _load_yaml(path)


def load_visual_diff_thresholds(data: dict[str, Any]) -> VisualDiffThresholds:
    visual = data.get("visual_diff", {}) or {}
    try:
        pixel_tolerance = int(visual["pixel_tolerance"])
        rmse_max = float(visual["rmse_max"])
        bad_pixel_ratio_max = float(visual["bad_pixel_ratio_max"])
    except KeyError as exc:
        raise ValueError(f"Missing visual_diff threshold: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid visual_diff threshold values: {exc}") from exc
    return VisualDiffThresholds(
        pixel_tolerance=pixel_tolerance,
        rmse_max=rmse_max,
        bad_pixel_ratio_max=bad_pixel_ratio_max,
    )


def load_geometry_thresholds(data: dict[str, Any]) -> GeometryThresholds:
    geometry = data.get("geometry", {}) or {}
    snap_tolerance = float(geometry.get("snap_tolerance", 0.5))
    dash_segment_length_max = float(geometry.get("dash_segment_length_max", 10.0))
    dash_segment_min_count = int(geometry.get("dash_segment_min_count", 6))
    return GeometryThresholds(
        snap_tolerance=snap_tolerance,
        dash_segment_length_max=dash_segment_length_max,
        dash_segment_min_count=dash_segment_min_count,
    )
