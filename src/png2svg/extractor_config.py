from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from png2svg.errors import Png2SvgError
from png2svg.extractor_constants import DEFAULT_ADAPTIVE_CONFIG


def _clamp_value(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _odd_int(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise Png2SvgError(
            code="E4012_ADAPTIVE_CONFIG_INVALID",
            message=f"Adaptive config must be a mapping: {path}",
            hint="Ensure the adaptive config YAML is a mapping at the top level.",
        )
    return data


def _load_adaptive_config(path: Path | None = None) -> dict[str, Any]:
    resolved = path or DEFAULT_ADAPTIVE_CONFIG
    if not resolved.exists():
        raise Png2SvgError(
            code="E4011_ADAPTIVE_CONFIG_MISSING",
            message=f"Adaptive config not found: {resolved}",
            hint="Ensure config/extract_adaptive.v1.yaml exists.",
        )
    return _load_yaml(resolved)


def _effective_config(config: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    base_long_edge = int(config.get("base_long_edge", 800))
    scale_cfg = config.get("scale", {}) if isinstance(config.get("scale"), dict) else {}
    scale = max(width, height) / float(base_long_edge or 1)
    scale = _clamp_value(scale, float(scale_cfg.get("min", 0.6)), float(scale_cfg.get("max", 1.8)))

    bin_cfg = config.get("binarization", {}) if isinstance(config.get("binarization"), dict) else {}
    block_base = int(bin_cfg.get("block_size", 31))
    block_size = _odd_int(max(int(round(block_base * scale)), 3))
    block_size = int(
        _clamp_value(
            block_size,
            float(bin_cfg.get("block_size_min", 15)),
            float(bin_cfg.get("block_size_max", 61)),
        )
    )
    if block_size % 2 == 0:
        block_size += 1
    binarization = {
        "enabled": bool(bin_cfg.get("enabled", True)),
        "block_size": block_size,
        "c": int(round(float(bin_cfg.get("c", 10)))),
        "alpha_threshold": int(bin_cfg.get("alpha_threshold", 10)),
        "luminance_threshold": int(bin_cfg.get("luminance_threshold", 245)),
    }

    lines_cfg = config.get("lines", {}) if isinstance(config.get("lines"), dict) else {}
    long_line_ratio = float(lines_cfg.get("long_line_ratio", 0.5))
    long_line_min_len = max(int(round(height * long_line_ratio * scale)), 1)

    dash_cfg = config.get("dashes", {}) if isinstance(config.get("dashes"), dict) else {}
    dash_min_len = max(int(round(height * float(dash_cfg.get("min_len_ratio", 0.03)) * scale)), 2)
    dash_max_len = max(
        int(round(height * float(dash_cfg.get("max_len_ratio", 0.12)) * scale)),
        dash_min_len + 1,
    )
    dash_min_count = int(dash_cfg.get("min_count", 6))

    text_cfg = config.get("text_boxes", {}) if isinstance(config.get("text_boxes"), dict) else {}
    min_area_ratio = float(text_cfg.get("min_area_ratio", 0.00005))
    min_area = max(int(round(width * height * min_area_ratio * scale)), 8)
    max_area_ratio = float(text_cfg.get("max_area_ratio", 0.02))
    max_area = max(int(round(width * height * max_area_ratio * scale)), min_area + 1)
    min_size = max(int(round(text_cfg.get("min_size", 3) * scale)), 2)
    text_boxes = {
        "min_area": min_area,
        "max_area": max_area,
        "min_size": min_size,
        "max_width_ratio": float(text_cfg.get("max_width_ratio", 0.6)),
        "max_height_ratio": float(text_cfg.get("max_height_ratio", 0.4)),
    }

    curve_cfg = config.get("curves", {}) if isinstance(config.get("curves"), dict) else {}
    curves = {
        "saturation_min": float(curve_cfg.get("saturation_min", 0.25)),
        "value_min": float(curve_cfg.get("value_min", 0.2)),
        "hue_tolerance_deg": float(curve_cfg.get("hue_tolerance_deg", 22.0)),
        "smooth_window": int(curve_cfg.get("smooth_window", 7)),
        "sample_spacing_px": int(curve_cfg.get("sample_spacing_px", 120)),
        "min_segments": int(curve_cfg.get("min_segments", 4)),
        "max_segments": int(curve_cfg.get("max_segments", 8)),
    }

    return {
        "scale": scale,
        "binarization": binarization,
        "text_boxes": text_boxes,
        "lines": {
            "long_line_min_len_px": long_line_min_len,
        },
        "dashes": {
            "min_len": dash_min_len,
            "max_len": dash_max_len,
            "min_count": dash_min_count,
            "min_span_ratio": float(dash_cfg.get("min_span_ratio", 0.6)),
            "min_coverage": float(dash_cfg.get("min_coverage", 0.02)),
            "max_coverage": float(dash_cfg.get("max_coverage", 0.35)),
            "min_gap_ratio": float(dash_cfg.get("min_gap_ratio", 0.5)),
            "max_lines_per_panel": int(dash_cfg.get("max_lines_per_panel", 2)),
        },
        "ocr": config.get("ocr", {}),
        "text_layout": config.get("text_layout", {}),
        "curves": curves,
    }
