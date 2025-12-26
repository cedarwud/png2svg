from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _gate_section_overrides(thresholds_path: Path, section_name: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(thresholds_path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    section = data.get(section_name)
    if not isinstance(section, dict):
        return {}
    overrides: dict[str, Any] = {}
    if "rmse_max" in section:
        overrides["rmse_max"] = section.get("rmse_max")
    if "bad_pixel_ratio_max" in section:
        overrides["bad_pixel_ratio_max"] = section.get("bad_pixel_ratio_max")
    if "pixel_tolerance" in section:
        overrides["pixel_tolerance"] = section.get("pixel_tolerance")
    return overrides


def _variant_gate_overrides(thresholds_path: Path, input_variant: str) -> dict[str, Any]:
    if input_variant != "hard":
        return {}
    return _gate_section_overrides(thresholds_path, "quality_gate_hard")


def _tier_gate_overrides(thresholds_path: Path, tier: str) -> dict[str, Any]:
    if tier != "hard":
        return {}
    return _gate_section_overrides(thresholds_path, "quality_gate_hard")
