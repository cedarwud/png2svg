from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("manifest.yaml must contain a mapping.")
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("manifest.yaml must contain a list under 'cases'.")
    return data


def _load_real_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("real manifest must contain a mapping.")
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("real manifest must contain a list under 'cases'.")
    return data


def _resolve_manifest(dataset: Path) -> tuple[Path, Path | None]:
    if dataset.is_file():
        return dataset, dataset.parent
    manifest = dataset / "manifest.yaml"
    if manifest.exists():
        return manifest, dataset
    return dataset, None


def _case_entry_dir(base_dir: Path, entry: Any) -> Path:
    if isinstance(entry, str):
        return base_dir / entry
    if isinstance(entry, dict) and "dir" in entry:
        return base_dir / str(entry["dir"])
    raise ValueError("Each case entry must be a string or have a 'dir' field.")


def _case_entry_id(entry: Any, case_dir: Path) -> str:
    if isinstance(entry, dict) and "id" in entry:
        return str(entry["id"])
    return case_dir.name


def _case_gate_overrides(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    gates = entry.get("gates")
    if not isinstance(gates, dict):
        return {}
    overrides: dict[str, Any] = {}
    if "rmse_max" in gates:
        overrides["rmse_max"] = gates.get("rmse_max")
    if "bad_pixel_ratio_max" in gates:
        overrides["bad_pixel_ratio_max"] = gates.get("bad_pixel_ratio_max")
    if "pixel_tolerance" in gates:
        overrides["pixel_tolerance"] = gates.get("pixel_tolerance")
    if "quality_gate" in gates:
        overrides["quality_gate"] = gates.get("quality_gate")
    return overrides


def _real_expected_templates(entry: dict[str, Any]) -> list[str]:
    if "expected_template" in entry:
        return [str(entry["expected_template"])]
    if "expected_templates" in entry and isinstance(entry["expected_templates"], list):
        return [str(item) for item in entry["expected_templates"] if str(item).strip()]
    return []


def _real_gates(entry: dict[str, Any]) -> dict[str, Any]:
    gates = entry.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("real manifest entry missing gates.")
    return gates


def _real_allow_force_template(entry: dict[str, Any]) -> bool:
    return bool(entry.get("allow_force_template", False))
