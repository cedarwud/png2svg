#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(add_completion=False, help="Check regression dataset sanity.")

DEFAULT_SANITY_CONFIG = REPO_ROOT / "config" / "dataset_sanity.v1.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _resolve_manifest(dataset: Path) -> tuple[Path, Path | None]:
    if dataset.is_file():
        return dataset, dataset.parent
    manifest = dataset / "manifest.yaml"
    if manifest.exists():
        return manifest, dataset
    return dataset, None


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("manifest.yaml must contain a mapping.")
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("manifest.yaml must contain a list under 'cases'.")
    return data


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


def _metrics(image: Image.Image, edge_threshold: float) -> dict[str, float]:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    ink = (alpha > 10) & (luminance < 245)
    ink_ratio = float(np.mean(ink))
    variance = float(np.var(luminance))
    dx = np.abs(np.diff(luminance, axis=1))
    dy = np.abs(np.diff(luminance, axis=0))
    edges = np.zeros(luminance.shape, dtype=bool)
    edges[:, 1:] |= dx > edge_threshold
    edges[1:, :] |= dy > edge_threshold
    edge_ratio = float(np.mean(edges))
    return {
        "ink_ratio": ink_ratio,
        "variance": variance,
        "edge_ratio": edge_ratio,
    }


@app.command()
def main(
    dataset: Path = typer.Argument(
        REPO_ROOT / "datasets" / "regression_v0",
        exists=True,
        readable=True,
        help="Dataset directory or manifest.yaml path.",
    ),
    config: Path = typer.Option(
        DEFAULT_SANITY_CONFIG,
        "--config",
        dir_okay=False,
        help="Path to dataset sanity config YAML.",
    ),
) -> None:
    manifest_path, base_dir = _resolve_manifest(dataset)
    if base_dir is None:
        case_dir = manifest_path
        if not case_dir.is_dir():
            typer.echo("ERROR E1700_MANIFEST_MISSING: manifest.yaml not found.", err=True)
            typer.echo("HINT: Provide a dataset directory with manifest.yaml.", err=True)
            raise typer.Exit(code=1)
        entries = [{"dir": case_dir.name, "id": case_dir.name}]
        base_dir = case_dir.parent
    else:
        try:
            manifest = _load_manifest(manifest_path)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"ERROR E1701_MANIFEST_INVALID: {exc}", err=True)
            typer.echo("HINT: Ensure manifest.yaml has a cases list.", err=True)
            raise typer.Exit(code=1)
        entries = manifest["cases"]

    cfg = _load_yaml(config)
    variance_min = float(cfg.get("variance_min", 2.0))
    ink_ratio_min = float(cfg.get("ink_ratio_min", 0.002))
    ink_ratio_max = float(cfg.get("ink_ratio_max", 0.98))
    edge_ratio_min = float(cfg.get("edge_ratio_min", 0.001))
    edge_threshold = float(cfg.get("edge_threshold", 8.0))

    failures: list[str] = []
    for entry in entries:
        case_dir = _case_entry_dir(base_dir, entry)
        case_id = _case_entry_id(entry, case_dir)
        for name in ("input.png", "input_hard.png"):
            path = case_dir / name
            if not path.exists():
                failures.append(f"{case_id}:{name} missing")
                continue
            with Image.open(path) as image:
                stats = _metrics(image, edge_threshold)
            if (
                stats["variance"] < variance_min
                or stats["ink_ratio"] < ink_ratio_min
                or stats["ink_ratio"] > ink_ratio_max
                or stats["edge_ratio"] < edge_ratio_min
            ):
                failures.append(
                    f"{case_id}:{name} variance={stats['variance']:.2f} "
                    f"ink={stats['ink_ratio']:.4f} edge={stats['edge_ratio']:.4f}"
                )

    if failures:
        typer.echo("Dataset sanity check failed:", err=True)
        for item in failures:
            typer.echo(f" - {item}", err=True)
        raise typer.Exit(code=1)
    typer.echo("Dataset sanity check OK.")


if __name__ == "__main__":
    app(prog_name="check_dataset_sanity")
