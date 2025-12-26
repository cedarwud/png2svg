#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from png2svg import render_svg  # noqa: E402
from validators.visual_diff import rasterize_svg_to_png  # noqa: E402


app = typer.Typer(add_completion=False, help="Build hard regression case assets.")

DEFAULT_DEGRADE_CONFIG = REPO_ROOT / "config" / "case_input_degrade.v1.yaml"


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


def _stable_seed(case_id: str) -> int:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _rand_range(rng: np.random.Generator, config: dict[str, Any], key: str) -> float:
    section = config.get(key, {})
    if not isinstance(section, dict):
        return 0.0
    min_val = float(section.get("min", 0.0))
    max_val = float(section.get("max", 0.0))
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    return float(rng.uniform(min_val, max_val))


def _scale_to_canvas(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), resample=Image.BICUBIC)
    if new_w >= width or new_h >= height:
        left = max(int((new_w - width) / 2), 0)
        top = max(int((new_h - height) / 2), 0)
        return resized.crop((left, top, left + width, top + height))
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    left = int((width - new_w) / 2)
    top = int((height - new_h) / 2)
    canvas.paste(resized, (left, top))
    return canvas


def _apply_gamma_contrast(rgb: np.ndarray, gamma: float, contrast: float) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = arr ** max(gamma, 0.001)
    arr = (arr - 0.5) * contrast + 0.5
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _apply_noise(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return rgb
    noise = rng.normal(0.0, sigma, rgb.shape).astype(np.float32)
    arr = rgb.astype(np.float32) + noise
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _degrade_image(image: Image.Image, rng: np.random.Generator, config: dict[str, Any]) -> Image.Image:
    img = image.convert("RGBA")
    angle = _rand_range(rng, config, "rotation_deg")
    scale = _rand_range(rng, config, "scale")
    blur_radius = _rand_range(rng, config, "blur_radius")
    noise_sigma = _rand_range(rng, config, "noise_sigma")
    gamma = _rand_range(rng, config, "gamma") or 1.0
    contrast = _rand_range(rng, config, "contrast") or 1.0

    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255, 0))
    img = _scale_to_canvas(img, scale)
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.asarray(img, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4]
    rgb = _apply_gamma_contrast(rgb, gamma, contrast)
    rgb = _apply_noise(rgb, noise_sigma, rng)
    out = np.concatenate([rgb, alpha], axis=2)
    return Image.fromarray(out, mode="RGBA")


def _render_expected_svg(params_path: Path, svg_path: Path) -> None:
    params = json.loads(params_path.read_text())
    canvas = params.get("canvas")
    if not isinstance(canvas, dict):
        raise ValueError("params.json must include canvas.width/height for hard cases.")
    width = canvas.get("width")
    height = canvas.get("height")
    if width is None or height is None:
        raise ValueError("canvas.width and canvas.height are required for hard cases.")
    width_int = int(width)
    height_int = int(height)
    if width_int <= 0 or height_int <= 0:
        raise ValueError("canvas width/height must be positive.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_png = Path(handle.name)
    try:
        Image.new("RGBA", (width_int, height_int), (255, 255, 255, 0)).save(temp_png)
        render_svg(temp_png, params_path, svg_path)
    finally:
        temp_png.unlink(missing_ok=True)


def _print_case(case_id: str, outputs: list[Path], seed: int, backend: str | None) -> None:
    files = ",".join(path.name for path in outputs)
    backend_note = f" backend={backend}" if backend else ""
    typer.echo(f"{case_id}: {files} seed={seed}{backend_note}")


@app.command()
def main(
    dataset: Path = typer.Argument(
        REPO_ROOT / "datasets" / "regression_hard_v1",
        exists=True,
        readable=True,
        help="Hard dataset directory or manifest.yaml path.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing expected/input assets.",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Only rebuild a specific case id.",
    ),
    degrade_config: Path = typer.Option(
        DEFAULT_DEGRADE_CONFIG,
        "--degrade-config",
        dir_okay=False,
        help="Path to degradation config YAML.",
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

    degrade_cfg = _load_yaml(degrade_config)

    for entry in entries:
        case_dir = _case_entry_dir(base_dir, entry)
        case_id = _case_entry_id(entry, case_dir)
        if only and case_id != only:
            continue
        params_path = case_dir / "params.json"
        if not params_path.exists():
            typer.echo(f"{case_id}: ERROR missing params.json", err=True)
            raise typer.Exit(code=1)

        expected_svg = case_dir / "expected.svg"
        expected_png = case_dir / "expected.png"
        input_png = case_dir / "input.png"

        seed = _stable_seed(case_id)
        rng = np.random.default_rng(seed)
        outputs: list[Path] = []
        backend_used: str | None = None

        if overwrite or not expected_svg.exists():
            _render_expected_svg(params_path, expected_svg)
        outputs.append(expected_svg)

        if overwrite or not expected_png.exists():
            backend_used = rasterize_svg_to_png(expected_svg, expected_png)
        outputs.append(expected_png)

        if overwrite or not input_png.exists():
            with Image.open(expected_png) as base_image:
                degraded = _degrade_image(base_image, rng, degrade_cfg)
                degraded.save(input_png)
        outputs.append(input_png)

        _print_case(case_id, outputs, seed, backend_used)


if __name__ == "__main__":
    app(prog_name="build_hard_case_assets")
