#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from validators.visual_diff import RasterizeError, rasterize_svg_to_png  # noqa: E402

app = typer.Typer(add_completion=False, help="Rebuild PNG assets for regression cases.")


def _load_params(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid params.json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("params.json must contain a JSON object.")
    return data


def _extract_canvas(params: dict[str, Any]) -> tuple[int, int] | None:
    canvas = params.get("canvas")
    if not isinstance(canvas, dict):
        return None
    width = canvas.get("width")
    height = canvas.get("height")
    if width is None or height is None:
        return None
    try:
        width_int = int(width)
        height_int = int(height)
    except (TypeError, ValueError):
        return None
    if width_int <= 0 or height_int <= 0:
        return None
    return width_int, height_int


def _write_blank_png(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.save(path)


def _rebuild_case(case_dir: Path) -> list[str]:
    errors: list[str] = []
    params_path = case_dir / "params.json"
    if not params_path.exists():
        errors.append(f"{case_dir}: params.json missing")
        return errors
    try:
        params = _load_params(params_path)
    except ValueError as exc:
        errors.append(f"{case_dir}: {exc}")
        return errors
    canvas = _extract_canvas(params)
    if canvas:
        _write_blank_png(case_dir / "input.png", canvas)
    expected_svg = case_dir / "expected.svg"
    if expected_svg.exists():
        expected_png = case_dir / "expected.png"
        try:
            rasterize_svg_to_png(expected_svg, expected_png)
        except RasterizeError as exc:
            errors.append(f"{case_dir}: expected.svg rasterize failed: {exc}")
    return errors


@app.command()
def main(
    cases_root: Path = typer.Option(
        REPO_ROOT / "datasets" / "regression_v0" / "cases",
        "--cases-root",
        help="Root directory containing regression cases.",
    ),
    samples_dir: Path = typer.Option(
        REPO_ROOT / "samples",
        "--samples-dir",
        help="Samples directory to rebuild input.png.",
    ),
    sample_size: int = typer.Option(
        32,
        "--sample-size",
        min=1,
        help="Square size for samples/input.png.",
    ),
) -> None:
    """Rebuild input.png and expected.png assets for regression cases."""
    errors: list[str] = []

    samples_input = samples_dir / "input.png"
    _write_blank_png(samples_input, (sample_size, sample_size))

    case_dirs = sorted({path.parent for path in cases_root.rglob("params.json")})
    for case_dir in case_dirs:
        errors.extend(_rebuild_case(case_dir))

    if errors:
        typer.echo("ERROR E1500_REBUILD_FAILED: one or more assets failed to rebuild.", err=True)
        typer.echo("HINT: Fix the listed cases and re-run rebuild_png_assets.py.", err=True)
        for item in errors:
            typer.echo(item, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK: rebuilt {len(case_dirs)} case(s) and {samples_input}.")


if __name__ == "__main__":
    app(prog_name="rebuild_png_assets")
