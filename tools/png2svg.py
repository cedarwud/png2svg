#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from png2svg import Png2SvgError, render_svg  # noqa: E402

app = typer.Typer(add_completion=False, help="Render SVG from PNG and params.")


@app.command()
def main(
    input_png: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input PNG to render.",
    ),
    params: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Template params.json file.",
    ),
    output_svg: Path = typer.Argument(
        ...,
        dir_okay=False,
        help="Output SVG path.",
    ),
) -> None:
    """Render a template-based SVG."""
    try:
        render_svg(input_png, params, output_svg)
    except Png2SvgError as exc:
        typer.echo(f"ERROR {exc.code}: {exc.message}", err=True)
        typer.echo(f"HINT: {exc.hint}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"ERROR E1199_UNEXPECTED: {exc}", err=True)
        typer.echo("HINT: Check input paths and params.json content.", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app(prog_name="png2svg")
