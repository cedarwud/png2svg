#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.png_utils import PNG_MAGIC, has_png_magic  # noqa: E402

app = typer.Typer(add_completion=False, help="Verify PNG magic headers.")


def _collect_pngs(paths: list[Path]) -> list[Path]:
    pngs: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".png":
            pngs.append(path)
        elif path.is_dir():
            pngs.extend(sorted(path.rglob("*.png")))
    return pngs


@app.command()
def main(
    targets: list[Path] | None = typer.Argument(
        None,
        help="Optional paths to scan (defaults to samples/ and datasets/).",
    )
) -> None:
    """Scan PNG files and report any with invalid headers."""
    if not targets:
        targets = [REPO_ROOT / "samples", REPO_ROOT / "datasets"]
    png_files = _collect_pngs([path.resolve() for path in targets])
    invalid = [path for path in png_files if not has_png_magic(path)]
    if invalid:
        typer.echo(f"ERROR E1400_PNG_MAGIC_INVALID: {len(invalid)} invalid PNG(s).", err=True)
        typer.echo(f"HINT: Regenerate files with a valid {PNG_MAGIC!r} header.", err=True)
        for path in invalid:
            typer.echo(str(path), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK: {len(png_files)} PNG(s) have valid headers.")


if __name__ == "__main__":
    app(prog_name="check_png_integrity")
