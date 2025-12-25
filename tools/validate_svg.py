#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from validators import validate_svg  # noqa: E402

app = typer.Typer(add_completion=False, help="Validate SVG against contract and thresholds.")


@app.command()
def main(
    input_svg: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the SVG to validate.",
    ),
    contract: Path = typer.Option(
        REPO_ROOT / "config" / "figure_contract.v1.yaml",
        "--contract",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the figure contract YAML.",
    ),
    thresholds: Path = typer.Option(
        REPO_ROOT / "config" / "validator_thresholds.v1.yaml",
        "--thresholds",
        "-t",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the validator thresholds YAML.",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        "-o",
        dir_okay=False,
        help="Optional path to write the JSON report.",
    ),
    expected: Path | None = typer.Option(
        None,
        "--expected",
        "-e",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional expected PNG for visual diff.",
    ),
) -> None:
    """Validate an SVG and emit a JSON report."""
    result = validate_svg(input_svg, contract, thresholds, expected)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if report is not None:
        report.write_text(payload)
    typer.echo(payload)
    raise typer.Exit(code=0 if result.status == "pass" else 1)


if __name__ == "__main__":
    app(prog_name="validate")
