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

from png2svg import (  # noqa: E402
    Png2SvgError,
    classify_png,
    convert_png,
    extract_skeleton,
    render_svg,
)

app = typer.Typer(
    add_completion=False,
    help="Render SVGs, classify templates, extract params, or convert PNGs.",
)


@app.command("render")
def render(
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


@app.command()
def classify(
    input_png: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input PNG to classify.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        dir_okay=False,
        help="Optional path to write classification JSON.",
    ),
    debug_dir: Path | None = typer.Option(
        None,
        "--debug-dir",
        help="Optional directory to write debug overlay and features.",
    ),
    thresholds: Path | None = typer.Option(
        None,
        "--classifier-thresholds",
        dir_okay=False,
        help="Optional classifier thresholds YAML.",
    ),
) -> None:
    """Classify a PNG into a known template."""
    result = classify_png(input_png, debug_dir, thresholds)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if out is not None:
        out.write_text(payload)
    typer.echo(payload)


@app.command()
def extract(
    input_png: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input PNG to extract params from.",
    ),
    template: str = typer.Option(
        "auto",
        "--template",
        help="Template id to extract (or 'auto' for classifier).",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        dir_okay=False,
        help="Path to write extracted params.json.",
    ),
    debug_dir: Path | None = typer.Option(
        None,
        "--debug-dir",
        help="Optional directory to write debug artifacts.",
    ),
    text_mode: str = typer.Option(
        "hybrid",
        "--text-mode",
        help="Text strategy: template_text, ocr_text, or hybrid.",
    ),
) -> None:
    """Extract params.json skeleton from a PNG."""
    try:
        params = extract_skeleton(input_png, template, debug_dir, text_mode=text_mode)
    except Png2SvgError as exc:
        typer.echo(f"ERROR {exc.code}: {exc.message}", err=True)
        typer.echo(f"HINT: {exc.hint}", err=True)
        raise typer.Exit(code=1)
    payload = json.dumps(params, indent=2, sort_keys=True)
    out.write_text(payload)
    typer.echo(payload)


@app.command()
def convert(
    input_png: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input PNG to convert.",
    ),
    output_svg: Path = typer.Option(
        ...,
        "--out",
        "-o",
        dir_okay=False,
        help="Output SVG path.",
    ),
    debug_dir: Path | None = typer.Option(
        None,
        "--debug-dir",
        help="Optional directory to write debug artifacts.",
    ),
    topk: int = typer.Option(
        2,
        "--topk",
        help="How many top classifier candidates to try.",
    ),
    force_template: str | None = typer.Option(
        None,
        "--force-template",
        help="Force a template id and skip unknown gating.",
    ),
    thresholds: Path | None = typer.Option(
        None,
        "--classifier-thresholds",
        dir_okay=False,
        help="Optional classifier thresholds YAML.",
    ),
    quality_gate: str = typer.Option(
        "on",
        "--quality-gate",
        help="Quality gate toggle: on or off.",
    ),
    gate_rmse_max: float | None = typer.Option(
        None,
        "--gate-rmse-max",
        help="Override quality gate RMSE maximum.",
    ),
    gate_bad_pixel_max: float | None = typer.Option(
        None,
        "--gate-bad-pixel-max",
        help="Override quality gate bad pixel ratio maximum.",
    ),
    gate_pixel_tolerance: int | None = typer.Option(
        None,
        "--gate-pixel-tolerance",
        help="Override quality gate pixel tolerance.",
    ),
    text_mode: str = typer.Option(
        "hybrid",
        "--text-mode",
        help="Text strategy: template_text, ocr_text, or hybrid.",
    ),
    allow_failed_gate: bool = typer.Option(
        False,
        "--allow-failed-gate",
        help="Write output SVG even if quality gate fails.",
    ),
    emit_report_json: bool = typer.Option(
        False,
        "--emit-report-json",
        help="Write convert_report.json into the debug directory.",
    ),
) -> None:
    """Classify, extract, render, and validate in one step."""
    gate_value = quality_gate.strip().lower()
    if gate_value in {"on", "true", "1", "yes"}:
        gate_enabled = True
    elif gate_value in {"off", "false", "0", "no"}:
        gate_enabled = False
    else:
        typer.echo("ERROR E1103_QUALITY_GATE_INVALID: quality_gate must be on or off.", err=True)
        typer.echo("HINT: Use --quality-gate on or --quality-gate off.", err=True)
        raise typer.Exit(code=1)
    try:
        result = convert_png(
            input_png,
            output_svg,
            debug_dir=debug_dir,
            topk=topk,
            force_template=force_template,
            classifier_thresholds_path=thresholds,
            quality_gate=gate_enabled,
            gate_rmse_max=gate_rmse_max,
            gate_bad_pixel_max=gate_bad_pixel_max,
            gate_pixel_tolerance=gate_pixel_tolerance,
            text_mode=text_mode,
            allow_failed_gate=allow_failed_gate,
            emit_report_json=emit_report_json,
        )
    except Png2SvgError as exc:
        typer.echo(f"ERROR {exc.code}: {exc.message}", err=True)
        typer.echo(f"HINT: {exc.hint}", err=True)
        raise typer.Exit(code=1)
    payload = json.dumps(result, indent=2, sort_keys=True)
    typer.echo(payload)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in {
        "render",
        "classify",
        "extract",
        "convert",
        "-h",
        "--help",
    }:
        sys.argv.insert(1, "render")
    app(prog_name="png2svg")
