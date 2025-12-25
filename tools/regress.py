#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from png2svg import Png2SvgError, convert_png, render_svg  # noqa: E402
from common.png_utils import has_png_magic  # noqa: E402
from validators import validate_svg  # noqa: E402
from validators.visual_diff import RasterizeError, rasterize_svg_to_png  # noqa: E402

app = typer.Typer(add_completion=False, help="Run regression cases.")


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("manifest.yaml must contain a mapping.")
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("manifest.yaml must contain a list under 'cases'.")
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


def _issue_payload(
    code: str,
    message: str,
    hint: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"code": code, "message": message, "hint": hint}
    if context:
        payload["context"] = context
    return payload


def _run_case(
    case_dir: Path,
    case_id: str,
    contract: Path,
    thresholds: Path,
    output_root: Path,
    pipeline: str,
) -> dict[str, Any]:
    input_png = case_dir / "input.png"
    params = case_dir / "params.json"
    expected_png = case_dir / "expected.png"
    expected_svg = case_dir / "expected.svg"
    output_dir = output_root / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_svg = output_dir / "generated.svg"
    output_png = output_dir / "generated.png"
    report_path = output_dir / "report.json"
    diff_path = output_dir / "diff.png"
    output_expected_png = output_dir / "expected.png"

    def _write_placeholder_svg(path: Path, reason: str) -> None:
        if path.exists():
            return
        path.write_text(
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\">"
            f"<!-- {reason} --></svg>"
        )

    def _write_placeholder_png(path: Path) -> None:
        if path.exists():
            return
        try:
            image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            image.save(path)
        except Exception:
            path.write_bytes(b"")

    _write_placeholder_svg(output_svg, "placeholder")
    _write_placeholder_png(output_png)

    def _write_report(payload: dict[str, Any]) -> None:
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _error_report(
        code: str, message: str, hint: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = {
            "status": "fail",
            "errors": [_issue_payload(code, message, hint, context)],
            "warnings": [],
            "stats": {},
        }
        _write_report(payload)
        return {
            "status": "fail",
            "report_path": str(report_path),
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_path": None,
            "report": payload,
        }

    if not input_png.exists():
        return _error_report(
            "E1200_INPUT_MISSING",
            "input.png not found.",
            "Place input.png in the case directory.",
            context={"path": str(input_png), "tag": "png"},
        )
    if pipeline == "render" and not params.exists():
        return _error_report(
            "E1201_PARAMS_MISSING",
            "params.json not found.",
            "Place params.json in the case directory.",
            context={"path": str(params), "tag": "json"},
        )

    regress_warnings: list[dict[str, Any]] = []
    params_data: dict[str, Any] | None = None
    canvas_present = False
    if pipeline == "render":
        try:
            params_data = json.loads(params.read_text())
            canvas = params_data.get("canvas")
            canvas_present = (
                isinstance(canvas, dict)
                and canvas.get("width") is not None
                and canvas.get("height") is not None
            )
        except json.JSONDecodeError:
            params_data = None

        if not has_png_magic(input_png) and params_data is not None:
            if canvas_present:
                regress_warnings.append(
                    _issue_payload(
                        "W1201_INPUT_PNG_INVALID_CANVAS_USED",
                        "input.png is not a valid PNG; using canvas from params.json.",
                        "Fix or replace input.png to avoid relying on fallback canvas.",
                        context={"path": str(input_png), "tag": "png"},
                    )
                )
            else:
                return _error_report(
                    "E1205_INPUT_PNG_INVALID",
                    "input.png is not a valid PNG and no canvas override was provided.",
                    "Provide a valid input.png or set canvas.width/height in params.json.",
                    context={"path": str(input_png), "tag": "png"},
                )

        try:
            render_svg(input_png, params, output_svg)
        except Png2SvgError as exc:
            return _error_report(exc.code, exc.message, exc.hint, context={"tag": "svg"})
        except Exception as exc:  # noqa: BLE001
            return _error_report(
                "E1299_RENDER_FAILED",
                f"{exc}",
                "Check input.png and params.json.",
                context={"tag": "svg"},
            )
    else:
        if not has_png_magic(input_png):
            return _error_report(
                "E1205_INPUT_PNG_INVALID",
                "input.png is not a valid PNG.",
                "Provide a valid input.png for convert pipeline.",
                context={"path": str(input_png), "tag": "png"},
            )
        try:
            convert_png(
                input_png,
                output_svg,
                debug_dir=output_dir / "convert",
                topk=2,
                contract_path=contract,
                thresholds_path=thresholds,
                enable_visual_diff=False,
            )
            regress_warnings.append(
                _issue_payload(
                    "W1300_EXPECTED_SKIPPED",
                    "Convert pipeline skips expected.svg/expected.png comparison.",
                    "Use render pipeline for strict regression diffs.",
                    context={"tag": "regress"},
                )
            )
        except Png2SvgError as exc:
            return _error_report(exc.code, exc.message, exc.hint, context={"tag": "svg"})

    expected_png_path: Path | None = None
    expected_error: dict[str, Any] | None = None
    if pipeline == "render":
        if expected_svg.exists():
            try:
                output_expected_png.parent.mkdir(parents=True, exist_ok=True)
                rasterize_svg_to_png(expected_svg, output_expected_png)
                expected_png_path = output_expected_png
            except RasterizeError as exc:
                expected_error = _issue_payload(
                    "E1206_EXPECTED_SVG_RASTERIZE_FAILED",
                    f"Failed to rasterize expected.svg: {exc}",
                    "Ensure expected.svg is valid and rasterizer dependencies are available.",
                    context={"path": str(expected_svg), "tag": "svg"},
                )
        elif expected_png.exists():
            if has_png_magic(expected_png):
                output_expected_png.write_bytes(expected_png.read_bytes())
                expected_png_path = output_expected_png
            else:
                expected_error = _issue_payload(
                    "E1204_EXPECTED_PNG_INVALID",
                    "expected.png is not a valid PNG and no expected.svg is available.",
                    "Provide a valid expected.svg (preferred) or a valid expected.png.",
                    context={"path": str(expected_png), "tag": "png"},
                )
        else:
            expected_error = _issue_payload(
                "E1203_EXPECTED_MISSING",
                "No expected.svg or expected.png found for this case.",
                "Add expected.svg (preferred) or expected.png to the case directory.",
                context={"path": str(case_dir), "tag": "case"},
            )

    result = validate_svg(
        output_svg,
        contract,
        thresholds,
        expected_png_path,
        actual_png_path=output_png if expected_png_path else None,
        diff_png_path=diff_path,
    )
    report_payload = result.to_dict()
    report_payload.setdefault("errors", [])
    report_payload.setdefault("warnings", [])
    if expected_error:
        report_payload["errors"].append(expected_error)
    if regress_warnings:
        report_payload["warnings"].extend(regress_warnings)
    if expected_png_path is None:
        try:
            rasterize_svg_to_png(output_svg, output_png)
        except RasterizeError as exc:
            report_payload["errors"].append(
                _issue_payload(
                    "E3001_RASTERIZE_FAILED",
                    f"Rasterization failed: {exc}",
                    "Install resvg or cairosvg and ensure the SVG is valid.",
                    context={"svg": str(output_svg), "tag": "svg"},
                )
            )
    if report_payload.get("errors"):
        report_payload["status"] = "fail"
    _write_report(report_payload)
    diff_path_value = str(diff_path) if diff_path.exists() else None
    return {
        "status": report_payload.get("status", "fail"),
        "report_path": str(report_path),
        "generated_svg": str(output_svg),
        "generated_png": str(output_png),
        "diff_path": diff_path_value,
        "report": report_payload,
    }


@app.command()
def main(
    dataset: Path = typer.Argument(
        REPO_ROOT / "datasets" / "regression_v0",
        exists=True,
        readable=True,
        help="Dataset directory or manifest.yaml path.",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        dir_okay=False,
        help="Optional path to write the JSON report.",
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
    pipeline: str = typer.Option(
        "render",
        "--pipeline",
        help="Pipeline to run: render or convert.",
    ),
) -> None:
    """Run regression cases listed in manifest.yaml."""
    dataset = dataset.resolve()
    manifest_path, base_dir = _resolve_manifest(dataset)
    if base_dir is None:
        case_dir = manifest_path
        if not case_dir.is_dir():
            typer.echo("ERROR E1300_MANIFEST_MISSING: manifest.yaml not found.", err=True)
            typer.echo("HINT: Provide a dataset directory with manifest.yaml.", err=True)
            raise typer.Exit(code=1)
        entries = [{"dir": case_dir.name, "id": case_dir.name}]
        base_dir = case_dir.parent
    else:
        try:
            manifest = _load_manifest(manifest_path)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"ERROR E1301_MANIFEST_INVALID: {exc}", err=True)
            typer.echo("HINT: Ensure manifest.yaml has a cases list.", err=True)
            raise typer.Exit(code=1)
        entries = manifest["cases"]

    if pipeline not in {"render", "convert"}:
        typer.echo("ERROR E1302_PIPELINE_INVALID: pipeline must be 'render' or 'convert'.", err=True)
        typer.echo("HINT: Use --pipeline render (default) or --pipeline convert.", err=True)
        raise typer.Exit(code=1)

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    template_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for entry in entries:
        case_dir = _case_entry_dir(base_dir, entry)
        case_id = _case_entry_id(entry, case_dir)
        params_path = case_dir / "params.json"
        template = "unknown"
        if params_path.exists():
            try:
                params = json.loads(params_path.read_text())
                template = str(params.get("template") or "unknown")
            except json.JSONDecodeError:
                template = "unknown"
        template_counts[template] = template_counts.get(template, 0) + 1
        if isinstance(entry, dict):
            tags = entry.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
        case_result = _run_case(
            case_dir,
            case_id,
            contract,
            thresholds,
            REPO_ROOT / "output" / "regress",
            pipeline,
        )
        case_result["id"] = case_id
        case_result["dir"] = str(case_dir)
        if case_result["status"] == "pass":
            passed += 1
        else:
            failed += 1
        results.append(case_result)
        status_line = f"{case_id}: {case_result['status']} (report: {case_result['report_path']})"
        if case_result["diff_path"]:
            status_line += f" diff: {case_result['diff_path']}"
        typer.echo(status_line)

    summary = {"total": len(results), "passed": passed, "failed": failed}
    payload = json.dumps({"summary": summary, "cases": results}, indent=2, sort_keys=True)
    if report is not None:
        report.write_text(payload)
    typer.echo(payload)
    typer.echo(f"Summary: total={summary['total']} passed={summary['passed']} failed={summary['failed']}")
    typer.echo("Template summary:")
    for template, count in sorted(template_counts.items()):
        typer.echo(f"  {template}: {count}")
    typer.echo("Tag summary:")
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        typer.echo(f"  {tag}: {count}")
    raise typer.Exit(code=0 if failed == 0 else 1)


if __name__ == "__main__":
    app(prog_name="regress")
