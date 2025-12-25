#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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
from validators.visual_diff import (  # noqa: E402
    DiffError,
    RasterizeError,
    compute_visual_diff,
    rasterize_svg_to_png,
    write_diff_image,
)

app = typer.Typer(add_completion=False, help="Run regression cases.")


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


def _run_real_case(
    entry: dict[str, Any],
    real_root: Path,
    contract: Path,
    thresholds: Path,
    output_root: Path,
) -> dict[str, Any]:
    case_id = str(entry.get("id") or "unknown")
    relative_path = entry.get("relative_path")
    output_dir = output_root / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    output_svg = output_dir / "generated.svg"
    output_png = output_dir / "generated.png"
    diff_png = output_dir / "diff.png"

    def _write_report(payload: dict[str, Any]) -> None:
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _fail(
        code: str,
        message: str,
        hint: str,
        context: dict[str, Any] | None = None,
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
            "diff_path": str(diff_png) if diff_png.exists() else None,
            "report": payload,
        }

    if not isinstance(relative_path, str) or not relative_path.strip():
        return _fail(
            "E1400_REAL_PATH_MISSING",
            "relative_path is required for real regression case.",
            "Set relative_path to a PNG under REAL_PNG_DIR.",
            context={"id": case_id, "tag": "manifest"},
        )
    input_png = real_root / relative_path
    if not input_png.exists():
        return _fail(
            "E1401_REAL_PNG_MISSING",
            "Real PNG not found.",
            "Ensure REAL_PNG_DIR contains the referenced file.",
            context={"path": str(input_png), "tag": "png"},
        )
    if not has_png_magic(input_png):
        return _fail(
            "E1402_REAL_PNG_INVALID",
            "Real PNG failed magic header check.",
            "Ensure the real PNG is a valid binary PNG.",
            context={"path": str(input_png), "tag": "png"},
        )

    try:
        gates = _real_gates(entry)
    except ValueError as exc:
        return _fail(
            "E1403_REAL_GATES_MISSING",
            str(exc),
            "Add a gates section with visual diff thresholds.",
            context={"id": case_id, "tag": "manifest"},
        )

    must_pass_validator = bool(gates.get("must_pass_validator", True))
    try:
        max_rmse = float(gates["max_rmse"])
        max_bad_pixel_ratio = float(gates["max_bad_pixel_ratio"])
        pixel_tolerance = int(gates["pixel_tolerance"])
    except KeyError as exc:
        return _fail(
            "E1404_REAL_GATES_INCOMPLETE",
            f"Missing gate value: {exc}",
            "Provide max_rmse, max_bad_pixel_ratio, and pixel_tolerance.",
            context={"id": case_id, "tag": "manifest"},
        )
    except (TypeError, ValueError) as exc:
        return _fail(
            "E1405_REAL_GATES_INVALID",
            f"Invalid gate value: {exc}",
            "Ensure gate values are numeric.",
            context={"id": case_id, "tag": "manifest"},
        )

    expected_templates = _real_expected_templates(entry)

    try:
        convert_result = convert_png(
            input_png,
            output_svg,
            debug_dir=output_dir,
            topk=2,
            contract_path=contract,
            thresholds_path=thresholds,
            enable_visual_diff=False,
        )
    except Png2SvgError as exc:
        return _fail(exc.code, exc.message, exc.hint, context={"tag": "convert"})

    selected_template = convert_result.get("selected_template")
    selected = convert_result.get("selected", {})
    validation = selected.get("validation", {})
    validation_errors = validation.get("errors", [])
    validation_warnings = validation.get("warnings", [])

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if expected_templates and selected_template not in expected_templates:
        errors.append(
            _issue_payload(
                "E1406_REAL_TEMPLATE_MISMATCH",
                "Selected template does not match expected template(s).",
                "Update expected_template(s) or fix classifier/extractor output.",
                context={"selected": selected_template, "expected": expected_templates, "tag": "manifest"},
            )
        )
    if must_pass_validator and validation_errors:
        errors.append(
            _issue_payload(
                "E1407_REAL_VALIDATION_FAILED",
                "Validator reported errors for converted SVG.",
                "Inspect validate_report.json under the debug directory.",
                context={"tag": "validator"},
            )
        )

    stats: dict[str, Any] = {
        "validation": validation,
    }

    try:
        backend = rasterize_svg_to_png(output_svg, output_png)
        metrics = compute_visual_diff(output_png, input_png, pixel_tolerance)
        stats["visual_diff"] = {
            "backend": backend,
            "rmse": metrics.rmse,
            "bad_pixel_ratio": metrics.bad_pixel_ratio,
            "pixel_tolerance": pixel_tolerance,
            "rmse_max": max_rmse,
            "bad_pixel_ratio_max": max_bad_pixel_ratio,
        }
        if metrics.rmse > max_rmse or metrics.bad_pixel_ratio > max_bad_pixel_ratio:
            errors.append(
                _issue_payload(
                    "E1408_REAL_VISUAL_THRESHOLD",
                    "Visual diff metrics exceed per-case thresholds.",
                    "Adjust extractor/renderer output or relax the gate thresholds.",
                    context={
                        "rmse": metrics.rmse,
                        "rmse_max": max_rmse,
                        "bad_pixel_ratio": metrics.bad_pixel_ratio,
                        "bad_pixel_ratio_max": max_bad_pixel_ratio,
                        "tag": "visual_diff",
                    },
                )
            )
            try:
                write_diff_image(output_png, input_png, diff_png, pixel_tolerance)
            except Exception:
                pass
    except (RasterizeError, DiffError, OSError) as exc:
        errors.append(
            _issue_payload(
                "E1409_REAL_VISUAL_DIFF_FAILED",
                f"Visual diff failed: {exc}",
                "Ensure rasterizer dependencies are available and PNG sizes match.",
                context={"tag": "visual_diff"},
            )
        )

    if validation_warnings:
        warnings.extend(validation_warnings)

    payload = {
        "status": "pass" if not errors else "fail",
        "id": case_id,
        "relative_path": str(relative_path),
        "input_png": str(input_png),
        "expected_templates": expected_templates,
        "selected_template": selected_template,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "paths": {
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_png": str(diff_png) if diff_png.exists() else None,
            "convert_report": str(output_dir / "convert_report.json"),
        },
    }
    _write_report(payload)
    return {
        "status": payload["status"],
        "report_path": str(report_path),
        "generated_svg": str(output_svg),
        "generated_png": str(output_png),
        "diff_path": str(diff_png) if diff_png.exists() else None,
        "report": payload,
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
    real_manifest: Path | None = typer.Option(
        REPO_ROOT / "datasets" / "real_regression_v1" / "manifest.yaml",
        "--real-manifest",
        dir_okay=False,
        help="Optional manifest for real PNG regression (uses REAL_PNG_DIR).",
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
    payload_dict: dict[str, Any] = {"summary": summary, "cases": results}

    real_results: list[dict[str, Any]] = []
    real_summary: dict[str, Any] | None = None
    real_failed = 0
    if real_manifest is not None and real_manifest.exists():
        real_root_value = os.environ.get("REAL_PNG_DIR")
        if not real_root_value:
            typer.echo("Real regression: skipped (no REAL_PNG_DIR).")
            real_summary = {"skipped": True, "reason": "no REAL_PNG_DIR"}
        else:
            real_root = Path(real_root_value).expanduser()
            typer.echo(f"Real regression: using REAL_PNG_DIR={real_root}")
            try:
                real_manifest_data = _load_real_manifest(real_manifest)
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"ERROR E1399_REAL_MANIFEST_INVALID: {exc}", err=True)
                typer.echo("HINT: Ensure real manifest has a cases list.", err=True)
                raise typer.Exit(code=1)
            for entry in real_manifest_data["cases"]:
                if not isinstance(entry, dict):
                    continue
                case_id = str(entry.get("id") or "unknown")
                case_result = _run_real_case(
                    entry,
                    real_root,
                    contract,
                    thresholds,
                    REPO_ROOT / "output" / "regress_real",
                )
                case_result["id"] = case_id
                real_results.append(case_result)
                if case_result["status"] != "pass":
                    real_failed += 1
                status_line = f"{case_id}: {case_result['status']} (report: {case_result['report_path']})"
                if case_result.get("diff_path"):
                    status_line += f" diff: {case_result['diff_path']}"
                typer.echo(status_line)
            real_summary = {
                "total": len(real_results),
                "passed": len(real_results) - real_failed,
                "failed": real_failed,
            }
        payload_dict["real_summary"] = real_summary
        payload_dict["real_cases"] = real_results
    elif real_manifest is not None:
        typer.echo("Real regression: skipped (manifest missing).")

    payload = json.dumps(payload_dict, indent=2, sort_keys=True)
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
    if real_summary is not None and real_summary.get("failed"):
        failed += int(real_summary["failed"])
    raise typer.Exit(code=0 if failed == 0 else 1)


if __name__ == "__main__":
    app(prog_name="regress")
