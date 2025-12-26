#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
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


def _real_allow_force_template(entry: dict[str, Any]) -> bool:
    return bool(entry.get("allow_force_template", False))


def _load_extract_warnings(extract_report_path: Path) -> list[dict[str, Any]]:
    if not extract_report_path.exists():
        return []
    try:
        payload = json.loads(extract_report_path.read_text())
    except json.JSONDecodeError:
        return []
    warnings = payload.get("warnings")
    return warnings if isinstance(warnings, list) else []


def _detect_ocr_empty(extract_report_path: Path) -> bool:
    warnings = _load_extract_warnings(extract_report_path)
    for item in warnings:
        code = str(item.get("code", ""))
        if code in {"W4011_OCR_EMPTY", "W4004_OCR_EMPTY"}:
            return True
    return False


def _classify_failure_reasons(
    errors: list[dict[str, Any]],
    validation: dict[str, Any] | None,
    ocr_empty: bool,
) -> list[str]:
    reasons: set[str] = set()
    error_codes = {str(item.get("code", "")) for item in errors}
    if "E5107_CLASSIFY_UNKNOWN" in error_codes:
        reasons.add("CLASSIFY_UNKNOWN")
    if "E5108_QUALITY_GATE_FAILED" in error_codes or "E1408_REAL_VISUAL_THRESHOLD" in error_codes:
        reasons.add("GATE_FAIL")
    if "E5105_CONVERT_FAILED" in error_codes or "E1407_REAL_VALIDATION_FAILED" in error_codes:
        reasons.add("VALIDATOR_FAIL")
    if "E1406_REAL_TEMPLATE_MISMATCH" in error_codes:
        reasons.add("TEMPLATE_MISMATCH")
    if "E1409_REAL_VISUAL_DIFF_FAILED" in error_codes:
        reasons.add("DIFF_FAIL")
    if any(code == "E2007_PATH_TOO_COMPLEX" for code in error_codes):
        reasons.add("CURVE_FAIL")
    if ocr_empty:
        reasons.add("OCR_EMPTY")
    if validation:
        for err in validation.get("errors", []) if isinstance(validation.get("errors"), list) else []:
            if str(err.get("code")) == "E2007_PATH_TOO_COMPLEX":
                reasons.add("CURVE_FAIL")
    if not reasons:
        reasons.add("UNKNOWN")
    return sorted(reasons)


def _write_real_summary(
    output_root: Path,
    real_results: list[dict[str, Any]],
    real_summary: dict[str, Any] | None,
) -> None:
    if real_summary is None:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    failures_by_reason: dict[str, int] = {}
    failed_cases: list[dict[str, Any]] = []
    for result in real_results:
        if result.get("status") == "pass":
            continue
        report = result.get("report", {}) if isinstance(result.get("report"), dict) else {}
        reasons = report.get("failure_reasons") or ["UNKNOWN"]
        for reason in reasons:
            failures_by_reason[str(reason)] = failures_by_reason.get(str(reason), 0) + 1
        failed_cases.append(
            {
                "id": result.get("id"),
                "reasons": reasons,
                "report_path": result.get("report_path"),
                "diff_path": result.get("diff_path"),
            }
        )
    payload = {
        "summary": real_summary,
        "failures_by_reason": failures_by_reason,
        "failed_cases": failed_cases,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Real Regression Summary",
        "",
        f"Total: {real_summary.get('total', 0)}",
        f"Passed: {real_summary.get('passed', 0)}",
        f"Failed: {real_summary.get('failed', 0)}",
        "",
        "## Failures by reason",
    ]
    if failures_by_reason:
        for reason, count in sorted(failures_by_reason.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Failed cases")
    if failed_cases:
        for case in failed_cases:
            reasons_str = ", ".join(case.get("reasons", []))
            lines.append(f"- {case.get('id')}: {reasons_str} ({case.get('report_path')})")
    else:
        lines.append("- none")
    (output_root / "summary.md").write_text("\n".join(lines))


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


def _variant_gate_overrides(thresholds_path: Path, input_variant: str) -> dict[str, Any]:
    if input_variant != "hard":
        return {}
    return _gate_section_overrides(thresholds_path, "quality_gate_hard")


def _gate_section_overrides(thresholds_path: Path, section_name: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(thresholds_path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    section = data.get(section_name)
    if not isinstance(section, dict):
        return {}
    overrides: dict[str, Any] = {}
    if "rmse_max" in section:
        overrides["rmse_max"] = section.get("rmse_max")
    if "bad_pixel_ratio_max" in section:
        overrides["bad_pixel_ratio_max"] = section.get("bad_pixel_ratio_max")
    if "pixel_tolerance" in section:
        overrides["pixel_tolerance"] = section.get("pixel_tolerance")
    return overrides


def _tier_gate_overrides(thresholds_path: Path, tier: str) -> dict[str, Any]:
    if tier != "hard":
        return {}
    return _gate_section_overrides(thresholds_path, "quality_gate_hard")


def _has_rasterizer() -> bool:
    if shutil.which("resvg"):
        return True
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        return False
    return True


def _run_dataset(
    dataset: Path,
    contract: Path,
    thresholds: Path,
    output_root: Path,
    pipeline: str,
    input_variant: str,
    limit: int | None,
    only: str | None,
    tier: str,
) -> dict[str, Any]:
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

    variant_gate_overrides = _variant_gate_overrides(thresholds, input_variant)
    tier_gate_overrides = _tier_gate_overrides(thresholds, tier)

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    template_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    processed = 0
    for entry in entries:
        case_dir = _case_entry_dir(base_dir, entry)
        case_id = _case_entry_id(entry, case_dir)
        if only and case_id != only:
            continue
        if limit is not None and processed >= limit:
            break
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
            output_root,
            pipeline,
            gate_overrides={
                **variant_gate_overrides,
                **tier_gate_overrides,
                **_case_gate_overrides(entry),
            },
            input_variant=input_variant,
        )
        case_result["id"] = case_id
        case_result["dir"] = str(case_dir)
        if case_result["status"] == "pass":
            passed += 1
        else:
            failed += 1
        results.append(case_result)
        processed += 1
        status_line = f"{case_id}: {case_result['status']} (report: {case_result['report_path']})"
        if case_result["diff_path"]:
            status_line += f" diff: {case_result['diff_path']}"
        typer.echo(status_line)

    return {
        "summary": {"total": len(results), "passed": passed, "failed": failed},
        "cases": results,
        "template_counts": template_counts,
        "tag_counts": tag_counts,
        "failed": failed,
    }


def _print_summary(
    label: str,
    summary: dict[str, int],
    template_counts: dict[str, int],
    tag_counts: dict[str, int],
    input_variant: str,
    pipeline: str,
) -> None:
    typer.echo(
        f"Summary ({label}): total={summary['total']} passed={summary['passed']} failed={summary['failed']} "
        f"variant={input_variant} pipeline={pipeline}"
    )
    typer.echo("Template summary:")
    for template, count in sorted(template_counts.items()):
        typer.echo(f"  {template}: {count}")
    typer.echo("Tag summary:")
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        typer.echo(f"  {tag}: {count}")


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
    gate_overrides: dict[str, Any] | None = None,
    input_variant: str = "fast",
) -> dict[str, Any]:
    input_name = "input_hard.png" if input_variant == "hard" else "input.png"
    input_png = case_dir / input_name
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
            "input_variant": input_variant,
        }
        _write_report(payload)
        return {
            "status": "fail",
            "report_path": str(report_path),
            "generated_svg": str(output_svg),
            "generated_png": str(output_png),
            "diff_path": None,
            "report": payload,
            "input_variant": input_variant,
        }

    if not input_png.exists():
        return _error_report(
            "E1200_INPUT_MISSING",
            "input.png not found.",
            f"Place {input_name} in the case directory.",
            context={"path": str(input_png), "tag": "png", "input_variant": input_variant},
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
                        context={
                            "path": str(input_png),
                            "tag": "png",
                            "input_variant": input_variant,
                        },
                    )
                )
            else:
                return _error_report(
                    "E1205_INPUT_PNG_INVALID",
                    "input.png is not a valid PNG and no canvas override was provided.",
                    "Provide a valid input.png or set canvas.width/height in params.json.",
                    context={
                        "path": str(input_png),
                        "tag": "png",
                        "input_variant": input_variant,
                    },
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
                context={
                    "path": str(input_png),
                    "tag": "png",
                    "input_variant": input_variant,
                },
            )
        gate_overrides = gate_overrides or {}
        gate_enabled = gate_overrides.get("quality_gate")
        try:
            convert_png(
                input_png,
                output_svg,
                debug_dir=output_dir / "convert",
                topk=2,
                contract_path=contract,
                thresholds_path=thresholds,
                quality_gate=bool(gate_enabled) if gate_enabled is not None else True,
                gate_rmse_max=gate_overrides.get("rmse_max"),
                gate_bad_pixel_max=gate_overrides.get("bad_pixel_ratio_max"),
                gate_pixel_tolerance=gate_overrides.get("pixel_tolerance"),
            )
            regress_warnings.append(
                _issue_payload(
                    "W1300_EXPECTED_SKIPPED",
                    "Convert pipeline compares against input.png instead of expected assets.",
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
    report_payload["input_variant"] = input_variant
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
        "input_variant": input_variant,
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
    allow_force_template = _real_allow_force_template(entry)
    force_template = expected_templates[0] if allow_force_template and expected_templates else None

    try:
        convert_result = convert_png(
            input_png,
            output_svg,
            debug_dir=output_dir,
            topk=1 if force_template else 2,
            force_template=force_template,
            contract_path=contract,
            thresholds_path=thresholds,
            gate_rmse_max=max_rmse,
            gate_bad_pixel_max=max_bad_pixel_ratio,
            gate_pixel_tolerance=pixel_tolerance,
        )
    except Png2SvgError as exc:
        failure_payload = _fail(exc.code, exc.message, exc.hint, context={"tag": "convert"})
        failure_payload["report"]["failure_reasons"] = _classify_failure_reasons(
            failure_payload["report"]["errors"], None, False
        )
        failure_payload["report"]["allow_force_template"] = allow_force_template
        _write_report(failure_payload["report"])
        return failure_payload

    selected_template = convert_result.get("selected_template")
    selected = convert_result.get("selected", {})
    validation = selected.get("validation", {})
    validation_errors = validation.get("errors", [])
    validation_warnings = validation.get("warnings", [])
    candidate_dir = None
    selected_paths = selected.get("paths", {})
    if isinstance(selected_paths, dict):
        svg_path = selected_paths.get("svg")
        if isinstance(svg_path, str):
            candidate_dir = Path(svg_path).parent

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

    ocr_empty = False
    if candidate_dir is not None:
        extract_report_path = candidate_dir / "extract" / "extract_report.json"
        ocr_empty = _detect_ocr_empty(extract_report_path)

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
        "allow_force_template": allow_force_template,
        "selected_template": selected_template,
        "errors": errors,
        "warnings": warnings,
        "failure_reasons": _classify_failure_reasons(errors, validation, ocr_empty)
        if errors
        else [],
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
    tier: str = typer.Option(
        "fast",
        "--tier",
        help="Regression tier: fast, hard, or all.",
    ),
    input_variant: str = typer.Option(
        "fast",
        "--input-variant",
        help="Input variant: fast (input.png) or hard (input_hard.png).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Optional limit on number of cases to run.",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Only run a specific case id.",
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

    if pipeline not in {"render", "convert"}:
        typer.echo("ERROR E1302_PIPELINE_INVALID: pipeline must be 'render' or 'convert'.", err=True)
        typer.echo("HINT: Use --pipeline render (default) or --pipeline convert.", err=True)
        raise typer.Exit(code=1)
    if tier not in {"fast", "hard", "all"}:
        typer.echo("ERROR E1304_TIER_INVALID: tier must be fast, hard, or all.", err=True)
        typer.echo("HINT: Use --tier fast, --tier hard, or --tier all.", err=True)
        raise typer.Exit(code=1)
    if input_variant not in {"fast", "hard"}:
        typer.echo("ERROR E1303_INPUT_VARIANT_INVALID: input variant must be fast or hard.", err=True)
        typer.echo("HINT: Use --input-variant fast or --input-variant hard.", err=True)
        raise typer.Exit(code=1)

    output_root = REPO_ROOT / "output"
    fast_dataset = dataset
    hard_dataset = REPO_ROOT / "datasets" / "regression_hard_v1"
    if tier == "hard":
        hard_dataset = dataset

    tier_payload: dict[str, Any] = {}
    failed = 0

    if tier in {"fast", "all"}:
        fast_results = _run_dataset(
            fast_dataset,
            contract,
            thresholds,
            output_root / "regress",
            pipeline,
            input_variant,
            limit,
            only,
            "fast",
        )
        tier_payload["fast"] = {
            "dataset": str(fast_dataset),
            "summary": fast_results["summary"],
            "cases": fast_results["cases"],
            "input_variant": input_variant,
        }
        failed += int(fast_results["failed"])
        _print_summary(
            "fast",
            fast_results["summary"],
            fast_results["template_counts"],
            fast_results["tag_counts"],
            input_variant,
            pipeline,
        )

    if tier in {"hard", "all"}:
        if tier == "all" and input_variant != "fast":
            typer.echo("Hard regression: using input.png (fast) for hard tier.")
        hard_input_variant = "fast" if tier == "all" else input_variant
        hard_manifest = hard_dataset if hard_dataset.is_file() else hard_dataset / "manifest.yaml"
        if not hard_manifest.exists():
            typer.echo("Hard regression: skipped (manifest missing).")
            tier_payload["hard"] = {
                "dataset": str(hard_dataset),
                "summary": {"skipped": True, "reason": "manifest missing"},
                "cases": [],
                "input_variant": hard_input_variant,
            }
        elif not _has_rasterizer():
            typer.echo("Hard regression: skipped (no rasterizer available).")
            tier_payload["hard"] = {
                "dataset": str(hard_dataset),
                "summary": {"skipped": True, "reason": "no rasterizer"},
                "cases": [],
                "input_variant": hard_input_variant,
            }
        else:
            hard_results = _run_dataset(
                hard_dataset,
                contract,
                thresholds,
                output_root / "regress_hard",
                pipeline,
                hard_input_variant,
                limit,
                only,
                "hard",
            )
            tier_payload["hard"] = {
                "dataset": str(hard_dataset),
                "summary": hard_results["summary"],
                "cases": hard_results["cases"],
                "input_variant": hard_input_variant,
            }
            failed += int(hard_results["failed"])
            _print_summary(
                "hard",
                hard_results["summary"],
                hard_results["template_counts"],
                hard_results["tag_counts"],
                hard_input_variant,
                pipeline,
            )

    payload_dict: dict[str, Any] = {
        "tier": tier,
        "pipeline": pipeline,
        "input_variant": input_variant,
    }
    if tier == "all":
        payload_dict.update(tier_payload)
    elif tier == "fast":
        payload_dict.update(tier_payload.get("fast", {}))
    else:
        payload_dict.update(tier_payload.get("hard", {}))

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

    if real_summary is not None:
        _write_real_summary(REPO_ROOT / "output" / "regress_real", real_results, real_summary)

    payload = json.dumps(payload_dict, indent=2, sort_keys=True)
    if report is not None:
        report.write_text(payload)
    typer.echo(payload)
    if tier == "all":
        combined_total = 0
        combined_passed = 0
        combined_failed = 0
        for label in ("fast", "hard"):
            entry = tier_payload.get(label, {})
            summary = entry.get("summary", {})
            if summary.get("skipped"):
                continue
            combined_total += int(summary.get("total", 0))
            combined_passed += int(summary.get("passed", 0))
            combined_failed += int(summary.get("failed", 0))
        typer.echo(
            f"Combined summary: total={combined_total} passed={combined_passed} failed={combined_failed} "
            f"pipeline={pipeline}"
        )
    if real_summary is not None and real_summary.get("failed"):
        failed += int(real_summary["failed"])
    raise typer.Exit(code=0 if failed == 0 else 1)


if __name__ == "__main__":
    app(prog_name="regress")
