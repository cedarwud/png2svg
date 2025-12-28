from __future__ import annotations

import json
import os
import signal
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.png_utils import has_png_magic
from png2svg.classifier import classify_png
from png2svg.errors import Png2SvgError
from png2svg.extractor import extract_skeleton
from png2svg.renderer import render_svg
from validators.config import load_thresholds
from validators.validate import validate_svg
from validators.visual_diff import (
    DiffError,
    RasterizeError,
    compute_visual_diff,
    rasterize_svg_to_png,
    write_diff_image,
)

DEFAULT_CONTRACT = Path(__file__).resolve().parents[2] / "config" / "figure_contract.v1.yaml"
DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parents[2] / "config" / "validator_thresholds.v1.yaml"
)
DEFAULT_CANDIDATE_TIMEOUT_SEC = 60.0


class CandidateTimeoutError(TimeoutError):
    pass


def _issue(
    code: str,
    message: str,
    hint: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"code": code, "message": message, "hint": hint}
    if context:
        payload["context"] = context
    return payload


def _resolve_config(path: Path | None, default: Path, label: str) -> Path:
    resolved = path or default
    if not resolved.exists():
        raise Png2SvgError(
            code="E5103_CONFIG_MISSING",
            message=f"{label} config not found: {resolved}",
            hint="Ensure required config files exist under config/.",
        )
    return resolved


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_candidate_timeout(candidate_timeout_sec: float | None) -> float | None:
    if candidate_timeout_sec is not None:
        return candidate_timeout_sec
    env_value = os.environ.get("PNG2SVG_CONVERT_TIMEOUT_SEC")
    if not env_value:
        return DEFAULT_CANDIDATE_TIMEOUT_SEC
    try:
        return float(env_value)
    except ValueError:
        return DEFAULT_CANDIDATE_TIMEOUT_SEC


@contextmanager
def _candidate_timeout(seconds: float | None):
    if seconds is None or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(_signum, _frame):
        raise CandidateTimeoutError(f"Candidate timed out after {seconds}s.")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _candidate_dir(debug_dir: Path | None, template_id: str) -> Path | None:
    if debug_dir is None:
        return None
    base = debug_dir / "candidates" / template_id
    path = base
    if path.exists():
        suffix = 2
        while (debug_dir / "candidates" / f"{template_id}_{suffix}").exists():
            suffix += 1
        path = debug_dir / "candidates" / f"{template_id}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise Png2SvgError(
            code="E5110_GATE_THRESHOLD_INVALID",
            message=f"Invalid {label} value: {value}",
            hint="Provide a numeric threshold value.",
        ) from exc


def _coerce_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise Png2SvgError(
            code="E5110_GATE_THRESHOLD_INVALID",
            message=f"Invalid {label} value: {value}",
            hint="Provide an integer threshold value.",
        ) from exc


def _resolve_gate_thresholds(
    thresholds_path: Path,
    gate_rmse_max: float | None,
    gate_bad_pixel_max: float | None,
    gate_pixel_tolerance: int | None,
) -> dict[str, Any]:
    thresholds = load_thresholds(thresholds_path)
    gate_section = thresholds.get("quality_gate") or thresholds.get("visual_diff") or {}
    if not gate_section:
        raise Png2SvgError(
            code="E5109_GATE_THRESHOLDS_MISSING",
            message=f"Quality gate thresholds missing in {thresholds_path}.",
            hint="Add quality_gate or visual_diff thresholds to the thresholds config.",
        )
    rmse_max = gate_rmse_max if gate_rmse_max is not None else gate_section.get("rmse_max")
    bad_pixel_ratio_max = (
        gate_bad_pixel_max
        if gate_bad_pixel_max is not None
        else gate_section.get("bad_pixel_ratio_max")
    )
    pixel_tolerance = (
        gate_pixel_tolerance
        if gate_pixel_tolerance is not None
        else gate_section.get("pixel_tolerance")
    )
    if rmse_max is None or bad_pixel_ratio_max is None or pixel_tolerance is None:
        raise Png2SvgError(
            code="E5109_GATE_THRESHOLDS_MISSING",
            message="Quality gate thresholds incomplete.",
            hint="Provide rmse_max, bad_pixel_ratio_max, and pixel_tolerance.",
        )
    return {
        "rmse_max": _coerce_float(rmse_max, "rmse_max"),
        "bad_pixel_ratio_max": _coerce_float(bad_pixel_ratio_max, "bad_pixel_ratio_max"),
        "pixel_tolerance": _coerce_int(pixel_tolerance, "pixel_tolerance"),
    }


def _run_quality_gate(
    svg_path: Path,
    input_png: Path,
    rendered_png: Path,
    diff_png: Path | None,
    gate_thresholds: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "fail",
        "backend": None,
        "rmse": None,
        "bad_pixel_ratio": None,
        "pixel_tolerance": gate_thresholds["pixel_tolerance"],
        "rmse_max": gate_thresholds["rmse_max"],
        "bad_pixel_ratio_max": gate_thresholds["bad_pixel_ratio_max"],
        "errors": [],
        "warnings": [],
        "diff_png": str(diff_png) if diff_png else None,
    }
    try:
        backend = rasterize_svg_to_png(svg_path, rendered_png)
    except RasterizeError as exc:
        payload["errors"].append(
            _issue(
                "E5201_GATE_RASTERIZE_FAILED",
                f"Quality gate rasterize failed: {exc}",
                "Install resvg or cairosvg and ensure the SVG is valid.",
            )
        )
        return payload
    payload["backend"] = backend
    try:
        metrics = compute_visual_diff(
            rendered_png, input_png, gate_thresholds["pixel_tolerance"]
        )
    except (DiffError, OSError) as exc:
        payload["errors"].append(
            _issue(
                "E5202_GATE_DIFF_FAILED",
                f"Quality gate diff failed: {exc}",
                "Ensure input PNG and rendered PNG sizes match.",
            )
        )
        return payload
    payload["rmse"] = metrics.rmse
    payload["bad_pixel_ratio"] = metrics.bad_pixel_ratio
    if (
        metrics.rmse <= gate_thresholds["rmse_max"]
        and metrics.bad_pixel_ratio <= gate_thresholds["bad_pixel_ratio_max"]
    ):
        payload["status"] = "pass"
        return payload
    payload["status"] = "fail"
    if diff_png is not None:
        try:
            write_diff_image(
                rendered_png, input_png, diff_png, gate_thresholds["pixel_tolerance"]
            )
        except Exception:
            pass
    return payload


def _gate_skipped_payload(
    reason: str,
    gate_thresholds: dict[str, Any],
    issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "skipped",
        "reason": reason,
        "backend": None,
        "rmse": None,
        "bad_pixel_ratio": None,
        "pixel_tolerance": gate_thresholds["pixel_tolerance"],
        "rmse_max": gate_thresholds["rmse_max"],
        "bad_pixel_ratio_max": gate_thresholds["bad_pixel_ratio_max"],
        "errors": [],
        "warnings": [],
        "diff_png": None,
    }
    if issue:
        payload["errors"].append(issue)
    return payload


def _text_expectations(params: dict[str, Any]) -> dict[str, Any] | None:
    extracted = params.get("extracted")
    if not isinstance(extracted, dict):
        return None
    detected = extracted.get("texts_detected")
    if detected is None:
        return None
    try:
        detected_count = int(detected)
    except (TypeError, ValueError):
        return None
    return {"texts_detected": detected_count}


def convert_png(
    input_png: Path,
    output_svg: Path,
    debug_dir: Path | None = None,
    topk: int = 2,
    force_template: str | None = None,
    contract_path: Path | None = None,
    thresholds_path: Path | None = None,
    classifier_thresholds_path: Path | None = None,
    quality_gate: bool = True,
    gate_rmse_max: float | None = None,
    gate_bad_pixel_max: float | None = None,
    gate_pixel_tolerance: int | None = None,
    candidate_timeout_sec: float | None = None,
    text_mode: str = "hybrid",
    allow_failed_gate: bool = False,
    emit_report_json: bool = False,
) -> dict[str, Any]:
    if topk <= 0:
        raise Png2SvgError(
            code="E5100_TOPK_INVALID",
            message="topk must be >= 1.",
            hint="Pass --topk 1 or higher.",
        )
    if not input_png.exists():
        raise Png2SvgError(
            code="E5101_INPUT_MISSING",
            message=f"Input PNG not found: {input_png}",
            hint="Provide a valid input PNG path.",
        )
    if not has_png_magic(input_png):
        raise Png2SvgError(
            code="E5102_INPUT_INVALID",
            message="Input file is not a valid PNG.",
            hint="Ensure the PNG has a correct magic header.",
        )

    contract_path = _resolve_config(contract_path, DEFAULT_CONTRACT, "Contract")
    thresholds_path = _resolve_config(thresholds_path, DEFAULT_THRESHOLDS, "Thresholds")
    gate_thresholds = _resolve_gate_thresholds(
        thresholds_path, gate_rmse_max, gate_bad_pixel_max, gate_pixel_tolerance
    )

    report_requested = emit_report_json or debug_dir is not None
    if emit_report_json and debug_dir is None:
        debug_dir = output_svg.parent / "debug"
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache_path = debug_dir / "ocr_cache.json" if debug_dir is not None else None
    timeout_sec = _resolve_candidate_timeout(candidate_timeout_sec)

    classify_debug = debug_dir / "classify" if debug_dir is not None else None
    classification_error: dict[str, Any] | None = None
    try:
        classification = classify_png(
            input_png,
            debug_dir=classify_debug,
            thresholds_path=classifier_thresholds_path,
        )
    except Exception as exc:  # noqa: BLE001
        classification = {
            "template_id": "unknown",
            "decision": "unknown",
            "reason_codes": ["CLASSIFY_FAILED"],
            "confidence": 0.0,
            "candidate_templates": [],
            "image_meta": {},
            "features_summary": {},
        }
        classification_error = _issue(
            "E5106_CLASSIFY_FAILED",
            f"Classification failed: {exc}",
            "Ensure the input PNG can be decoded by Pillow.",
        )
        if not force_template:
            raise Png2SvgError(
                code="E5106_CLASSIFY_FAILED",
                message=f"Classification failed: {exc}",
                hint="Ensure the input PNG can be decoded by Pillow.",
            ) from exc
    if debug_dir is not None:
        _write_json(debug_dir / "classification.json", classification)

    candidates = classification.get("candidate_templates", [])
    candidate_list = candidates[: min(topk, len(candidates))]

    results: list[dict[str, Any]] = []

    if classification.get("decision") == "unknown" and not force_template:
        candidate_str = ", ".join(
            f"{item['template_id']}:{float(item['score']):.2f}" for item in candidates
        )
        report = {
            "status": "fail",
            "output_svg": str(output_svg),
            "classification": classification,
            "classification_error": classification_error,
            "candidates": [],
            "errors": [],
            "warnings": [],
        }
        if classification_error:
            report["errors"].append(classification_error)
        if debug_dir is not None and report_requested:
            _write_json(debug_dir / "convert_report.json", report)
        raise Png2SvgError(
            code="E5107_CLASSIFY_UNKNOWN",
            message=f"Classifier decision unknown. Top candidates: {candidate_str}.",
            hint="Provide --force-template <id> to override or adjust classifier thresholds.",
        )

    if force_template:
        candidate_list = [{"template_id": force_template, "score": 0.0}]
    elif not candidate_list:
        raise Png2SvgError(
            code="E5104_NO_CANDIDATES",
            message="Classifier returned no candidates.",
            hint="Check input PNG and classifier rules.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for idx, candidate in enumerate(candidate_list, start=1):
            template_id = str(candidate.get("template_id"))
            score = float(candidate.get("score", 0.0))
            candidate_dir = _candidate_dir(debug_dir, template_id)
            params_path = (
                candidate_dir / "params.json" if candidate_dir else temp_root / f"{idx}.json"
            )
            svg_path = (
                candidate_dir / "out.svg" if candidate_dir else temp_root / f"{idx}.svg"
            )
            report_path = (
                candidate_dir / "validate_report.json"
                if candidate_dir
                else temp_root / f"{idx}_report.json"
            )
            generated_png = (
                candidate_dir / "rendered.png" if candidate_dir else temp_root / f"{idx}.png"
            )
            diff_png = candidate_dir / "diff.png" if candidate_dir else None
            gate_report_path = (
                candidate_dir / "gate_report.json"
                if candidate_dir
                else temp_root / f"{idx}_gate.json"
            )

            attempt: dict[str, Any] = {
                "template_id": template_id,
                "score": score,
                "status": "fail",
                "validation_warning_count": 0,
                "validation_error_count": 0,
                "quality_gate": None,
                "paths": {
                    "params": str(params_path),
                    "svg": str(svg_path),
                    "validate_report": str(report_path),
                    "rendered_png": str(generated_png),
                    "diff_png": str(diff_png) if diff_png else None,
                    "gate_report": str(gate_report_path),
                    "snap_preview_svg": str(candidate_dir / "snap_preview.svg")
                    if candidate_dir
                    else None,
                    "snap_preview_png": str(candidate_dir / "snap_preview.png")
                    if candidate_dir
                    else None,
                },
            }

            extract_debug = candidate_dir / "extract" if candidate_dir else None
            try:
                with _candidate_timeout(timeout_sec):
                    params = extract_skeleton(
                        input_png,
                        template_id,
                        extract_debug,
                        ocr_cache_path=ocr_cache_path,
                        text_mode=text_mode,
                    )
            except CandidateTimeoutError as exc:
                attempt["error"] = _issue(
                    "E5198_CANDIDATE_TIMEOUT",
                    str(exc),
                    "Increase PNG2SVG_CONVERT_TIMEOUT_SEC or reduce OCR workload.",
                    {"stage": "extract"},
                )
                gate_payload = _gate_skipped_payload(
                    "candidate_timeout", gate_thresholds, attempt["error"]
                )
                attempt["quality_gate"] = gate_payload
                _write_json(gate_report_path, gate_payload)
                results.append(attempt)
                continue
            except Png2SvgError as exc:
                attempt["error"] = _issue(exc.code, exc.message, exc.hint, {"stage": "extract"})
                gate_payload = _gate_skipped_payload(
                    "extract_failed", gate_thresholds, attempt["error"]
                )
                attempt["quality_gate"] = gate_payload
                _write_json(gate_report_path, gate_payload)
                results.append(attempt)
                continue
            params_path.write_text(json.dumps(params, indent=2, sort_keys=True))
            extracted = params.get("extracted") if isinstance(params, dict) else None
            if isinstance(extracted, dict):
                attempt["ocr_stats"] = extracted.get("ocr_stats")
                attempt["text_mode"] = extracted.get("text_mode")

            try:
                with _candidate_timeout(timeout_sec):
                    render_svg(input_png, params_path, svg_path)
            except CandidateTimeoutError as exc:
                attempt["error"] = _issue(
                    "E5198_CANDIDATE_TIMEOUT",
                    str(exc),
                    "Increase PNG2SVG_CONVERT_TIMEOUT_SEC or simplify rendering.",
                    {"stage": "render"},
                )
                gate_payload = _gate_skipped_payload(
                    "candidate_timeout", gate_thresholds, attempt["error"]
                )
                attempt["quality_gate"] = gate_payload
                _write_json(gate_report_path, gate_payload)
                results.append(attempt)
                continue
            except Png2SvgError as exc:
                attempt["error"] = _issue(exc.code, exc.message, exc.hint, {"stage": "render"})
                gate_payload = _gate_skipped_payload(
                    "render_failed", gate_thresholds, attempt["error"]
                )
                attempt["quality_gate"] = gate_payload
                _write_json(gate_report_path, gate_payload)
                results.append(attempt)
                continue
            except Exception as exc:  # noqa: BLE001
                attempt["error"] = _issue(
                    "E5199_RENDER_FAILED",
                    f"{exc}",
                    "Check input PNG and extracted params.",
                    {"stage": "render"},
                )
                gate_payload = _gate_skipped_payload(
                    "render_failed", gate_thresholds, attempt["error"]
                )
                attempt["quality_gate"] = gate_payload
                _write_json(gate_report_path, gate_payload)
                results.append(attempt)
                continue

            if candidate_dir is not None:
                snap_svg = candidate_dir / "snap_preview.svg"
                snap_png = candidate_dir / "snap_preview.png"
                snap_svg.write_bytes(svg_path.read_bytes())
                try:
                    rasterize_svg_to_png(snap_svg, snap_png)
                except RasterizeError as exc:
                    attempt["snap_preview_error"] = _issue(
                        "W5102_SNAP_PREVIEW_FAILED",
                        f"Failed to rasterize snap preview: {exc}",
                        "Install resvg or cairosvg to render snap preview.",
                        {"stage": "snap_preview"},
                    )

            try:
                with _candidate_timeout(timeout_sec):
                    report = validate_svg(
                        svg_path,
                        contract_path,
                        thresholds_path,
                        text_expectations=_text_expectations(params),
                    )
            except CandidateTimeoutError as exc:
                attempt["error"] = _issue(
                    "E5198_CANDIDATE_TIMEOUT",
                    str(exc),
                    "Increase PNG2SVG_CONVERT_TIMEOUT_SEC for validation.",
                    {"stage": "validate"},
                )
                gate_payload = _gate_skipped_payload(
                    "candidate_timeout", gate_thresholds, attempt["error"]
                )
                attempt["quality_gate"] = gate_payload
                _write_json(gate_report_path, gate_payload)
                results.append(attempt)
                continue
            report_payload = report.to_dict()
            _write_json(report_path, report_payload)
            attempt["validation"] = report_payload
            attempt["validation_warning_count"] = len(report.warnings)
            attempt["validation_error_count"] = len(report.errors)
            if quality_gate:
                try:
                    with _candidate_timeout(timeout_sec):
                        gate_payload = _run_quality_gate(
                            svg_path, input_png, generated_png, diff_png, gate_thresholds
                        )
                except CandidateTimeoutError as exc:
                    attempt["error"] = _issue(
                        "E5198_CANDIDATE_TIMEOUT",
                        str(exc),
                        "Increase PNG2SVG_CONVERT_TIMEOUT_SEC for quality gate.",
                        {"stage": "quality_gate"},
                    )
                    gate_payload = _gate_skipped_payload(
                        "candidate_timeout", gate_thresholds, attempt["error"]
                    )
                attempt["quality_gate"] = gate_payload
                _write_json(gate_report_path, gate_payload)
            else:
                gate_payload = _gate_skipped_payload("quality_gate_disabled", gate_thresholds)
                gate_payload["diff_png"] = str(diff_png) if diff_png else None
                attempt["quality_gate"] = gate_payload
                if candidate_dir is not None:
                    try:
                        backend = rasterize_svg_to_png(svg_path, generated_png)
                        gate_payload["backend"] = backend
                    except RasterizeError as exc:
                        gate_payload["warnings"].append(
                            _issue(
                                "W5201_GATE_RASTERIZE_SKIPPED",
                                f"Quality gate rasterize skipped: {exc}",
                                "Install resvg or cairosvg to render debug previews.",
                            )
                        )
                _write_json(gate_report_path, gate_payload)

            if report.errors:
                results.append(attempt)
                continue
            if quality_gate and gate_payload.get("status") != "pass":
                results.append(attempt)
                continue

            attempt["status"] = "pass"
            results.append(attempt)

        validation_passing = [item for item in results if item.get("validation_error_count") == 0]
        passing = [item for item in results if item.get("status") == "pass"]
        report: dict[str, Any] = {
            "status": "pass" if passing else "fail",
            "output_svg": str(output_svg),
            "classification": classification,
            "classification_error": classification_error,
            "candidates": results,
            "quality_gate": "on" if quality_gate else "off",
            "allow_failed_gate": allow_failed_gate,
            "text_mode": text_mode,
            "errors": [],
            "warnings": [],
        }
        if classification_error:
            report["errors"].append(classification_error)

        if not validation_passing:
            if debug_dir is not None and report_requested:
                _write_json(debug_dir / "convert_report.json", report)
            raise Png2SvgError(
                code="E5105_CONVERT_FAILED",
                message="No candidate templates passed validation.",
                hint="Use --debug-dir to inspect candidate outputs and warnings.",
            )
        if quality_gate and not passing and not allow_failed_gate:
            if debug_dir is not None and report_requested:
                _write_json(debug_dir / "convert_report.json", report)
            raise Png2SvgError(
                code="E5108_QUALITY_GATE_FAILED",
                message="No candidate templates passed the quality gate.",
                hint="Inspect gate_report.json and diff.png under the debug directory.",
            )

        def _metric_or_inf(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("inf")

        def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
            gate = item.get("quality_gate") or {}
            rmse = _metric_or_inf(gate.get("rmse"))
            bad_ratio = _metric_or_inf(gate.get("bad_pixel_ratio"))
            warnings = float(item.get("validation_warning_count", 0))
            score = -float(item.get("score", 0.0))
            if quality_gate:
                return (rmse, bad_ratio, warnings, score)
            return (warnings, rmse, bad_ratio, score)

        if quality_gate and passing:
            selected_pool = passing
        else:
            selected_pool = validation_passing
        selected = sorted(selected_pool, key=_sort_key)[0]
        report["selected_template"] = selected.get("template_id")
        report["selected"] = selected
        report["ocr_stats"] = selected.get("ocr_stats")
        if quality_gate and not passing:
            report["status"] = "fail"
            report["quality_gate_status"] = "fail"
            report["errors"].append(
                _issue(
                    "E5108_QUALITY_GATE_FAILED",
                    "No candidate templates passed the quality gate.",
                    "Inspect gate_report.json and diff.png under the debug directory.",
                )
            )
        elif quality_gate:
            report["quality_gate_status"] = "pass"

        output_svg.parent.mkdir(parents=True, exist_ok=True)
        src_svg = Path(selected["paths"]["svg"])
        if src_svg.exists():
            output_svg.write_bytes(src_svg.read_bytes())

        if debug_dir is not None:
            final_dir = debug_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_svg = final_dir / "out.svg"
            final_svg.write_bytes(src_svg.read_bytes())
            rendered_path = Path(selected["paths"]["rendered_png"])
            if rendered_path.exists():
                final_png = final_dir / "rendered.png"
                final_png.write_bytes(rendered_path.read_bytes())
            gate_report = Path(selected["paths"]["gate_report"])
            if gate_report.exists():
                final_gate = final_dir / "gate_report.json"
                final_gate.write_bytes(gate_report.read_bytes())

        if debug_dir is None:
            for item in report["candidates"]:
                item["paths"] = None

        if debug_dir is not None and report_requested:
            _write_json(debug_dir / "convert_report.json", report)

        return report
