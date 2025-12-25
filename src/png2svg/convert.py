from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from common.png_utils import has_png_magic
from png2svg.classifier import classify_png
from png2svg.errors import Png2SvgError
from png2svg.extractor import extract_skeleton
from png2svg.renderer import render_svg
from validators.config import load_thresholds, load_visual_diff_thresholds
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


def _candidate_dir(debug_dir: Path | None, index: int, template_id: str) -> Path | None:
    if debug_dir is None:
        return None
    name = f"candidate_{index:02d}_{template_id}"
    path = debug_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_visual_diff(
    svg_path: Path,
    input_png: Path,
    output_png: Path,
    diff_png: Path | None,
    thresholds_path: Path,
) -> dict[str, Any]:
    thresholds = load_thresholds(thresholds_path)
    visual_thresholds = load_visual_diff_thresholds(thresholds)
    backend = rasterize_svg_to_png(svg_path, output_png)
    metrics = compute_visual_diff(output_png, input_png, visual_thresholds.pixel_tolerance)
    payload = {
        "backend": backend,
        "rmse": metrics.rmse,
        "bad_pixel_ratio": metrics.bad_pixel_ratio,
        "pixel_tolerance": visual_thresholds.pixel_tolerance,
        "rmse_max": visual_thresholds.rmse_max,
        "bad_pixel_ratio_max": visual_thresholds.bad_pixel_ratio_max,
    }
    if diff_png is not None and (
        metrics.rmse > visual_thresholds.rmse_max
        or metrics.bad_pixel_ratio > visual_thresholds.bad_pixel_ratio_max
    ):
        try:
            write_diff_image(output_png, input_png, diff_png, visual_thresholds.pixel_tolerance)
        except Exception:
            pass
    return payload


def convert_png(
    input_png: Path,
    output_svg: Path,
    debug_dir: Path | None = None,
    topk: int = 2,
    contract_path: Path | None = None,
    thresholds_path: Path | None = None,
    enable_visual_diff: bool = True,
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

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    classify_debug = debug_dir / "classify" if debug_dir is not None else None
    try:
        classification = classify_png(input_png, debug_dir=classify_debug)
    except Exception as exc:  # noqa: BLE001
        raise Png2SvgError(
            code="E5106_CLASSIFY_FAILED",
            message=f"Classification failed: {exc}",
            hint="Ensure the input PNG can be decoded by Pillow.",
        ) from exc
    if debug_dir is not None:
        _write_json(debug_dir / "classification.json", classification)

    candidates = classification.get("candidate_templates", [])
    candidate_list = candidates[: min(topk, len(candidates))]
    if not candidate_list:
        raise Png2SvgError(
            code="E5104_NO_CANDIDATES",
            message="Classifier returned no candidates.",
            hint="Check input PNG and classifier rules.",
        )

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for idx, candidate in enumerate(candidate_list, start=1):
            template_id = str(candidate.get("template_id"))
            score = float(candidate.get("score", 0.0))
            candidate_dir = _candidate_dir(debug_dir, idx, template_id)
            params_path = (
                candidate_dir / "params.json" if candidate_dir else temp_root / f"{idx}.json"
            )
            svg_path = (
                candidate_dir / "generated.svg" if candidate_dir else temp_root / f"{idx}.svg"
            )
            report_path = (
                candidate_dir / "validate_report.json"
                if candidate_dir
                else temp_root / f"{idx}_report.json"
            )
            generated_png = (
                candidate_dir / "generated.png" if candidate_dir else temp_root / f"{idx}.png"
            )
            diff_png = candidate_dir / "diff.png" if candidate_dir else None

            attempt: dict[str, Any] = {
                "template_id": template_id,
                "score": score,
                "status": "fail",
                "validation_warning_count": 0,
                "validation_error_count": 0,
                "paths": {
                    "params": str(params_path),
                    "svg": str(svg_path),
                    "validate_report": str(report_path),
                    "generated_png": str(generated_png),
                    "diff_png": str(diff_png) if diff_png else None,
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
                params = extract_skeleton(input_png, template_id, extract_debug)
            except Png2SvgError as exc:
                attempt["error"] = _issue(exc.code, exc.message, exc.hint, {"stage": "extract"})
                results.append(attempt)
                continue
            params_path.write_text(json.dumps(params, indent=2, sort_keys=True))

            try:
                render_svg(input_png, params_path, svg_path)
            except Png2SvgError as exc:
                attempt["error"] = _issue(exc.code, exc.message, exc.hint, {"stage": "render"})
                results.append(attempt)
                continue
            except Exception as exc:  # noqa: BLE001
                attempt["error"] = _issue(
                    "E5199_RENDER_FAILED",
                    f"{exc}",
                    "Check input PNG and extracted params.",
                    {"stage": "render"},
                )
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

            report = validate_svg(svg_path, contract_path, thresholds_path)
            report_payload = report.to_dict()
            _write_json(report_path, report_payload)
            attempt["validation"] = report_payload
            attempt["validation_warning_count"] = len(report.warnings)
            attempt["validation_error_count"] = len(report.errors)
            if report.errors:
                results.append(attempt)
                continue

            attempt["status"] = "pass"
            if enable_visual_diff:
                try:
                    visual_payload = _run_visual_diff(
                        svg_path, input_png, generated_png, diff_png, thresholds_path
                    )
                    attempt["visual_diff"] = visual_payload
                except (RasterizeError, DiffError, OSError) as exc:
                    attempt["visual_diff_error"] = _issue(
                        "W5101_VISUAL_DIFF_SKIPPED",
                        f"Visual diff skipped: {exc}",
                        "Install resvg or cairosvg for rasterization.",
                        {"stage": "visual_diff"},
                    )
            results.append(attempt)

        passing = [item for item in results if item.get("status") == "pass"]
        report: dict[str, Any] = {
            "status": "pass" if passing else "fail",
            "output_svg": str(output_svg),
            "classification": classification,
            "candidates": results,
        }

        if not passing:
            if debug_dir is not None:
                _write_json(debug_dir / "convert_report.json", report)
            raise Png2SvgError(
                code="E5105_CONVERT_FAILED",
                message="No candidate templates passed validation.",
                hint="Use --debug-dir to inspect candidate outputs and warnings.",
            )

        def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
            warnings = float(item.get("validation_warning_count", 0))
            visual = item.get("visual_diff") or {}
            rmse = float(visual.get("rmse", float("inf")))
            bad_ratio = float(visual.get("bad_pixel_ratio", float("inf")))
            score = -float(item.get("score", 0.0))
            return (warnings, rmse, bad_ratio, score)

        selected = sorted(passing, key=_sort_key)[0]
        report["selected_template"] = selected.get("template_id")
        report["selected"] = selected

        output_svg.parent.mkdir(parents=True, exist_ok=True)
        src_svg = Path(selected["paths"]["svg"])
        if src_svg.exists():
            output_svg.write_bytes(src_svg.read_bytes())

        if debug_dir is None:
            for item in report["candidates"]:
                item["paths"] = None

        if debug_dir is not None:
            _write_json(debug_dir / "convert_report.json", report)

        return report
