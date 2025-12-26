from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


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


def _has_rasterizer() -> bool:
    if shutil.which("resvg"):
        return True
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        return False
    return True
