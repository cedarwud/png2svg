from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml

from png2svg.ocr import (
    DEFAULT_OCR_MAX_DIM,
    DEFAULT_OCR_TIMEOUT_SEC,
    OcrTimeoutError,
    has_pytesseract,
    has_tesseract,
    ocr_image,
)
from png2svg.extractor_preprocess import _prepare_ocr_image

@dataclass(frozen=True)
class ClassifierThresholds:
    min_confidence: float
    min_margin: float


DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "classifier_thresholds.v1.yaml"
)

TEMPLATES = [
    "t_3gpp_events_3panel",
    "t_procedure_flow",
    "t_performance_lineplot",
    "t_project_architecture_v1",
    "t_rl_agent_loop_v1",
    "t_performance_grid_v1",
]

RL_KEYWORDS = {
    "agent",
    "environment",
    "state",
    "action",
    "reward",
    "policy",
    "value",
    "qnetwork",
    "q-network",
    "critic",
    "actor",
    "buffer",
    "replay",
    "constraint",
    "lagrangian",
    "multiobjective",
    "morl",
}

GRID_KEYWORDS = {
    "cdf",
    "latency",
    "throughput",
    "rmse",
    "packet",
    "handover",
    "rlf",
    "pingpong",
    "ping-pong",
    "uho",
    "energy",
    "efficiency",
    "kpi",
}


def _load_image(path: Path) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    height, width = arr.shape[0], arr.shape[1]
    return arr, width, height


def _ink_mask(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return (alpha > 10) & (luminance < 245)


def _color_count(rgba: np.ndarray, mask: np.ndarray | None = None, colors: int = 16) -> int:
    rgb = rgba[:, :, :3]
    if mask is not None:
        if not np.any(mask):
            return 0
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        rgb = rgb[y0:y1, x0:x1].copy()
        sub_mask = mask[y0:y1, x0:x1]
        rgb[~sub_mask] = 255
    if rgb.size == 0:
        return 0
    try:
        image = Image.fromarray(rgb, mode="RGB")
        quant = image.quantize(colors=colors, method=Image.MEDIANCUT)
        indices = np.asarray(quant)
        if mask is not None:
            indices = indices[sub_mask]
        return int(np.unique(indices).size)
    except Exception:
        flat = rgb.reshape(-1, 3)
        quantized = (flat // 32).astype(np.uint8)
        packed = (
            (quantized[:, 0].astype(np.int32) << 16)
            | (quantized[:, 1].astype(np.int32) << 8)
            | quantized[:, 2].astype(np.int32)
        )
        return int(np.unique(packed).size)


def _ocr_tokens(rgba: np.ndarray, width: int, height: int) -> list[str]:
    if not (has_pytesseract() or has_tesseract()):
        return []
    ocr_image_input = _prepare_ocr_image(rgba)
    rois = [
        {"id": "header", "x": 0, "y": 0, "width": width, "height": int(height * 0.2)},
        {
            "id": "middle",
            "x": 0,
            "y": int(height * 0.18),
            "width": width,
            "height": int(height * 0.35),
        },
        {
            "id": "bottom",
            "x": 0,
            "y": int(height * 0.55),
            "width": width,
            "height": int(height * 0.4),
        },
    ]
    try:
        results = ocr_image(
            ocr_image_input,
            backend="auto",
            rois=rois,
            timeout_sec=DEFAULT_OCR_TIMEOUT_SEC,
            max_dim=DEFAULT_OCR_MAX_DIM,
        )
    except (ValueError, OcrTimeoutError):
        return []
    tokens: list[str] = []
    for item in results:
        text = str(item.get("text") or "").strip().lower()
        if not text:
            continue
        cleaned = "".join(ch if ch.isalnum() else " " for ch in text)
        tokens.extend(token for token in cleaned.split() if token)
    return tokens


def _keyword_hits(tokens: list[str], keywords: set[str]) -> int:
    if not tokens:
        return 0
    joined = " ".join(tokens)
    hits = 0
    for keyword in keywords:
        if keyword in joined:
            hits += 1
    return hits


def _project_architecture_score(
    width: int,
    height: int,
    color_count: int,
    tokens: list[str],
    ink_ratio: float,
) -> float:
    if height <= 0:
        return 0.0
    aspect = width / height
    if not (1.55 <= aspect <= 2.05 and width >= 1200 and height >= 700):
        return 0.0
    score = 0.0
    if 1.6 <= aspect <= 2.0:
        score += 1.0
    if width >= 1400 and height >= 800:
        score += 1.0
    if color_count <= 30:
        score += 1.0
    elif color_count <= 80:
        score += 0.8
    elif color_count <= 200:
        score += 0.5

    token_set = set(tokens)
    token_count = len(tokens)
    joined = " ".join(tokens)
    if "project" in token_set and "architecture" in token_set:
        score += 1.6
    if "work" in token_set and ("packages" in token_set or "package" in token_set):
        score += 1.2
    for wp in ("wp1", "wp2", "wp3", "wp4"):
        if wp in joined:
            score += 0.4
    for panel in ("panel", "panela", "panelb", "panelc"):
        if panel in token_set:
            score += 0.2
    if token_count >= 30:
        score += 1.0
    if ink_ratio < 0.18:
        score *= 0.5
    return max(score, 0.0)


def _saturation_ratio(rgba: np.ndarray, mask: np.ndarray) -> float:
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    saturated = mask & (saturation > 0.25) & (max_rgb > 0.2)
    denom = max(int(mask.sum()), 1)
    return float(saturated.sum()) / denom


def _max_run_length(values: np.ndarray) -> int:
    max_run = 0
    run = 0
    for value in values:
        if value:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


def _max_runs(mask: np.ndarray, axis: int) -> list[int]:
    runs: list[int] = []
    if axis == 0:
        for col in range(mask.shape[1]):
            runs.append(_max_run_length(mask[:, col]))
    else:
        for row in range(mask.shape[0]):
            runs.append(_max_run_length(mask[row, :]))
    return runs


def _cluster_indices(indices: list[int]) -> list[int]:
    if not indices:
        return []
    clusters: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx <= clusters[-1][-1] + 1:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    centers: list[int] = []
    for cluster in clusters:
        centers.append(int(round(sum(cluster) / len(cluster))))
    return centers


def _count_segments(mask: np.ndarray, axis: int, min_len: int, max_len: int) -> int:
    total = 0
    if axis == 0:
        for col in range(mask.shape[1]):
            run = 0
            for value in mask[:, col]:
                if value:
                    run += 1
                else:
                    if min_len <= run <= max_len:
                        total += 1
                    run = 0
            if min_len <= run <= max_len:
                total += 1
    else:
        for row in range(mask.shape[0]):
            run = 0
            for value in mask[row, :]:
                if value:
                    run += 1
                else:
                    if min_len <= run <= max_len:
                        total += 1
                    run = 0
            if min_len <= run <= max_len:
                total += 1
    return total


def _axis_aligned_ratio(gray: np.ndarray, mask: np.ndarray) -> float:
    norm = gray.astype(np.float32) / 255.0
    dx = np.abs(norm[:, 1:] - norm[:, :-1])
    dy = np.abs(norm[1:, :] - norm[:-1, :])
    dx = dx[:-1, :]
    dy = dy[:, :-1]
    edge_strength = dx + dy
    edge_mask = (edge_strength > 0.08) & mask[:-1, :-1]
    if not edge_mask.any():
        return 0.0
    vertical = (dx > dy * 2) & edge_mask
    horizontal = (dy > dx * 2) & edge_mask
    aligned = vertical.sum() + horizontal.sum()
    return float(aligned) / float(edge_mask.sum())


def _compute_features(rgba: np.ndarray, width: int, height: int) -> dict[str, Any]:
    mask = _ink_mask(rgba)
    ink_ratio = float(mask.sum()) / float(mask.size) if mask.size else 0.0
    saturated_ratio = _saturation_ratio(rgba, mask)
    gray = rgba[:, :, :3].mean(axis=2)
    axis_aligned_ratio = _axis_aligned_ratio(gray, mask)
    color_count = _color_count(rgba, mask)

    long_v_min = max(int(height * 0.45), 1)
    long_h_min = max(int(width * 0.45), 1)
    v_runs = _max_runs(mask, axis=0)
    h_runs = _max_runs(mask, axis=1)
    long_v_indices = [idx for idx, run in enumerate(v_runs) if run >= long_v_min]
    long_h_indices = [idx for idx, run in enumerate(h_runs) if run >= long_h_min]
    long_v_lines = _cluster_indices(long_v_indices)
    long_h_lines = _cluster_indices(long_h_indices)

    short_v = _count_segments(mask, axis=0, min_len=2, max_len=max(int(height * 0.08), 2))
    short_h = _count_segments(mask, axis=1, min_len=2, max_len=max(int(width * 0.08), 2))

    features: dict[str, Any] = {
        "width": width,
        "height": height,
        "ink_ratio": ink_ratio,
        "saturated_ratio": saturated_ratio,
        "axis_aligned_ratio": axis_aligned_ratio,
        "long_vertical_lines": len(long_v_lines),
        "long_horizontal_lines": len(long_h_lines),
        "short_vertical_segments": short_v,
        "short_horizontal_segments": short_h,
        "color_count": color_count,
        "thresholds": {
            "long_v_min": long_v_min,
            "long_h_min": long_h_min,
        },
        "line_positions": {
            "vertical": long_v_lines,
            "horizontal": long_h_lines,
        },
    }
    return features


def _score_templates(features: dict[str, Any]) -> dict[str, float]:
    long_v = features["long_vertical_lines"]
    long_h = features["long_horizontal_lines"]
    saturated = features["saturated_ratio"]
    axis_ratio = features["axis_aligned_ratio"]
    short_total = features["short_vertical_segments"] + features["short_horizontal_segments"]
    width = int(features.get("width", 0))
    height = int(features.get("height", 0))
    color_count = int(features.get("color_count", 0))
    ocr_tokens = features.get("ocr_tokens") or []
    if not isinstance(ocr_tokens, list):
        ocr_tokens = []
    rl_hits = _keyword_hits(ocr_tokens, RL_KEYWORDS)
    grid_hits = _keyword_hits(ocr_tokens, GRID_KEYWORDS)
    ocr_token_count = len(ocr_tokens)

    score_3gpp = 0.0
    score_3gpp += min(long_v, 8) * 0.4
    if long_v >= 4:
        score_3gpp += 1.0
    if saturated >= 0.02:
        score_3gpp += 0.2
    if long_v >= 4 and ocr_token_count >= 6:
        score_3gpp += 0.6
    elif ocr_token_count >= 12:
        score_3gpp += 0.4
    if long_v >= 6 and long_h <= 1 and axis_ratio >= 0.7:
        score_3gpp += 0.8
        if ocr_token_count >= 12:
            score_3gpp += 0.4
        if short_total >= 8000:
            score_3gpp += 0.6

    score_lineplot = 0.0
    if long_v >= 1:
        score_lineplot += 1.0
    if long_h >= 1:
        score_lineplot += 1.0
    if saturated >= 0.02:
        score_lineplot += 0.4
    if short_total >= 8:
        score_lineplot += 0.2
    if long_v >= 4:
        score_lineplot -= 0.6
    if saturated < 0.1:
        score_lineplot -= 0.6
    if features["ink_ratio"] > 0.2:
        score_lineplot -= 0.6
    if long_v >= 1 and long_h >= 1 and short_total <= 5000:
        score_lineplot += 0.4
    if long_v >= 1 and long_h >= 1 and short_total >= 8000:
        score_lineplot -= 0.6

    score_flow = 0.0
    if axis_ratio >= 0.7:
        score_flow += 0.7
    if saturated < 0.02:
        score_flow += 0.6
    if long_v == 0:
        score_flow += 0.3
    if long_h <= 1:
        score_flow += 0.2
    if long_h >= 2 and features["ink_ratio"] < 0.2:
        score_flow -= 0.4
    if features["ink_ratio"] > 0.2 and color_count <= 5:
        score_flow -= 0.6
    if long_h >= 2 and features["ink_ratio"] >= 0.3:
        score_flow += 0.4
    if long_h >= 3 and axis_ratio >= 0.8 and saturated < 0.05:
        score_flow += 1.0
    if long_v == 0 and long_h == 0 and axis_ratio >= 0.75 and color_count > 10:
        score_flow += 0.6

    score_project = _project_architecture_score(
        width, height, color_count, ocr_tokens, features["ink_ratio"]
    )
    if long_v >= 6 and long_h <= 1 and axis_ratio >= 0.7:
        score_project -= 0.8

    score_rl = 0.0
    if rl_hits >= 2:
        score_rl += 2.2
    elif rl_hits == 1:
        score_rl += 1.0
    if axis_ratio >= 0.6:
        score_rl += 0.3
    if color_count <= 80:
        score_rl += 0.2
    if saturated >= 0.02:
        score_rl += 0.4
    if features["ink_ratio"] >= 0.08:
        score_rl += 1.0
    elif features["ink_ratio"] >= 0.05:
        score_rl += 0.6
    if long_h >= 2 and long_v == 0 and features["ink_ratio"] >= 0.08:
        score_rl += 1.2
    if long_v >= 1:
        score_rl -= 0.4
    if long_h < 1:
        score_rl -= 0.6
    if color_count <= 5:
        score_rl -= 0.8

    score_grid = 0.0
    if long_v >= 1:
        score_grid += 1.2
        if long_v >= 2:
            score_grid += 0.3
        if long_h >= 1:
            score_grid += 0.4
    if axis_ratio >= 0.6:
        score_grid += 0.2
    if grid_hits >= 1:
        score_grid += 0.8
    if features["ink_ratio"] > 0.2:
        score_grid -= 0.6
    if long_v >= 1 and short_total >= 8000:
        score_grid += 1.2

    return {
        "t_3gpp_events_3panel": score_3gpp,
        "t_performance_lineplot": score_lineplot,
        "t_procedure_flow": score_flow,
        "t_project_architecture_v1": score_project,
        "t_rl_agent_loop_v1": score_rl,
        "t_performance_grid_v1": score_grid,
    }


def _confidence(scores: dict[str, float], top_id: str) -> float:
    positive_scores = [value for value in scores.values() if value > 0]
    min_score = min(positive_scores) if positive_scores else min(scores.values())
    shifted = {key: max(value - min_score, 0.0) for key, value in scores.items()}
    total = sum(shifted.values())
    if total <= 0:
        return 1.0 / len(scores)
    return shifted[top_id] / total


def _evidence_scale(features: dict[str, Any], top_id: str) -> float:
    scale = 1.0
    if top_id == "t_performance_lineplot":
        if features["long_vertical_lines"] < 1 or features["long_horizontal_lines"] < 1:
            scale *= 0.6
        if features["ink_ratio"] > 0.2:
            scale *= 0.4
        if features["saturated_ratio"] < 0.2:
            scale *= 0.6
        if (
            features["long_vertical_lines"] >= 1
            and features["long_horizontal_lines"] >= 1
            and features["saturated_ratio"] >= 0.2
            and features["axis_aligned_ratio"] >= 0.7
            and (features["short_vertical_segments"] + features["short_horizontal_segments"]) <= 7000
        ):
            scale *= 1.1
    elif top_id == "t_3gpp_events_3panel":
        if features["long_vertical_lines"] < 4:
            scale *= 0.4
        if features["long_horizontal_lines"] > 1:
            scale *= 0.6
        if features["saturated_ratio"] > 0.4:
            scale *= 0.6
        if (
            features["long_vertical_lines"] >= 4
            and features.get("ocr_token_count", 0) >= 6
            and features["axis_aligned_ratio"] >= 0.7
        ):
            scale *= 1.2
    elif top_id == "t_procedure_flow":
        if features["axis_aligned_ratio"] < 0.7:
            scale *= 0.6
        if features["axis_aligned_ratio"] > 0.97:
            scale *= 0.6
        if features["saturated_ratio"] > 0.35:
            scale *= 0.4
        if features["ink_ratio"] > 0.2 and features.get("color_count", 0) <= 5:
            scale *= 0.4
        if (
            features.get("color_count", 0) > 20
            and features["ink_ratio"] < 0.1
            and features["long_vertical_lines"] == 0
            and features["long_horizontal_lines"] == 0
        ):
            scale *= 1.3
        if (
            features.get("color_count", 0) > 20
            and features["ink_ratio"] >= 0.3
            and features["long_horizontal_lines"] >= 3
        ):
            scale *= 1.6
        if features["long_horizontal_lines"] >= 3 and features["ink_ratio"] >= 0.3:
            scale *= 1.3
        if (
            features["long_vertical_lines"] == 0
            and features["long_horizontal_lines"] == 0
            and features.get("color_count", 0) > 10
            and features["axis_aligned_ratio"] >= 0.75
        ):
            scale *= 1.2
    elif top_id == "t_project_architecture_v1":
        color_count = int(features.get("color_count", 0))
        width = int(features.get("width", 0))
        height = int(features.get("height", 0))
        if height <= 0 or width <= 0:
            scale *= 0.6
        aspect = width / height if height else 0.0
        if aspect < 1.5 or aspect > 2.1:
            scale *= 0.6
        if color_count > 220:
            scale *= 0.6
    elif top_id == "t_rl_agent_loop_v1":
        if features.get("ocr_tokens"):
            scale *= 1.0
        elif features["axis_aligned_ratio"] < 0.6:
            scale *= 0.6
    elif top_id == "t_performance_grid_v1":
        if features["long_vertical_lines"] < 1 and features["long_horizontal_lines"] < 1:
            scale *= 0.6
    return scale


def _load_thresholds(path: Path | None) -> ClassifierThresholds:
    thresholds_path = path or DEFAULT_THRESHOLDS_PATH
    data: dict[str, Any] = {}
    if thresholds_path.exists():
        loaded = yaml.safe_load(thresholds_path.read_text())
        if isinstance(loaded, dict):
            data = loaded
    decision = data.get("decision", {}) or {}
    min_confidence = float(decision.get("min_confidence", 0.55))
    min_margin = float(decision.get("min_margin", 0.45))
    return ClassifierThresholds(min_confidence=min_confidence, min_margin=min_margin)


def _write_debug(
    rgba: np.ndarray,
    debug_dir: Path,
    features: dict[str, Any],
    scores: dict[str, float],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = overlay.size
    for x in features["line_positions"]["vertical"]:
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0, 128), width=1)
    for y in features["line_positions"]["horizontal"]:
        draw.line([(0, y), (width, y)], fill=(0, 255, 0, 128), width=1)
    overlay.save(debug_dir / "overlay.png")

    payload = {
        "features": features,
        "scores": scores,
    }
    (debug_dir / "features.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def classify_png(
    input_png: Path,
    debug_dir: Path | None = None,
    thresholds_path: Path | None = None,
) -> dict[str, Any]:
    rgba, width, height = _load_image(input_png)
    features = _compute_features(rgba, width, height)
    ocr_tokens: list[str] = []
    if width >= 600 and height >= 300:
        ocr_tokens = _ocr_tokens(rgba, width, height)
    features["ocr_tokens"] = ocr_tokens
    scores = _score_templates(features)
    candidates = sorted(
        (
            {"template_id": template_id, "score": score}
            for template_id, score in scores.items()
        ),
        key=lambda item: (-item["score"], item["template_id"]),
    )
    top_id = candidates[0]["template_id"]
    confidence = _confidence(scores, top_id)
    confidence *= _evidence_scale(features, top_id)
    top_score = float(candidates[0]["score"])
    second_score = float(candidates[1]["score"]) if len(candidates) > 1 else top_score
    margin = top_score - second_score
    thresholds = _load_thresholds(thresholds_path)
    reason_codes: list[str] = []
    if confidence < thresholds.min_confidence:
        reason_codes.append("LOW_CONFIDENCE")
    if margin < thresholds.min_margin:
        reason_codes.append("AMBIGUOUS_MARGIN")
    decision = "unknown" if reason_codes else "known"
    template_id = "unknown" if decision == "unknown" else top_id
    result = {
        "template_id": template_id,
        "decision": decision,
        "reason_codes": reason_codes,
        "confidence": confidence,
        "candidate_templates": candidates,
        "image_meta": {"width": width, "height": height},
        "features_summary": {
            "ink_ratio": features["ink_ratio"],
            "saturated_ratio": features["saturated_ratio"],
            "axis_aligned_ratio": features["axis_aligned_ratio"],
            "long_vertical_lines": features["long_vertical_lines"],
            "long_horizontal_lines": features["long_horizontal_lines"],
            "short_vertical_segments": features["short_vertical_segments"],
            "short_horizontal_segments": features["short_horizontal_segments"],
            "color_count": features["color_count"],
            "ocr_token_count": len(ocr_tokens),
        },
    }
    if debug_dir is not None:
        _write_debug(rgba, debug_dir, features, scores)
    return result
