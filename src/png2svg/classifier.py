from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

TEMPLATES = [
    "t_3gpp_events_3panel",
    "t_procedure_flow",
    "t_performance_lineplot",
]


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
        "ink_ratio": ink_ratio,
        "saturated_ratio": saturated_ratio,
        "axis_aligned_ratio": axis_aligned_ratio,
        "long_vertical_lines": len(long_v_lines),
        "long_horizontal_lines": len(long_h_lines),
        "short_vertical_segments": short_v,
        "short_horizontal_segments": short_h,
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

    score_3gpp = 0.0
    score_3gpp += min(long_v, 8) * 0.4
    if long_v >= 4:
        score_3gpp += 1.0
    if saturated >= 0.02:
        score_3gpp += 0.2

    score_lineplot = 0.0
    if long_v >= 1:
        score_lineplot += 0.8
    if long_h >= 1:
        score_lineplot += 0.8
    if saturated >= 0.02:
        score_lineplot += 0.3
    if short_total >= 8:
        score_lineplot += 0.2
    if long_v >= 4:
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

    return {
        "t_3gpp_events_3panel": score_3gpp,
        "t_performance_lineplot": score_lineplot,
        "t_procedure_flow": score_flow,
    }


def _confidence(scores: dict[str, float], top_id: str) -> float:
    min_score = min(scores.values())
    shifted = {key: value - min_score for key, value in scores.items()}
    total = sum(shifted.values())
    if total <= 0:
        return 1.0 / len(scores)
    return shifted[top_id] / total


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


def classify_png(input_png: Path, debug_dir: Path | None = None) -> dict[str, Any]:
    rgba, width, height = _load_image(input_png)
    features = _compute_features(rgba, width, height)
    scores = _score_templates(features)
    candidates = sorted(
        (
            {"template_id": template_id, "score": score}
            for template_id, score in scores.items()
        ),
        key=lambda item: (-item["score"], item["template_id"]),
    )
    top_id = candidates[0]["template_id"]
    result = {
        "template_id": top_id,
        "confidence": _confidence(scores, top_id),
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
        },
    }
    if debug_dir is not None:
        _write_debug(rgba, debug_dir, features, scores)
    return result
