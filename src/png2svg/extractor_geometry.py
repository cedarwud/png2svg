from __future__ import annotations

from typing import Any

import numpy as np

from png2svg.extractor_math import _smooth_series
from png2svg.extractor_preprocess import _connected_components
from png2svg.arrow_detector import ArrowType, classify_line_arrows
from png2svg.dash_pattern import detect_dash_pattern


def _max_run_length(values: np.ndarray) -> int:
    run = 0
    best = 0
    for value in values:
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _max_runs(mask: np.ndarray, axis: int) -> list[int]:
    runs: list[int] = []
    if axis == 0:
        for col in mask.T:
            runs.append(_max_run_length(col))
    else:
        for row in mask:
            runs.append(_max_run_length(row))
    return runs


def _cluster_indices(indices: list[int]) -> list[int]:
    if not indices:
        return []
    indices = sorted(indices)
    clusters: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx - clusters[-1][-1] <= 3:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [int(sum(cluster) / len(cluster)) for cluster in clusters]


def _long_line_positions(mask: np.ndarray, axis: int, min_len: int) -> list[int]:
    runs = _max_runs(mask, axis=axis)
    positions = [idx for idx, run in enumerate(runs) if run >= min_len]
    return _cluster_indices(positions)


def _find_separator(values: np.ndarray, target: int, span: int) -> int | None:
    start = max(int(target - span), 0)
    end = min(int(target + span), len(values))
    if end <= start:
        return None
    segment = values[start:end]
    if segment.size == 0:
        return None
    offset = int(np.argmin(segment))
    return start + offset


def _detect_panel_columns(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    col_ink = mask.sum(axis=0)
    window = max(int(width * 0.05), 5)
    smoothed = _smooth_series(col_ink, window)
    span = int(width * 0.12)
    sep1 = _find_separator(smoothed, int(width / 3), span)
    sep2 = _find_separator(smoothed, int(width * 2 / 3), span)
    if sep1 is None or sep2 is None:
        return []
    if sep2 - sep1 < int(width * 0.18):
        return []
    return sorted([sep1, sep2])


def _detect_panels(mask: np.ndarray, width: int, height: int) -> list[dict[str, Any]]:
    separators = _detect_panel_columns(mask)
    if not separators:
        return []
    edges = [0, separators[0], separators[1], width]
    panels: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right - left < max(int(width * 0.15), 40):
            return []
        submask = mask[:, left:right]
        row_ink = submask.sum(axis=1)
        row_threshold = max(int((right - left) * 0.02), 4)
        search_top = int(height * 0.15)
        search_bottom = int(height * 0.9)
        top_candidates = np.where(row_ink[search_top:search_bottom] > row_threshold)[0]
        if top_candidates.size > 0:
            top = int(top_candidates[0]) + search_top
        else:
            top = int(height * 0.2)
        bottom_candidates = np.where(row_ink[search_top:search_bottom] > row_threshold)[0]
        if bottom_candidates.size > 0:
            bottom = int(bottom_candidates[-1]) + search_top
        else:
            bottom = int(height * 0.8)
        if bottom - top < int(height * 0.2):
            top = int(height * 0.2)
            bottom = int(height * 0.8)
        margin = max(int((right - left) * 0.04), 10)
        panel = {
            "id": ["A3", "A4", "A5"][idx],
            "label": ["A3", "A4", "A5"][idx],
            "x": float(left + margin),
            "y": float(top),
            "width": float(max(right - left - margin * 2, 20)),
            "height": float(max(bottom - top, 20)),
        }
        panels.append(panel)
    return panels


def _line_extent(mask: np.ndarray, axis: int, idx: int) -> tuple[int, int] | None:
    if axis == 0:
        values = mask[:, idx]
    else:
        values = mask[idx, :]
    positions = np.where(values > 0)[0]
    if positions.size == 0:
        return None
    return int(positions[0]), int(positions[-1])


def _detect_axes_lines(
    mask: np.ndarray, panel: dict[str, Any], adaptive: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    x0 = int(panel["x"])
    y0 = int(panel["y"])
    x1 = int(panel["x"] + panel["width"])
    y1 = int(panel["y"] + panel["height"])
    submask = mask[y0:y1, x0:x1]
    if submask.size == 0:
        return lines
    min_len = max(int(submask.shape[0] * 0.5), 1)
    if adaptive and adaptive.get("lines"):
        min_len = int(adaptive["lines"].get("long_line_min_len_px", min_len))
    v_runs = _max_runs(submask, axis=0)
    h_runs = _max_runs(submask, axis=1)
    v_candidates = [idx for idx, run in enumerate(v_runs) if run >= min_len]
    h_candidates = [idx for idx, run in enumerate(h_runs) if run >= min_len]
    for x in _cluster_indices(v_candidates):
        extent = _line_extent(submask, axis=0, idx=x)
        if not extent:
            continue
        y_start, y_end = extent
        lines.append(
            {
                "x1": float(x0 + x),
                "y1": float(y0 + y_start),
                "x2": float(x0 + x),
                "y2": float(y0 + y_end),
                "stroke": "#000000",
                "stroke_width": 2,
                "role": "axis_y",
            }
        )
    for y in _cluster_indices(h_candidates):
        extent = _line_extent(submask, axis=1, idx=y)
        if not extent:
            continue
        x_start, x_end = extent
        lines.append(
            {
                "x1": float(x0 + x_start),
                "y1": float(y0 + y),
                "x2": float(x0 + x_end),
                "y2": float(y0 + y),
                "stroke": "#000000",
                "stroke_width": 2,
                "role": "axis_x",
            }
        )
    return lines


def _run_lengths(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run = 0
    start = 0
    for idx, value in enumerate(values):
        if value:
            if run == 0:
                start = idx
            run += 1
        else:
            if run > 0:
                runs.append((start, run))
                run = 0
    if run > 0:
        runs.append((start, run))
    return runs


def _dashed_line_candidates(mask: np.ndarray, axis: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if axis == 0:
        for x, col in enumerate(mask.T):
            runs = _run_lengths(col)
            candidates.append({"idx": x, "runs": runs, "total": int(col.sum())})
    else:
        for y, row in enumerate(mask):
            runs = _run_lengths(row)
            candidates.append({"idx": y, "runs": runs, "total": int(row.sum())})
    return candidates


def _detect_dashed_lines(
    mask: np.ndarray, adaptive: dict[str, Any] | None = None, panels: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if adaptive and adaptive.get("dashes"):
        dash_cfg = adaptive["dashes"]
        min_len = int(dash_cfg.get("min_len", 6))
        max_len = int(dash_cfg.get("max_len", 40))
        min_count = int(dash_cfg.get("min_count", 6))
        min_span_ratio = float(dash_cfg.get("min_span_ratio", 0.6))
        min_coverage = float(dash_cfg.get("min_coverage", 0.02))
        max_coverage = float(dash_cfg.get("max_coverage", 0.35))
        min_gap_ratio = float(dash_cfg.get("min_gap_ratio", 0.5))
        max_lines_per_panel = int(dash_cfg.get("max_lines_per_panel", 2))
    else:
        min_len = 6
        max_len = 40
        min_count = 6
        min_span_ratio = 0.6
        min_coverage = 0.02
        max_coverage = 0.35
        min_gap_ratio = 0.5
        max_lines_per_panel = 2
    height, width = mask.shape
    candidates = _dashed_line_candidates(mask, axis=0)
    lines: list[dict[str, Any]] = []
    for candidate in candidates:
        idx = candidate["idx"]
        runs = candidate["runs"]
        valid_runs = [run for run in runs if min_len <= run[1] <= max_len]
        if len(valid_runs) < min_count:
            continue
        span_start = min(run[0] for run in valid_runs)
        span_end = max(run[0] + run[1] for run in valid_runs)
        span = span_end - span_start
        if span < height * min_span_ratio:
            continue
        coverage = sum(run[1] for run in valid_runs) / max(span, 1)
        if coverage < min_coverage or coverage > max_coverage:
            continue
        gaps = []
        for prev, curr in zip(valid_runs, valid_runs[1:]):
            gap = curr[0] - (prev[0] + prev[1])
            gaps.append(gap)
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap <= min_len * min_gap_ratio:
                continue

        # Detect actual dash pattern from runs
        dash_pattern = detect_dash_pattern(valid_runs)
        dasharray = dash_pattern.dasharray if dash_pattern.confidence > 0.3 else [4, 4]

        lines.append(
            {
                "x1": float(idx),
                "y1": float(span_start),
                "x2": float(idx),
                "y2": float(span_end),
                "stroke": "#555555",
                "stroke_width": 1,
                "dasharray": dasharray,
                "role": "t_line",
                "dash_info": {
                    "pattern_type": dash_pattern.pattern_type,
                    "confidence": round(dash_pattern.confidence, 3),
                },
            }
        )

    horizontal_lines: list[dict[str, Any]] = []
    candidates = _dashed_line_candidates(mask, axis=1)
    for candidate in candidates:
        idx = candidate["idx"]
        runs = candidate["runs"]
        valid_runs = [run for run in runs if min_len <= run[1] <= max_len]
        if len(valid_runs) < min_count:
            continue
        span_start = min(run[0] for run in valid_runs)
        span_end = max(run[0] + run[1] for run in valid_runs)
        span = span_end - span_start
        if span < width * min_span_ratio:
            continue
        coverage = sum(run[1] for run in valid_runs) / max(span, 1)
        if coverage < min_coverage or coverage > max_coverage:
            continue
        gaps = []
        for prev, curr in zip(valid_runs, valid_runs[1:]):
            gap = curr[0] - (prev[0] + prev[1])
            gaps.append(gap)
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap <= min_len * min_gap_ratio:
                continue

        # Detect actual dash pattern from runs
        dash_pattern = detect_dash_pattern(valid_runs)
        dasharray = dash_pattern.dasharray if dash_pattern.confidence > 0.3 else [4, 4]

        horizontal_lines.append(
            {
                "x1": float(span_start),
                "y1": float(idx),
                "x2": float(span_end),
                "y2": float(idx),
                "stroke": "#888888",
                "stroke_width": 1,
                "dasharray": dasharray,
                "role": "threshold_line",
                "dash_info": {
                    "pattern_type": dash_pattern.pattern_type,
                    "confidence": round(dash_pattern.confidence, 3),
                },
            }
        )

    if panels:
        filtered: list[dict[str, Any]] = []
        for panel in panels:
            px = panel["x"]
            pw = panel["width"]
            py = panel["y"]
            ph = panel["height"]
            panel_lines = [
                line
                for line in lines
                if px <= float(line["x1"]) <= px + pw and abs(line["x1"] - line["x2"]) <= 1.0
            ]
            panel_lines.sort(key=lambda line: float(line["x1"]))
            filtered.extend(panel_lines[:max_lines_per_panel])

            h_lines = [
                line
                for line in horizontal_lines
                if py <= float(line["y1"]) <= py + ph and abs(line["y1"] - line["y2"]) <= 1.0
            ]
            h_lines.sort(key=lambda line: float(line["y1"]))
            filtered.extend(h_lines[:max_lines_per_panel])
        return filtered
    return lines + horizontal_lines


def _detect_markers(rgba: np.ndarray) -> list[dict[str, Any]]:
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3]
    red = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100) & (alpha > 10)
    components = _connected_components(red, min_area=6)
    markers: list[dict[str, Any]] = []
    for comp in components:
        width = comp["width"]
        height = comp["height"]
        if width > 30 or height > 30:
            continue
        if width < 2 or height < 2:
            continue
        cx = comp["x"] + width / 2
        cy = comp["y"] + height / 2
        radius = max(width, height) / 2
        markers.append(
            {
                "x": float(cx),
                "y": float(cy),
                "radius": float(radius),
                "fill": "#dd6b20",
                "role": "event_marker",
            }
        )
    return markers


def _detect_line_arrows(
    rgba: np.ndarray,
    lines: list[dict[str, Any]],
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect and classify arrow heads on detected lines.

    Args:
        rgba: Source image as RGBA numpy array
        lines: List of line dicts with x1, y1, x2, y2
        min_confidence: Minimum confidence to include arrow info

    Returns:
        Lines with added arrow_start and arrow_end fields
    """
    enhanced_lines: list[dict[str, Any]] = []
    for line in lines:
        try:
            enhanced = classify_line_arrows(rgba, line)
            # Only include arrow info if confidence is high enough
            arrow_start = enhanced.get("arrow_start", {})
            arrow_end = enhanced.get("arrow_end", {})
            if arrow_start.get("confidence", 0) < min_confidence:
                arrow_start["type"] = ArrowType.NONE.value
            if arrow_end.get("confidence", 0) < min_confidence:
                arrow_end["type"] = ArrowType.NONE.value
            enhanced["arrow_start"] = arrow_start
            enhanced["arrow_end"] = arrow_end
            enhanced_lines.append(enhanced)
        except (KeyError, TypeError, ValueError):
            enhanced_lines.append(line)
    return enhanced_lines


def _enhance_axes_with_arrows(
    rgba: np.ndarray,
    lines: list[dict[str, Any]],
    adaptive: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Detect arrows on axis lines.

    For axis lines, arrows typically appear at the end of the axis
    (positive direction).

    Args:
        rgba: Source image
        lines: Detected axis lines
        adaptive: Configuration

    Returns:
        Lines with arrow type information
    """
    min_conf = 0.4
    if adaptive and adaptive.get("arrows"):
        min_conf = float(adaptive["arrows"].get("min_confidence", 0.4))
    return _detect_line_arrows(rgba, lines, min_confidence=min_conf)
