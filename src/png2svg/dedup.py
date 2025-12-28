"""Deduplication utilities for geometric elements."""
from __future__ import annotations

from typing import Any

import numpy as np


def _line_distance(line1: dict[str, Any], line2: dict[str, Any]) -> float:
    """Calculate distance between two lines based on their endpoints.

    Args:
        line1: First line dict with x1, y1, x2, y2
        line2: Second line dict with x1, y1, x2, y2

    Returns:
        Distance metric (0 = identical)
    """
    try:
        # Get endpoints
        x1_1, y1_1 = float(line1["x1"]), float(line1["y1"])
        x2_1, y2_1 = float(line1["x2"]), float(line1["y2"])
        x1_2, y1_2 = float(line2["x1"]), float(line2["y1"])
        x2_2, y2_2 = float(line2["x2"]), float(line2["y2"])
    except (KeyError, TypeError, ValueError):
        return float("inf")

    # Distance between start points + distance between end points
    d1 = ((x1_1 - x1_2) ** 2 + (y1_1 - y1_2) ** 2) ** 0.5
    d2 = ((x2_1 - x2_2) ** 2 + (y2_1 - y2_2) ** 2) ** 0.5

    # Also check if lines are swapped (start of one close to end of other)
    d3 = ((x1_1 - x2_2) ** 2 + (y1_1 - y2_2) ** 2) ** 0.5
    d4 = ((x2_1 - x1_2) ** 2 + (y2_1 - y1_2) ** 2) ** 0.5

    return min(d1 + d2, d3 + d4)


def _line_length(line: dict[str, Any]) -> float:
    """Calculate line length."""
    try:
        x1, y1 = float(line["x1"]), float(line["y1"])
        x2, y2 = float(line["x2"]), float(line["y2"])
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    except (KeyError, TypeError, ValueError):
        return 0.0


def _is_vertical_line(line: dict[str, Any], tolerance: float = 2.0) -> bool:
    """Check if line is vertical."""
    try:
        x1, x2 = float(line["x1"]), float(line["x2"])
        return abs(x1 - x2) <= tolerance
    except (KeyError, TypeError, ValueError):
        return False


def _is_horizontal_line(line: dict[str, Any], tolerance: float = 2.0) -> bool:
    """Check if line is horizontal."""
    try:
        y1, y2 = float(line["y1"]), float(line["y2"])
        return abs(y1 - y2) <= tolerance
    except (KeyError, TypeError, ValueError):
        return False


def deduplicate_lines(
    lines: list[dict[str, Any]],
    distance_threshold: float = 5.0,
) -> list[dict[str, Any]]:
    """Remove duplicate or nearly overlapping lines.

    Uses a greedy clustering approach: for each cluster of nearby lines,
    keep only the longest/best one.

    Args:
        lines: List of line dicts with x1, y1, x2, y2
        distance_threshold: Maximum distance to consider lines as duplicates

    Returns:
        Deduplicated list of lines
    """
    if not lines or len(lines) <= 1:
        return lines

    # Separate by orientation for more accurate comparison
    vertical_lines = [l for l in lines if _is_vertical_line(l)]
    horizontal_lines = [l for l in lines if _is_horizontal_line(l)]
    other_lines = [l for l in lines if not _is_vertical_line(l) and not _is_horizontal_line(l)]

    result = []

    # Deduplicate each category
    for line_group in [vertical_lines, horizontal_lines, other_lines]:
        if not line_group:
            continue

        # Mark which lines are used
        used = [False] * len(line_group)

        for i in range(len(line_group)):
            if used[i]:
                continue

            # Find all lines close to this one
            cluster_indices = [i]
            for j in range(i + 1, len(line_group)):
                if used[j]:
                    continue
                dist = _line_distance(line_group[i], line_group[j])
                if dist <= distance_threshold:
                    cluster_indices.append(j)
                    used[j] = True

            # Keep the longest line in the cluster
            if cluster_indices:
                best_idx = max(cluster_indices, key=lambda idx: _line_length(line_group[idx]))
                result.append(line_group[best_idx])
                used[i] = True

    return result


def _rect_overlap_ratio(rect1: dict[str, Any], rect2: dict[str, Any]) -> float:
    """Calculate overlap ratio between two rectangles.

    Args:
        rect1, rect2: Dicts with x, y, width, height

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    try:
        x1, y1 = float(rect1["x"]), float(rect1["y"])
        w1, h1 = float(rect1["width"]), float(rect1["height"])
        x2, y2 = float(rect2["x"]), float(rect2["y"])
        w2, h2 = float(rect2["width"]), float(rect2["height"])
    except (KeyError, TypeError, ValueError):
        return 0.0

    # Calculate intersection
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)

    if left >= right or top >= bottom:
        return 0.0

    intersection = (right - left) * (bottom - top)
    area1 = w1 * h1
    area2 = w2 * h2

    if area1 <= 0 or area2 <= 0:
        return 0.0

    return intersection / min(area1, area2)


def deduplicate_rects(
    rects: list[dict[str, Any]],
    overlap_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Remove duplicate or overlapping rectangles.

    Args:
        rects: List of rect dicts with x, y, width, height
        overlap_threshold: Minimum overlap ratio to consider duplicates

    Returns:
        Deduplicated list of rectangles
    """
    if not rects or len(rects) <= 1:
        return rects

    # Sort by area (largest first)
    def rect_area(r):
        try:
            return float(r["width"]) * float(r["height"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    sorted_rects = sorted(rects, key=rect_area, reverse=True)

    result = []
    for rect in sorted_rects:
        is_duplicate = False
        for kept in result:
            if _rect_overlap_ratio(rect, kept) >= overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            result.append(rect)

    return result


def deduplicate_circles(
    circles: list[dict[str, Any]],
    distance_threshold: float = 5.0,
) -> list[dict[str, Any]]:
    """Remove duplicate circles.

    Args:
        circles: List of circle dicts with x, y, radius
        distance_threshold: Maximum center distance to consider duplicates

    Returns:
        Deduplicated list of circles
    """
    if not circles or len(circles) <= 1:
        return circles

    result = []
    for circle in circles:
        try:
            cx, cy = float(circle["x"]), float(circle["y"])
            r = float(circle.get("radius", circle.get("r", 0)))
        except (KeyError, TypeError, ValueError):
            result.append(circle)
            continue

        is_duplicate = False
        for kept in result:
            try:
                kx, ky = float(kept["x"]), float(kept["y"])
                kr = float(kept.get("radius", kept.get("r", 0)))
            except (KeyError, TypeError, ValueError):
                continue

            dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
            if dist <= distance_threshold and abs(r - kr) <= distance_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(circle)

    return result


def deduplicate_geometry(params: dict[str, Any]) -> dict[str, Any]:
    """Apply deduplication to all geometric elements in params.

    Args:
        params: Extraction params dict with lines, rects, circles, etc.

    Returns:
        Params with deduplicated geometry
    """
    result = dict(params)

    # Deduplicate lines in panels
    if "panels" in result:
        for panel in result["panels"]:
            if "axis_lines" in panel:
                panel["axis_lines"] = deduplicate_lines(panel["axis_lines"])
            if "t_lines" in panel:
                panel["t_lines"] = deduplicate_lines(panel["t_lines"])
            if "threshold_lines" in panel:
                panel["threshold_lines"] = deduplicate_lines(panel["threshold_lines"])

    # Deduplicate top-level geometry
    if "lines" in result:
        result["lines"] = deduplicate_lines(result["lines"])

    if "geom_lines" in result:
        result["geom_lines"] = deduplicate_lines(result["geom_lines"])

    if "rects" in result:
        result["rects"] = deduplicate_rects(result["rects"])

    if "circles" in result:
        result["circles"] = deduplicate_circles(result["circles"])

    if "markers" in result:
        result["markers"] = deduplicate_circles(result["markers"])

    # Deduplicate nested geometry dict
    if "geometry" in result and isinstance(result["geometry"], dict):
        geom = result["geometry"]
        if "lines" in geom:
            geom["lines"] = deduplicate_lines(geom["lines"])
        if "rects" in geom:
            geom["rects"] = deduplicate_rects(geom["rects"])
        if "markers" in geom:
            geom["markers"] = deduplicate_circles(geom["markers"])
        if "circles" in geom:
            geom["circles"] = deduplicate_circles(geom["circles"])

    # Deduplicate axes lines
    if "axes" in result and isinstance(result["axes"], dict):
        axes = result["axes"]
        if "lines" in axes:
            axes["lines"] = deduplicate_lines(axes["lines"])

    return result
