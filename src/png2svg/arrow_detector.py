"""Arrow head detection and classification module.

Detects and classifies arrow head types (triangle, line, dot, open) from images.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ArrowType(Enum):
    """Types of arrow heads."""
    TRIANGLE = "triangle"      # Filled triangle (most common)
    TRIANGLE_OPEN = "open"     # Open/unfilled triangle
    LINE = "line"              # Simple line arrows (< or >)
    DOT = "dot"                # Circle/dot at endpoint
    DIAMOND = "diamond"        # Diamond shape
    NONE = "none"              # No arrow head


@dataclass
class ArrowHead:
    """Detected arrow head information."""
    arrow_type: ArrowType
    position: tuple[float, float]  # (x, y) position
    direction: float               # Angle in degrees (0 = right, 90 = down)
    size: float                    # Size in pixels
    confidence: float              # Detection confidence (0-1)


def _crop_region(
    image: np.ndarray,
    center: tuple[float, float],
    size: int,
) -> np.ndarray | None:
    """Crop a square region around a point."""
    h, w = image.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    half = size // 2

    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def _compute_ink_mask(region: np.ndarray, threshold: int = 200) -> np.ndarray:
    """Create binary mask of dark pixels (ink)."""
    if region.ndim == 3:
        if region.shape[2] == 4:  # RGBA
            gray = np.mean(region[:, :, :3], axis=2)
            alpha = region[:, :, 3]
            mask = (gray < threshold) & (alpha > 10)
        else:  # RGB
            gray = np.mean(region, axis=2)
            mask = gray < threshold
    else:
        mask = region < threshold
    return mask.astype(np.uint8)


def _analyze_shape(mask: np.ndarray) -> dict[str, Any]:
    """Analyze shape properties of a binary mask."""
    if mask.sum() < 4:
        return {"area": 0, "aspect": 1.0, "solidity": 0.0, "vertices": 0}

    # Find contour points
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return {"area": len(xs), "aspect": 1.0, "solidity": 0.0, "vertices": 0}

    area = len(xs)

    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    aspect = width / height

    # Solidity (area / convex hull area approximation)
    bbox_area = width * height
    solidity = area / bbox_area if bbox_area > 0 else 0

    # Estimate vertices by analyzing boundary changes
    # Simple approach: count direction changes in boundary
    boundary_points = []
    for y in range(mask.shape[0]):
        row = mask[y, :]
        if row.sum() > 0:
            xs_row = np.where(row > 0)[0]
            boundary_points.append((xs_row[0], y))
            if len(xs_row) > 1:
                boundary_points.append((xs_row[-1], y))

    vertices = _estimate_vertices(boundary_points)

    return {
        "area": area,
        "aspect": aspect,
        "solidity": solidity,
        "vertices": vertices,
        "width": width,
        "height": height,
    }


def _estimate_vertices(points: list[tuple[int, int]]) -> int:
    """Estimate number of vertices from boundary points."""
    if len(points) < 3:
        return len(points)

    # Use Douglas-Peucker-like approach to count significant corners
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 4:
        return len(pts)

    # Calculate angles between consecutive segments
    angles = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 > 0.1 and norm2 > 0.1:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)

    # Count significant angle changes (potential vertices)
    vertices = sum(1 for a in angles if a > 30) + 2  # +2 for endpoints
    return min(vertices, 8)  # Cap at reasonable number


def detect_arrow_type(
    image: np.ndarray,
    endpoint: tuple[float, float],
    direction_hint: float | None = None,
    region_size: int = 24,
) -> ArrowHead:
    """Detect arrow type at a line endpoint.

    Args:
        image: Source image (RGB/RGBA numpy array)
        endpoint: (x, y) position of line endpoint
        direction_hint: Expected arrow direction in degrees (optional)
        region_size: Size of region to analyze

    Returns:
        ArrowHead with detected type and properties
    """
    region = _crop_region(image, endpoint, region_size)
    if region is None:
        return ArrowHead(
            arrow_type=ArrowType.NONE,
            position=endpoint,
            direction=direction_hint or 0,
            size=0,
            confidence=0,
        )

    mask = _compute_ink_mask(region)
    shape = _analyze_shape(mask)

    if shape["area"] < 6:
        return ArrowHead(
            arrow_type=ArrowType.NONE,
            position=endpoint,
            direction=direction_hint or 0,
            size=0,
            confidence=0.9,
        )

    # Classify based on shape properties
    arrow_type = ArrowType.TRIANGLE
    confidence = 0.5

    area = shape["area"]
    aspect = shape["aspect"]
    solidity = shape["solidity"]
    vertices = shape["vertices"]

    # Triangle detection
    if vertices == 3 or (vertices in [3, 4] and solidity > 0.4):
        if solidity > 0.6:
            arrow_type = ArrowType.TRIANGLE
            confidence = 0.8 + 0.1 * solidity
        else:
            arrow_type = ArrowType.TRIANGLE_OPEN
            confidence = 0.7

    # Circle/dot detection
    elif 0.8 < aspect < 1.2 and solidity > 0.7:
        arrow_type = ArrowType.DOT
        confidence = 0.75

    # Line arrow detection (< or > shape)
    elif vertices == 2 or (solidity < 0.3 and aspect > 1.5):
        arrow_type = ArrowType.LINE
        confidence = 0.6

    # Diamond detection
    elif vertices == 4 and 0.7 < aspect < 1.3 and solidity > 0.5:
        arrow_type = ArrowType.DIAMOND
        confidence = 0.65

    # Estimate size from area
    size = (area ** 0.5) * 1.5

    return ArrowHead(
        arrow_type=arrow_type,
        position=endpoint,
        direction=direction_hint or 0,
        size=size,
        confidence=confidence,
    )


def classify_line_arrows(
    image: np.ndarray,
    line: dict[str, Any],
) -> dict[str, Any]:
    """Classify arrow heads on both ends of a line.

    Args:
        image: Source image
        line: Line dict with x1, y1, x2, y2

    Returns:
        Line dict with added arrow_start and arrow_end fields
    """
    try:
        x1, y1 = float(line["x1"]), float(line["y1"])
        x2, y2 = float(line["x2"]), float(line["y2"])
    except (KeyError, TypeError, ValueError):
        return line

    result = dict(line)

    # Calculate line direction
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return result

    # Direction angles
    import math
    angle_to_end = math.degrees(math.atan2(dy, dx))
    angle_to_start = (angle_to_end + 180) % 360

    # Detect arrows at both ends
    start_arrow = detect_arrow_type(image, (x1, y1), angle_to_start)
    end_arrow = detect_arrow_type(image, (x2, y2), angle_to_end)

    result["arrow_start"] = {
        "type": start_arrow.arrow_type.value,
        "size": start_arrow.size,
        "confidence": start_arrow.confidence,
    }
    result["arrow_end"] = {
        "type": end_arrow.arrow_type.value,
        "size": end_arrow.size,
        "confidence": end_arrow.confidence,
    }

    return result


def generate_arrow_points(
    tip: tuple[float, float],
    direction: float,
    size: float,
    arrow_type: ArrowType = ArrowType.TRIANGLE,
) -> list[tuple[float, float]]:
    """Generate polygon points for rendering an arrow head.

    Args:
        tip: (x, y) position of arrow tip
        direction: Direction in degrees (0 = right, 90 = down)
        size: Size of arrow head
        arrow_type: Type of arrow to generate

    Returns:
        List of (x, y) points for polygon
    """
    import math

    rad = math.radians(direction)
    cos_d, sin_d = math.cos(rad), math.sin(rad)

    # Base vectors
    dx, dy = -cos_d * size, -sin_d * size  # Back from tip
    px, py = -sin_d * size * 0.5, cos_d * size * 0.5  # Perpendicular

    tx, ty = tip

    if arrow_type == ArrowType.TRIANGLE:
        # Filled triangle
        return [
            (tx, ty),
            (tx + dx + px, ty + dy + py),
            (tx + dx - px, ty + dy - py),
        ]

    elif arrow_type == ArrowType.TRIANGLE_OPEN:
        # Open triangle (just the outline, but same points)
        return [
            (tx, ty),
            (tx + dx + px, ty + dy + py),
            (tx + dx - px, ty + dy - py),
        ]

    elif arrow_type == ArrowType.LINE:
        # Line arrows: two lines from tip
        return [
            (tx + dx * 0.7 + px, ty + dy * 0.7 + py),
            (tx, ty),
            (tx + dx * 0.7 - px, ty + dy * 0.7 - py),
        ]

    elif arrow_type == ArrowType.DIAMOND:
        # Diamond shape
        return [
            (tx, ty),
            (tx + dx * 0.5 + px, ty + dy * 0.5 + py),
            (tx + dx, ty + dy),
            (tx + dx * 0.5 - px, ty + dy * 0.5 - py),
        ]

    else:
        # Default to triangle
        return [
            (tx, ty),
            (tx + dx + px, ty + dy + py),
            (tx + dx - px, ty + dy - py),
        ]


def render_arrow_svg(
    arrow: ArrowHead,
    fill: str = "#000000",
    stroke: str | None = None,
    stroke_width: float = 1.0,
) -> dict[str, Any]:
    """Generate SVG rendering parameters for an arrow head.

    Args:
        arrow: ArrowHead to render
        fill: Fill color
        stroke: Stroke color (None for no stroke)
        stroke_width: Stroke width

    Returns:
        Dict with SVG rendering parameters
    """
    points = generate_arrow_points(
        arrow.position,
        arrow.direction,
        arrow.size,
        arrow.arrow_type,
    )

    result = {
        "type": "polygon" if arrow.arrow_type != ArrowType.DOT else "circle",
        "points": points,
        "fill": fill if arrow.arrow_type != ArrowType.TRIANGLE_OPEN else "none",
    }

    if arrow.arrow_type == ArrowType.DOT:
        result["type"] = "circle"
        result["center"] = arrow.position
        result["radius"] = arrow.size / 2

    if stroke:
        result["stroke"] = stroke
        result["stroke_width"] = stroke_width

    return result
