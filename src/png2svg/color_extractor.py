"""Color extraction module for accurate SVG color reproduction.

Extracts dominant colors from image regions to ensure SVG elements
match the original PNG colors.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ColorPalette:
    """Extracted color palette from an image region."""
    background: str           # Dominant background color (hex)
    primary_colors: list[str] # Main content colors
    accent_colors: list[str]  # Secondary/accent colors
    text_color: str           # Detected text color
    confidence: float         # Extraction confidence


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Compute Euclidean distance between two colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _is_near_white(r: int, g: int, b: int, threshold: int = 245) -> bool:
    """Check if color is near white."""
    return r >= threshold and g >= threshold and b >= threshold


def _is_near_black(r: int, g: int, b: int, threshold: int = 30) -> bool:
    """Check if color is near black."""
    return r <= threshold and g <= threshold and b <= threshold


def _cluster_colors(
    colors: list[tuple[int, int, int]],
    threshold: float = 30.0,
) -> list[tuple[tuple[int, int, int], int]]:
    """Cluster similar colors together.

    Args:
        colors: List of RGB tuples
        threshold: Maximum distance to consider colors similar

    Returns:
        List of (representative_color, count) tuples
    """
    if not colors:
        return []

    # Count colors
    color_counts = Counter(colors)

    # Cluster similar colors
    clusters: list[list[tuple[tuple[int, int, int], int]]] = []

    for color, count in color_counts.most_common():
        # Find if this color belongs to an existing cluster
        found_cluster = False
        for cluster in clusters:
            rep_color = cluster[0][0]
            if _color_distance(color, rep_color) < threshold:
                cluster.append((color, count))
                found_cluster = True
                break

        if not found_cluster:
            clusters.append([(color, count)])

    # Compute representative color for each cluster (weighted average)
    result = []
    for cluster in clusters:
        total_count = sum(c[1] for c in cluster)
        weighted_r = sum(c[0][0] * c[1] for c in cluster) / total_count
        weighted_g = sum(c[0][1] * c[1] for c in cluster) / total_count
        weighted_b = sum(c[0][2] * c[1] for c in cluster) / total_count
        rep_color = (int(weighted_r), int(weighted_g), int(weighted_b))
        result.append((rep_color, total_count))

    return sorted(result, key=lambda x: -x[1])


def extract_region_colors(
    rgba: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    sample_step: int = 5,
) -> list[tuple[str, int]]:
    """Extract colors from a rectangular region.

    Args:
        rgba: Source image as RGBA numpy array
        x, y: Top-left corner of region
        width, height: Region dimensions
        sample_step: Sampling interval to reduce computation

    Returns:
        List of (hex_color, count) tuples, sorted by count
    """
    h, w = rgba.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + width)
    y2 = min(h, y + height)

    if x2 <= x1 or y2 <= y1:
        return []

    region = rgba[y1:y2:sample_step, x1:x2:sample_step]

    # Extract RGB (ignore alpha)
    colors = []
    for row in region:
        for pixel in row:
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
            colors.append((r, g, b))

    # Cluster and return
    clustered = _cluster_colors(colors, threshold=25.0)
    return [(_rgb_to_hex(*c), n) for c, n in clustered]


def extract_background_color(
    rgba: np.ndarray,
    margin: int = 20,
) -> str:
    """Extract the dominant background color.

    Samples from corners and edges of the image.

    Args:
        rgba: Source image as RGBA numpy array
        margin: Margin from edge to sample

    Returns:
        Hex color string
    """
    h, w = rgba.shape[:2]

    # Sample corners
    corners = [
        rgba[margin:margin*3, margin:margin*3],          # Top-left
        rgba[margin:margin*3, w-margin*3:w-margin],      # Top-right
        rgba[h-margin*3:h-margin, margin:margin*3],      # Bottom-left
        rgba[h-margin*3:h-margin, w-margin*3:w-margin],  # Bottom-right
    ]

    colors = []
    for corner in corners:
        for row in corner[::3]:
            for pixel in row[::3]:
                r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                colors.append((r, g, b))

    if not colors:
        return "#ffffff"

    # Get most common color
    clustered = _cluster_colors(colors, threshold=15.0)
    if clustered:
        return _rgb_to_hex(*clustered[0][0])
    return "#ffffff"


def extract_panel_fill_color(
    rgba: np.ndarray,
    panel: dict[str, Any],
    sample_margin: float = 0.1,
) -> str:
    """Extract the fill color for a panel region.

    Samples from the interior of the panel, avoiding edges and content.

    Args:
        rgba: Source image
        panel: Panel dict with x, y, width, height
        sample_margin: Fraction of panel size to use as margin

    Returns:
        Hex color string
    """
    x = int(panel.get("x", 0))
    y = int(panel.get("y", 0))
    w = int(panel.get("width", 100))
    h = int(panel.get("height", 100))

    # Sample from inner region
    margin_x = int(w * sample_margin)
    margin_y = int(h * sample_margin)

    colors = extract_region_colors(
        rgba,
        x + margin_x,
        y + margin_y,
        w - margin_x * 2,
        h - margin_y * 2,
        sample_step=10,
    )

    if not colors:
        return "#ffffff"

    # Return most common non-black color
    for hex_color, count in colors:
        rgb = _hex_to_rgb(hex_color)
        if not _is_near_black(*rgb):
            return hex_color

    return "#ffffff"


def extract_curve_colors(
    rgba: np.ndarray,
    panel: dict[str, Any],
    num_colors: int = 5,
) -> list[str]:
    """Extract the main curve colors from a panel.

    Args:
        rgba: Source image
        panel: Panel dict
        num_colors: Maximum number of colors to return

    Returns:
        List of hex color strings
    """
    x = int(panel.get("x", 0))
    y = int(panel.get("y", 0))
    w = int(panel.get("width", 100))
    h = int(panel.get("height", 100))

    colors = extract_region_colors(rgba, x, y, w, h, sample_step=5)

    # Filter out near-white and near-black
    curve_colors = []
    for hex_color, count in colors:
        rgb = _hex_to_rgb(hex_color)
        if _is_near_white(*rgb) or _is_near_black(*rgb):
            continue
        # Also filter out very light colors (likely backgrounds)
        if sum(rgb) > 700:  # Very light
            continue
        curve_colors.append(hex_color)
        if len(curve_colors) >= num_colors:
            break

    return curve_colors


def extract_threshold_band_colors(
    rgba: np.ndarray,
    panel: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect colored threshold bands in a panel.

    Looks for horizontal bands of color that span the panel width.

    Args:
        rgba: Source image
        panel: Panel dict

    Returns:
        List of band dicts with y, height, and fill color
    """
    x = int(panel.get("x", 0))
    y = int(panel.get("y", 0))
    w = int(panel.get("width", 100))
    h = int(panel.get("height", 100))

    img_h, img_w = rgba.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    # Analyze horizontal strips
    bands = []
    strip_height = max(5, h // 50)
    prev_color = None

    for strip_y in range(y, y + h - strip_height, strip_height):
        # Sample this strip
        strip = rgba[strip_y:strip_y + strip_height, x:x + w:5]
        colors = []
        for row in strip:
            for pixel in row:
                r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                if not _is_near_white(r, g, b, 250) and not _is_near_black(r, g, b):
                    colors.append((r, g, b))

        if not colors:
            prev_color = None
            continue

        # Get dominant color
        clustered = _cluster_colors(colors, threshold=20.0)
        if not clustered:
            prev_color = None
            continue

        dominant = clustered[0][0]
        hex_color = _rgb_to_hex(*dominant)

        # Check if this is a continuation of previous band
        if prev_color and _color_distance(dominant, _hex_to_rgb(prev_color)) < 30:
            # Extend previous band
            if bands:
                bands[-1]["height"] = strip_y + strip_height - bands[-1]["y"]
        else:
            # Start new band
            bands.append({
                "y": strip_y,
                "height": strip_height,
                "fill": hex_color,
            })

        prev_color = hex_color

    # Filter out very thin bands and adjust to panel-relative coordinates
    result = []
    for band in bands:
        if band["height"] < h * 0.05:  # Skip thin bands
            continue
        band["y"] = band["y"] - y  # Make relative to panel
        result.append(band)

    return result


def extract_palette(
    rgba: np.ndarray,
    panels: list[dict[str, Any]] | None = None,
) -> ColorPalette:
    """Extract a complete color palette from an image.

    Args:
        rgba: Source image
        panels: Optional list of panel dicts

    Returns:
        ColorPalette with extracted colors
    """
    # Get background
    background = extract_background_color(rgba)

    # Get all non-background colors
    h, w = rgba.shape[:2]
    all_colors = extract_region_colors(rgba, 0, 0, w, h, sample_step=10)

    # Separate into categories
    primary = []
    accent = []
    text_color = "#000000"

    for hex_color, count in all_colors:
        rgb = _hex_to_rgb(hex_color)

        if _is_near_black(*rgb):
            text_color = hex_color
            continue

        if _is_near_white(*rgb):
            continue

        # Saturated colors are primary
        r, g, b = rgb
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        saturation = (max_c - min_c) / max_c if max_c > 0 else 0

        if saturation > 0.3:
            primary.append(hex_color)
        else:
            accent.append(hex_color)

    # Limit lists
    primary = primary[:5]
    accent = accent[:5]

    return ColorPalette(
        background=background,
        primary_colors=primary,
        accent_colors=accent,
        text_color=text_color,
        confidence=0.8 if all_colors else 0.0,
    )


__all__ = [
    'ColorPalette',
    'extract_background_color',
    'extract_panel_fill_color',
    'extract_curve_colors',
    'extract_threshold_band_colors',
    'extract_palette',
    'extract_region_colors',
]
