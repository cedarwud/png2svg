from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from png2svg.extractor_math import _smooth_series, _snap_flat_series


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    hue = np.zeros_like(cmax)
    mask = delta > 1e-6
    mask_r = (cmax == r) & mask
    mask_g = (cmax == g) & mask
    mask_b = (cmax == b) & mask
    hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360.0
    hue[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) % 360.0
    hue[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) % 360.0
    sat = np.zeros_like(cmax)
    nonzero = cmax > 1e-6
    sat[nonzero] = delta[nonzero] / cmax[nonzero]
    val = cmax
    return hue, sat, val


def _hue_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _hue_in_range(hue: np.ndarray, low: float, high: float) -> np.ndarray:
    low = low % 360.0
    high = high % 360.0
    if low <= high:
        return (hue >= low) & (hue <= high)
    return (hue >= low) | (hue <= high)


def _pick_hue_center(hue: np.ndarray, mask: np.ndarray, target: float, max_distance: float) -> float | None:
    if mask.sum() == 0:
        return None
    hue_vals = hue[mask]
    bins = ((hue_vals // 10) % 36).astype(np.int32)
    counts = np.bincount(bins, minlength=36)
    best_bin = None
    best_count = 0
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        center = idx * 10.0 + 5.0
        if _hue_distance(center, target) <= max_distance and count > best_count:
            best_bin = idx
            best_count = int(count)
    if best_bin is None:
        return None
    return best_bin * 10.0 + 5.0


def _morph_dilation(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    # Pad with False (0) for dilation
    padded = np.pad(mask, pad, mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return np.max(windows, axis=(2, 3))


def _morph_erosion(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    # Pad with True (1) for erosion
    padded = np.pad(mask, pad, mode='constant', constant_values=1)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return np.min(windows, axis=(2, 3))


def _morph_close(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Perform morphological closing (dilation -> erosion)."""
    dilated = _morph_dilation(mask, kernel_size)
    return _morph_erosion(dilated, kernel_size)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    if min_area <= 1:
        return mask

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask, dtype=bool)
    
    # We only care about True pixels
    # Optimization: Iterate only over True pixels using np.where
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return output

    # Set of unvisited True pixels for O(1) lookup
    unvisited = set(zip(ys, xs))
    
    while unvisited:
        # Pick a random start node
        y, x = unvisited.pop()
        
        # Flood fill
        stack = [(y, x)]
        component = [(y, x)]
        visited[y, x] = True
        
        idx = 0
        while idx < len(component):
            cy, cx = component[idx]
            idx += 1
            
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        coord = (ny, nx)
                        component.append(coord)
                        if coord in unvisited:
                            unvisited.remove(coord)
        
        if len(component) >= min_area:
            for cy, cx in component:
                output[cy, cx] = True
                
    return output


def _curve_color_mask(
    rgba: np.ndarray,
    target_hue: float,
    adaptive: dict[str, Any],
) -> np.ndarray:
    hue, sat, val = _rgb_to_hsv(rgba[:, :, :3])
    alpha = rgba[:, :, 3]
    sat_min = float(adaptive.get("saturation_min", 0.25))
    val_min = float(adaptive.get("value_min", 0.2))
    hue_tol = float(adaptive.get("hue_tolerance_deg", 22.0))

    def _build_mask(sat_threshold: float, val_threshold: float, tol: float) -> np.ndarray:
        base = (alpha > 10) & (sat >= sat_threshold) & (val >= val_threshold)
        center = _pick_hue_center(hue, base, target_hue, max_distance=tol * 2.0)
        if center is None:
            center = target_hue
        return base & _hue_in_range(hue, center - tol, center + tol)

    mask = _build_mask(sat_min, val_min, hue_tol)
    min_pixels = max(int(rgba.shape[0] * rgba.shape[1] * 0.0005), 8)
    
    if mask.sum() < min_pixels:
        relaxed_sat = max(sat_min * 0.6, 0.05)
        relaxed_val = max(val_min * 0.6, 0.05)
        relaxed_tol = min(hue_tol * 1.4, 60.0)
        mask = _build_mask(relaxed_sat, relaxed_val, relaxed_tol)
    
    # Improve mask quality (P0.1 improvements)
    # 1. Morphological closing to fill gaps
    mask = _morph_close(mask, kernel_size=3)
    
    # 2. Remove small noise
    mask = _remove_small_components(mask, min_area=30)
    
    return mask


def _extract_dominant_color(
    rgba: np.ndarray,
    mask: np.ndarray,
    fallback_color: str = "#2b6cb0",
) -> str:
    """Extract the dominant color from pixels matching a mask.

    Args:
        rgba: Source image as RGBA numpy array
        mask: Boolean mask indicating which pixels to sample
        fallback_color: Color to return if extraction fails

    Returns:
        Hex color string of dominant color
    """
    if mask.sum() < 10:
        return fallback_color

    # Get pixels matching the mask
    rgb = rgba[:, :, :3]
    masked_pixels = rgb[mask]

    if len(masked_pixels) == 0:
        return fallback_color

    # Calculate median color (more robust than mean)
    median_r = int(np.median(masked_pixels[:, 0]))
    median_g = int(np.median(masked_pixels[:, 1]))
    median_b = int(np.median(masked_pixels[:, 2]))

    return f"#{median_r:02x}{median_g:02x}{median_b:02x}"


def extract_curve_color(
    rgba: np.ndarray,
    target_hue: float,
    adaptive: dict[str, Any],
    fallback_color: str = "#2b6cb0",
) -> tuple[np.ndarray, str]:
    """Extract curve mask and its dominant color.

    Args:
        rgba: Source image
        target_hue: Target hue angle (0-360)
        adaptive: Configuration dict
        fallback_color: Color to use if extraction fails

    Returns:
        Tuple of (mask, hex_color)
    """
    mask = _curve_color_mask(rgba, target_hue, adaptive)
    color = _extract_dominant_color(rgba, mask, fallback_color)
    return mask, color


def _point_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        dx = point[0] - start[0]
        dy = point[1] - start[1]
        return (dx * dx + dy * dy) ** 0.5
    x0, y0 = point
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return num / den


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_dist = 0.0
    index = 0
    for idx, point in enumerate(points[1:-1], start=1):
        dist = _point_distance(point, start, end)
        if dist > max_dist:
            max_dist = dist
            index = idx
    if max_dist > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def _simplify_curve_points(
    points: list[tuple[float, float]],
    max_points: int,
    min_points: int,
    epsilon: float,
) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    simplified = _rdp(points, epsilon)
    step = 0
    while len(simplified) > max_points and step < 6:
        epsilon *= 1.6
        simplified = _rdp(points, epsilon)
        step += 1
    if len(simplified) > max_points:
        keep = [points[0]]
        if max_points > 2:
            stride = max(1, int((len(points) - 2) / (max_points - 2)))
            keep.extend(points[1:-1:stride])
        keep.append(points[-1])
        simplified = keep[:max_points]
    if len(simplified) < min_points:
        if len(points) >= min_points:
            simplified = points[:min_points]
        else:
            simplified = points
    return simplified


def _curve_centerline_points(
    mask: np.ndarray,
    adaptive: dict[str, Any],
) -> list[tuple[float, float]]:
    height, width = mask.shape
    xs: list[int] = []
    ys: list[float] = []
    # Relaxed minimum samples to detect thin lines (e.g. 2-3px) even in large panels
    min_samples = max(int(height * 0.002), 2)
    for x in range(width):
        ys_col = np.where(mask[:, x])[0]
        if ys_col.size >= min_samples:
            xs.append(x)
            ys.append(float(np.median(ys_col)))
    if len(xs) < 4:
        return []
    xs_arr = np.array(xs, dtype=np.int32)
    ys_arr = np.array(ys, dtype=np.float32)
    xs_full = np.arange(xs_arr[0], xs_arr[-1] + 1)
    ys_full = np.interp(xs_full, xs_arr.astype(np.float32), ys_arr)
    smooth_window = int(adaptive.get("smooth_window", 7))
    smooth_window = max(smooth_window, 3)
    if smooth_window % 2 == 0:
        smooth_window += 1
    ys_smooth = _smooth_series(ys_full, smooth_window)
    ys_smooth = _snap_flat_series(ys_smooth, tolerance=1.5)
    points = list(zip(xs_full.astype(np.float32), ys_smooth.astype(np.float32)))

    sample_spacing = max(int(adaptive.get("sample_spacing_px", 120)), 10)
    min_segments = int(adaptive.get("min_segments", 4))
    max_segments = int(adaptive.get("max_segments", 8))
    
    # Increase limits for higher quality Bezier fitting
    # We want more points so the fitter can do a better job
    max_segments_hq = max_segments * 3
    
    # Less aggressive epsilon for RDP
    # was: max(sample_spacing * 0.15, 4.0)
    # now: smaller epsilon to keep details
    epsilon = max(sample_spacing * 0.05, 2.0)
    
    min_points = max(min_segments + 1, 4)
    max_points = max(max_segments_hq + 1, min_points)
    
    return _simplify_curve_points(points, max_points=max_points, min_points=min_points, epsilon=epsilon)
