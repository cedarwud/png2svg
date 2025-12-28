"""Bezier curve fitting using the Philip J. Schneider algorithm.

This module provides high-quality curve fitting that converts a sequence of points
into smooth cubic Bezier curves with minimal error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BezierSegment:
    """A single cubic Bezier curve segment."""
    p0: tuple[float, float]  # Start point
    p1: tuple[float, float]  # Control point 1
    p2: tuple[float, float]  # Control point 2
    p3: tuple[float, float]  # End point


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    length = np.linalg.norm(v)
    if length < 1e-10:
        return np.array([1.0, 0.0])
    return v / length


def _bezier_point(t: float, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Evaluate cubic Bezier curve at parameter t."""
    u = 1.0 - t
    return (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3


def _bezier_derivative(t: float, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Evaluate derivative of cubic Bezier curve at parameter t."""
    u = 1.0 - t
    return 3 * (u ** 2) * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * (t ** 2) * (p3 - p2)


def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    """Assign parameter values to points using chord-length method."""
    n = len(points)
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = u[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    if u[-1] > 1e-10:
        u /= u[-1]
    return u


def _compute_left_tangent(points: np.ndarray) -> np.ndarray:
    """Compute unit tangent at start of curve."""
    if len(points) < 2:
        return np.array([1.0, 0.0])
    return _normalize(points[1] - points[0])


def _compute_right_tangent(points: np.ndarray) -> np.ndarray:
    """Compute unit tangent at end of curve."""
    if len(points) < 2:
        return np.array([1.0, 0.0])
    return _normalize(points[-2] - points[-1])


def _compute_center_tangent(points: np.ndarray, index: int) -> np.ndarray:
    """Compute unit tangent at a center point."""
    if index <= 0 or index >= len(points) - 1:
        return np.array([1.0, 0.0])
    v1 = points[index - 1] - points[index]
    v2 = points[index] - points[index + 1]
    return _normalize(v1 + v2)


def _generate_bezier(
    points: np.ndarray,
    u: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> BezierSegment:
    """Use least-squares to find Bezier control points for region.

    Based on the algorithm in "An Algorithm for Automatically Fitting Digitized Curves"
    by Philip J. Schneider, from "Graphics Gems".
    """
    n = len(points)
    p0 = points[0]
    p3 = points[-1]

    # Compute the A matrix
    a = np.zeros((n, 2, 2))
    for i in range(n):
        t = u[i]
        b1 = 3 * t * (1 - t) ** 2
        b2 = 3 * (t ** 2) * (1 - t)
        a[i, 0] = left_tangent * b1
        a[i, 1] = right_tangent * b2

    # Create C and X matrices
    c = np.zeros((2, 2))
    x = np.zeros(2)

    for i in range(n):
        c[0, 0] += np.dot(a[i, 0], a[i, 0])
        c[0, 1] += np.dot(a[i, 0], a[i, 1])
        c[1, 0] = c[0, 1]
        c[1, 1] += np.dot(a[i, 1], a[i, 1])

        t = u[i]
        tmp = points[i] - (
            p0 * ((1 - t) ** 3)
            + p0 * (3 * t * (1 - t) ** 2)
            + p3 * (3 * (t ** 2) * (1 - t))
            + p3 * (t ** 3)
        )
        x[0] += np.dot(a[i, 0], tmp)
        x[1] += np.dot(a[i, 1], tmp)

    # Solve for alpha values
    det_c0_c1 = c[0, 0] * c[1, 1] - c[1, 0] * c[0, 1]
    det_c0_x = c[0, 0] * x[1] - c[1, 0] * x[0]
    det_x_c1 = x[0] * c[1, 1] - x[1] * c[0, 1]

    if abs(det_c0_c1) < 1e-10:
        # Use Wu/Barsky heuristic when det is too small
        dist = np.linalg.norm(p3 - p0) / 3.0
        alpha_l = dist
        alpha_r = dist
    else:
        alpha_l = det_x_c1 / det_c0_c1
        alpha_r = det_c0_x / det_c0_c1

    # Check for negative alphas (shouldn't happen but be safe)
    seg_length = np.linalg.norm(p3 - p0)
    epsilon = 1e-6 * seg_length

    if alpha_l < epsilon or alpha_r < epsilon:
        # Fall back to heuristic
        dist = seg_length / 3.0
        alpha_l = dist
        alpha_r = dist

    p1 = p0 + left_tangent * alpha_l
    p2 = p3 + right_tangent * alpha_r

    return BezierSegment(
        p0=tuple(p0),
        p1=tuple(p1),
        p2=tuple(p2),
        p3=tuple(p3),
    )


def _compute_max_error(
    points: np.ndarray,
    segment: BezierSegment,
    u: np.ndarray,
) -> tuple[float, int]:
    """Compute max squared distance from points to fitted curve."""
    p0 = np.array(segment.p0)
    p1 = np.array(segment.p1)
    p2 = np.array(segment.p2)
    p3 = np.array(segment.p3)

    max_dist_sq = 0.0
    split_point = len(points) // 2

    for i, (point, t) in enumerate(zip(points, u)):
        curve_point = _bezier_point(t, p0, p1, p2, p3)
        dist_sq = np.sum((point - curve_point) ** 2)
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            split_point = i

    return max_dist_sq, split_point


def _reparameterize(
    points: np.ndarray,
    segment: BezierSegment,
    u: np.ndarray,
) -> np.ndarray:
    """Improve parameter values using Newton-Raphson iteration."""
    p0 = np.array(segment.p0)
    p1 = np.array(segment.p1)
    p2 = np.array(segment.p2)
    p3 = np.array(segment.p3)

    new_u = u.copy()
    for i, (point, t) in enumerate(zip(points, u)):
        if t <= 0 or t >= 1:
            continue

        # Compute Q(t)
        q = _bezier_point(t, p0, p1, p2, p3)

        # Compute Q'(t)
        q_prime = _bezier_derivative(t, p0, p1, p2, p3)

        # Compute Q''(t)
        q_prime_prime = 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)

        # Newton-Raphson
        numerator = np.dot(q - point, q_prime)
        denominator = np.dot(q_prime, q_prime) + np.dot(q - point, q_prime_prime)

        if abs(denominator) > 1e-10:
            new_t = t - numerator / denominator
            if 0 < new_t < 1:
                new_u[i] = new_t

    return new_u


def _fit_cubic_impl(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    max_error: float,
    max_iterations: int = 4,
) -> list[BezierSegment]:
    """Recursive implementation of curve fitting."""
    if len(points) == 2:
        dist = np.linalg.norm(points[1] - points[0]) / 3.0
        return [BezierSegment(
            p0=tuple(points[0]),
            p1=tuple(points[0] + left_tangent * dist),
            p2=tuple(points[1] + right_tangent * dist),
            p3=tuple(points[1]),
        )]

    # Chord-length parameterization
    u = _chord_length_parameterize(points)

    # Generate initial bezier curve
    segment = _generate_bezier(points, u, left_tangent, right_tangent)

    # Compute max error
    error_sq, split_point = _compute_max_error(points, segment, u)
    max_error_sq = max_error ** 2

    if error_sq < max_error_sq:
        return [segment]

    # Try to improve with reparameterization
    if error_sq < max_error_sq * 4:
        for _ in range(max_iterations):
            u = _reparameterize(points, segment, u)
            segment = _generate_bezier(points, u, left_tangent, right_tangent)
            error_sq, split_point = _compute_max_error(points, segment, u)
            if error_sq < max_error_sq:
                return [segment]

    # Split and recursively fit
    split_point = max(1, min(split_point, len(points) - 2))
    center_tangent = _compute_center_tangent(points, split_point)

    left_segments = _fit_cubic_impl(
        points[: split_point + 1],
        left_tangent,
        center_tangent,
        max_error,
        max_iterations,
    )
    right_segments = _fit_cubic_impl(
        points[split_point:],
        -center_tangent,
        right_tangent,
        max_error,
        max_iterations,
    )

    return left_segments + right_segments


def fit_bezier_curve(
    points: list[tuple[float, float]],
    max_error: float = 2.0,
) -> list[BezierSegment]:
    """Fit a sequence of points to cubic Bezier curves.

    Uses the Philip J. Schneider algorithm from Graphics Gems.

    Args:
        points: List of (x, y) point coordinates
        max_error: Maximum allowed error in pixels

    Returns:
        List of BezierSegment objects
    """
    if len(points) < 2:
        return []

    if len(points) == 2:
        p0, p1 = points[0], points[1]
        dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5 / 3.0
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-10:
            return []
        tx, ty = dx / length, dy / length
        return [BezierSegment(
            p0=p0,
            p1=(p0[0] + tx * dist, p0[1] + ty * dist),
            p2=(p1[0] - tx * dist, p1[1] - ty * dist),
            p3=p1,
        )]

    pts = np.array(points, dtype=np.float64)
    left_tangent = _compute_left_tangent(pts)
    right_tangent = _compute_right_tangent(pts)

    return _fit_cubic_impl(pts, left_tangent, right_tangent, max_error)


def bezier_segments_to_svg_path(segments: list[BezierSegment]) -> str:
    """Convert Bezier segments to SVG path string.

    Args:
        segments: List of BezierSegment objects

    Returns:
        SVG path 'd' attribute string
    """
    if not segments:
        return ""

    parts = []
    for i, seg in enumerate(segments):
        if i == 0:
            parts.append(f"M {seg.p0[0]:.2f} {seg.p0[1]:.2f}")
        parts.append(
            f"C {seg.p1[0]:.2f} {seg.p1[1]:.2f} "
            f"{seg.p2[0]:.2f} {seg.p2[1]:.2f} "
            f"{seg.p3[0]:.2f} {seg.p3[1]:.2f}"
        )

    return " ".join(parts)


def fit_and_convert_to_path(
    points: list[tuple[float, float]],
    max_error: float = 2.0,
) -> str:
    """Convenience function: fit points and return SVG path string.

    Args:
        points: List of (x, y) point coordinates
        max_error: Maximum allowed error in pixels

    Returns:
        SVG path 'd' attribute string
    """
    segments = fit_bezier_curve(points, max_error)
    return bezier_segments_to_svg_path(segments)


def simplify_path_with_bezier(
    points: list[tuple[float, float]],
    max_error: float = 3.0,
    max_segments: int = 10,
) -> str:
    """Simplify a point sequence to Bezier curves with segment limit.

    Args:
        points: List of (x, y) point coordinates
        max_error: Starting max error for fitting
        max_segments: Maximum number of Bezier segments

    Returns:
        SVG path 'd' attribute string
    """
    if len(points) < 2:
        return ""

    # Start with default error
    error = max_error
    segments = fit_bezier_curve(points, error)

    # If too many segments, increase error tolerance
    iterations = 0
    while len(segments) > max_segments and iterations < 5:
        error *= 1.5
        segments = fit_bezier_curve(points, error)
        iterations += 1

    # If still too many, just sample points evenly
    if len(segments) > max_segments:
        step = max(1, len(points) // (max_segments + 1))
        sampled = points[::step]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        segments = fit_bezier_curve(sampled, error)

    return bezier_segments_to_svg_path(segments)
