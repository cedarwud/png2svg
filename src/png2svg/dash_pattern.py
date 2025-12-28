"""Dash pattern detection and classification module.

Analyzes run patterns in binary masks to determine dash-gap patterns
for accurate SVG stroke-dasharray rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DashPattern:
    """Detected dash pattern information."""
    dash_length: float       # Length of dash segment
    gap_length: float        # Length of gap segment
    dasharray: list[float]   # SVG dasharray [dash, gap] or [dash, gap, dash, gap, ...]
    confidence: float        # Detection confidence (0-1)
    pattern_type: str        # 'uniform', 'alternating', 'dotted', 'unknown'


def _analyze_runs(runs: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Extract dash lengths and gap lengths from run data.

    Args:
        runs: List of (start_position, length) tuples for ink segments

    Returns:
        Tuple of (dash_lengths, gap_lengths)
    """
    if len(runs) < 2:
        return [], []

    dash_lengths = [run[1] for run in runs]
    gap_lengths = []

    for i in range(len(runs) - 1):
        gap = runs[i + 1][0] - (runs[i][0] + runs[i][1])
        if gap > 0:
            gap_lengths.append(gap)

    return dash_lengths, gap_lengths


def _compute_pattern_stats(
    dash_lengths: list[int],
    gap_lengths: list[int],
) -> dict[str, float]:
    """Compute statistics about dash and gap patterns.

    Args:
        dash_lengths: List of dash segment lengths
        gap_lengths: List of gap segment lengths

    Returns:
        Dict with mean, std, coefficient of variation for dashes and gaps
    """
    if not dash_lengths:
        return {}

    dash_mean = sum(dash_lengths) / len(dash_lengths)
    dash_std = (sum((d - dash_mean) ** 2 for d in dash_lengths) / len(dash_lengths)) ** 0.5
    dash_cv = dash_std / dash_mean if dash_mean > 0 else 0

    if gap_lengths:
        gap_mean = sum(gap_lengths) / len(gap_lengths)
        gap_std = (sum((g - gap_mean) ** 2 for g in gap_lengths) / len(gap_lengths)) ** 0.5
        gap_cv = gap_std / gap_mean if gap_mean > 0 else 0
    else:
        gap_mean = 0
        gap_std = 0
        gap_cv = 0

    return {
        'dash_mean': dash_mean,
        'dash_std': dash_std,
        'dash_cv': dash_cv,
        'gap_mean': gap_mean,
        'gap_std': gap_std,
        'gap_cv': gap_cv,
    }


def _classify_pattern_type(
    dash_lengths: list[int],
    gap_lengths: list[int],
    stats: dict[str, float],
) -> str:
    """Classify the type of dash pattern.

    Args:
        dash_lengths: List of dash segment lengths
        gap_lengths: List of gap segment lengths
        stats: Pattern statistics from _compute_pattern_stats

    Returns:
        Pattern type: 'uniform', 'alternating', 'dotted', 'unknown'
    """
    if not dash_lengths or not gap_lengths:
        return 'unknown'

    dash_mean = stats.get('dash_mean', 0)
    gap_mean = stats.get('gap_mean', 0)
    dash_cv = stats.get('dash_cv', 1)
    gap_cv = stats.get('gap_cv', 1)

    # Check for dotted pattern (very short dashes, longer gaps)
    if dash_mean < 4 and gap_mean > dash_mean * 1.5:
        return 'dotted'

    # Check for uniform pattern (low variation in both)
    if dash_cv < 0.3 and gap_cv < 0.3:
        return 'uniform'

    # Check for alternating pattern (two distinct lengths)
    if len(set(dash_lengths)) <= 3:
        return 'alternating'

    return 'unknown'


def _round_to_standard(value: float) -> float:
    """Round a dash/gap value to a standard SVG-friendly value.

    Args:
        value: Raw measured value

    Returns:
        Rounded value (multiples of 2 for small values, 5 for larger)
    """
    if value < 10:
        return round(value / 2) * 2
    return round(value / 5) * 5


def detect_dash_pattern(
    runs: list[tuple[int, int]],
    min_count: int = 3,
) -> DashPattern:
    """Detect the dash pattern from a sequence of run segments.

    Args:
        runs: List of (start_position, length) tuples for ink segments
        min_count: Minimum number of dashes to analyze

    Returns:
        DashPattern with detected parameters
    """
    if len(runs) < min_count:
        return DashPattern(
            dash_length=4.0,
            gap_length=4.0,
            dasharray=[4.0, 4.0],
            confidence=0.0,
            pattern_type='unknown',
        )

    dash_lengths, gap_lengths = _analyze_runs(runs)

    if not gap_lengths:
        return DashPattern(
            dash_length=4.0,
            gap_length=4.0,
            dasharray=[4.0, 4.0],
            confidence=0.0,
            pattern_type='unknown',
        )

    stats = _compute_pattern_stats(dash_lengths, gap_lengths)
    pattern_type = _classify_pattern_type(dash_lengths, gap_lengths, stats)

    dash_mean = stats.get('dash_mean', 4.0)
    gap_mean = stats.get('gap_mean', 4.0)
    dash_cv = stats.get('dash_cv', 1.0)
    gap_cv = stats.get('gap_cv', 1.0)

    # Round to standard values
    dash_length = max(2.0, _round_to_standard(dash_mean))
    gap_length = max(2.0, _round_to_standard(gap_mean))

    # Compute confidence based on pattern regularity
    regularity = 1.0 - min(1.0, (dash_cv + gap_cv) / 2)
    sample_confidence = min(1.0, len(runs) / 10)
    confidence = regularity * 0.6 + sample_confidence * 0.4

    # Create dasharray
    if pattern_type == 'dotted':
        dasharray = [2.0, gap_length]
    elif pattern_type == 'uniform':
        dasharray = [dash_length, gap_length]
    elif pattern_type == 'alternating':
        # For alternating, try to detect two distinct lengths
        sorted_dashes = sorted(dash_lengths)
        if len(sorted_dashes) >= 4:
            short = sum(sorted_dashes[:len(sorted_dashes)//2]) / (len(sorted_dashes)//2)
            long = sum(sorted_dashes[len(sorted_dashes)//2:]) / (len(sorted_dashes) - len(sorted_dashes)//2)
            dasharray = [
                _round_to_standard(short),
                gap_length,
                _round_to_standard(long),
                gap_length,
            ]
        else:
            dasharray = [dash_length, gap_length]
    else:
        dasharray = [dash_length, gap_length]

    return DashPattern(
        dash_length=dash_length,
        gap_length=gap_length,
        dasharray=dasharray,
        confidence=confidence,
        pattern_type=pattern_type,
    )


def enhance_line_with_dash_pattern(
    line: dict[str, Any],
    runs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Enhance a line dict with detected dash pattern.

    Args:
        line: Line dict with stroke properties
        runs: List of (start, length) tuples from the image

    Returns:
        Enhanced line dict with dasharray and dash_info
    """
    pattern = detect_dash_pattern(runs)

    result = dict(line)

    # Only update if we have reasonable confidence
    if pattern.confidence > 0.3:
        result['dasharray'] = pattern.dasharray
        result['dash_info'] = {
            'dash_length': pattern.dash_length,
            'gap_length': pattern.gap_length,
            'pattern_type': pattern.pattern_type,
            'confidence': round(pattern.confidence, 3),
        }

    return result


def analyze_mask_dash_pattern(
    mask: np.ndarray,
    x: int,
    axis: int = 0,
) -> DashPattern:
    """Analyze a column or row of a mask to detect dash pattern.

    Args:
        mask: Binary mask (2D numpy array)
        x: Column index (for axis=0) or row index (for axis=1)
        axis: 0 for vertical lines, 1 for horizontal lines

    Returns:
        DashPattern with detected parameters
    """
    if axis == 0:
        values = mask[:, x]
    else:
        values = mask[x, :]

    # Extract runs
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0

    for i, val in enumerate(values):
        if val > 0 and not in_run:
            in_run = True
            start = i
        elif val == 0 and in_run:
            in_run = False
            runs.append((start, i - start))

    if in_run:
        runs.append((start, len(values) - start))

    return detect_dash_pattern(runs)


__all__ = [
    'DashPattern',
    'detect_dash_pattern',
    'enhance_line_with_dash_pattern',
    'analyze_mask_dash_pattern',
]
