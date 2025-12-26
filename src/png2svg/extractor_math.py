from __future__ import annotations

import numpy as np


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window, window), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[window:-window]


def _snap_flat_series(values: np.ndarray, tolerance: float) -> np.ndarray:
    if values.size == 0:
        return values
    snapped = values.copy()
    last = snapped[0]
    for idx in range(1, snapped.size):
        if abs(snapped[idx] - last) <= tolerance:
            snapped[idx] = last
        else:
            last = snapped[idx]
    return snapped
