"""MRST private ``computeStatistic.m`` counterpart."""

import numpy as np


def compute_statistic(vals, stat, prop=None):
    vals = np.asarray(vals, dtype=float)
    harmonic = isinstance(prop, str) and prop[:3] == "TOF"
    if stat == "mean":
        return 1.0 / np.mean(1.0 / vals, axis=1) if harmonic else np.mean(vals, axis=1)
    if stat == "std":
        return 1.0 / np.std(1.0 / vals, axis=1, ddof=0) if harmonic else np.std(vals, axis=1, ddof=0)
    if stat == "max diff":
        return np.max(vals, axis=1) - np.min(vals, axis=1)
    raise ValueError(f"Unknown statistic: {stat}")


computeStatistic = compute_statistic

