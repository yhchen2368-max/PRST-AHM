"""MRST ``computeLorenz.m`` counterpart."""

from __future__ import annotations

import numpy as np


def compute_lorenz(F: np.ndarray, Phi: np.ndarray) -> float:
    F = np.asarray(F, dtype=float).reshape(-1)
    Phi = np.asarray(Phi, dtype=float).reshape(-1)
    if F.size != Phi.size:
        raise ValueError("F and Phi must have the same length")
    if F.size < 2:
        return 0.0
    volumes = np.diff(Phi)
    value = 2.0 * (np.sum((F[:-1] + F[1:]) / 2.0 * volumes) - 0.5)
    return float(value)


computeLorenz = compute_lorenz

