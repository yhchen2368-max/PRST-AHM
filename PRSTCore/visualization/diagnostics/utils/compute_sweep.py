"""MRST ``computeSweep.m`` counterpart."""

from __future__ import annotations

import numpy as np


def compute_sweep(F: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sweep efficiency and dimensionless time from F-Phi data."""
    F = np.asarray(F, dtype=float).reshape(-1)
    Phi = np.asarray(Phi, dtype=float).reshape(-1)
    if F.size != Phi.size:
        raise ValueError("F and Phi must have the same length")
    if F.size == 0:
        return np.zeros(0), np.zeros(0)

    keep = np.ones(F.size, dtype=bool)
    keep[1:] = F[:-1] <= F[1:] - np.sqrt(np.finfo(float).eps)
    F = F[keep]
    Phi = Phi[keep]

    td = np.zeros_like(F)
    if F.size > 1:
        td[1:] = (Phi[1:] - Phi[:-1]) / (F[1:] - F[:-1])
    Ev = Phi + (1.0 - F) * td
    return Ev, td


computeSweep = compute_sweep
