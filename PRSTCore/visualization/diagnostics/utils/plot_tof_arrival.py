"""MRST ``plotTOFArrival.m`` counterpart."""

from __future__ import annotations

import numpy as np

from .helpers import get_field


def plot_tof_arrival(state, W, pv, fluid, prod, D, *, ax=None, return_data: bool = False, **kwargs):
    """Plot/return cumulative pore-volume arrival by reverse TOF."""
    del state, W, fluid, prod, kwargs
    tof = np.asarray(get_field(D, "tof"), dtype=float)
    pv = np.asarray(pv, dtype=float).ravel()
    order = np.argsort(tof[:, 1])
    x = tof[order, 1]
    y = np.cumsum(pv[order]) / max(float(np.sum(pv)), np.finfo(float).eps)
    if return_data:
        return x, y
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return x, y
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Reverse TOF")
    ax.set_ylabel("Cumulative PV")
    return ax


plotTOFArrival = plot_tof_arrival

