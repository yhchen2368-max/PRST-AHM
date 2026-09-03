"""MRST ``plotWellAllocationPanel.m`` counterpart."""

from __future__ import annotations

import numpy as np

from .helpers import get_field


def plot_well_allocation_panel(D, WP, *, ax=None, return_data: bool = False, **kwargs):
    """Plot/return injector allocation matrix from ``WP``."""
    del D, kwargs
    matrix = np.vstack([np.sum(np.asarray(item.alloc, dtype=float), axis=0) for item in get_field(WP, "inj", [])]) if get_field(WP, "inj", []) else np.zeros((0, 0))
    if return_data:
        return matrix
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return matrix
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(matrix, aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Producer")
    ax.set_ylabel("Injector")
    return ax


plotWellAllocationPanel = plot_well_allocation_panel

