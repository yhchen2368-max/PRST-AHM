"""MRST ``plotWellAllocationComparison.m`` counterpart."""

from __future__ import annotations

import numpy as np

from .plot_well_allocation_panel import plot_well_allocation_panel


def plot_well_allocation_comparison(D1, WP1, D2, WP2, *, ax=None, return_data: bool = False, **kwargs):
    """Compare two injector allocation matrices."""
    A = plot_well_allocation_panel(D1, WP1, return_data=True)
    B = plot_well_allocation_panel(D2, WP2, return_data=True)
    diff = A - B
    if return_data:
        return diff
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return diff
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(diff, aspect="auto", cmap=kwargs.get("cmap", "coolwarm"))
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Allocation difference")
    return ax


plotWellAllocationComparison = plot_well_allocation_comparison

