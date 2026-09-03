"""MRST ``plotTracerBlend.m`` counterpart."""

from __future__ import annotations

import numpy as np


def plot_tracer_blend(G, partition, maxconc, *, ax=None, return_data: bool = False, **kwargs):
    """Plot a simple tracer-blend scalar field or return the blended values."""
    del kwargs
    partition = np.asarray(partition)
    maxconc = np.asarray(maxconc, dtype=float)
    if maxconc.ndim == 2:
        data = np.max(maxconc, axis=1)
    else:
        data = maxconc.ravel()
    if return_data:
        return data
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return data
    if ax is None:
        _, ax = plt.subplots()
    centroids = np.asarray(G["cells"].get("centroids", np.column_stack([np.arange(data.size), np.zeros(data.size)])), dtype=float)
    sc = ax.scatter(centroids[:, 0], centroids[:, 1], c=data, s=20)
    ax.figure.colorbar(sc, ax=ax)
    return ax


plotTracerBlend = plot_tracer_blend

