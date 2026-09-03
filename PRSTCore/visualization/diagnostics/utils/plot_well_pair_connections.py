"""MRST ``plotWellPairConnections.m`` counterpart."""

from __future__ import annotations

from typing import Any

import numpy as np

from .helpers import get_field, well_cells


def plot_well_pair_connections(G: dict[str, Any], WP: Any, D: Any, W: list[Any], pv, minAlloc: float = 0.01):
    """Plot or return well-pair connection curves.

    If matplotlib is available, a 3D plot is produced and the axes object is
    returned.  The function also attaches the computed line descriptors to
    ``ax._prstcore_connection_lines`` for tests/downstream use.
    """
    lines = _well_pair_connection_lines(G, WP, D, W, pv, minAlloc=minAlloc)
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for line in lines:
        pts = line["points"]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=max(line["width"], 0.5))
        ax.text(pts[1, 0], pts[1, 1], pts[1, 2], f"{100 * line['fraction']:.1f}%")
    ax._prstcore_connection_lines = lines
    return ax


def _well_pair_connection_lines(G, WP, D, W, pv, *, minAlloc=0.01):
    centroids = np.asarray(G["cells"].get("centroids"), dtype=float)
    pv = np.asarray(pv, dtype=float).ravel()
    inj = np.asarray(get_field(D, "inj"), dtype=int).ravel()
    prod = np.asarray(get_field(D, "prod"), dtype=int).ravel()
    ipart = np.asarray(get_field(D, "ipart"), dtype=int).ravel()
    ppart = np.asarray(get_field(D, "ppart"), dtype=int).ravel()
    max_alloc = 0.0
    for item in get_field(WP, "inj", []):
        max_alloc = max(max_alloc, float(np.max(np.sum(np.asarray(item.alloc, dtype=float), axis=0), initial=0.0)))
    lines = []
    for ii, iw in enumerate(inj):
        icells = well_cells(W[int(iw)], centroids.shape[0])
        ipos = np.mean(centroids[icells], axis=0)
        ialloc = np.sum(np.asarray(get_field(WP, "inj")[ii].alloc, dtype=float), axis=0)
        total = float(np.sum(ialloc))
        if abs(total) == 0.0:
            continue
        for pp, pw in enumerate(prod):
            alloc = float(ialloc[pp])
            frac = alloc / total
            if frac <= minAlloc:
                continue
            pcells = well_cells(W[int(pw)], centroids.shape[0])
            ppos = np.mean(centroids[pcells], axis=0)
            region = np.flatnonzero((ipart == ii + 1) & (ppart == pp + 1))
            if region.size:
                weights = pv[region]
                center = np.sum(centroids[region] * weights[:, None], axis=0) / max(float(np.sum(weights)), np.finfo(float).eps)
            else:
                center = 0.5 * (ipos + ppos)
            width = 20.0 * alloc / max(max_alloc, np.finfo(float).eps)
            lines.append({"injector": int(iw), "producer": int(pw), "allocation": alloc, "fraction": frac, "width": width, "points": np.vstack([ipos, center, ppos])})
    return lines


plotWellPairConnections = plot_well_pair_connections

