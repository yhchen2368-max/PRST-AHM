"""Python port of MRST's ``triangleGrid.m`` (mrst-2026a/core/gridprocessing)."""

from __future__ import annotations

import numpy as np

from .tessellation_grid import tessellation_grid


def triangle_grid(p, t=None):
    """Construct a valid grid definition from points and a triangle list.

    Parameters
    ----------
    p : (n, 2) array
        Node coordinates.
    t : (m, 3) int array, optional
        Triangle list (node indices into ``p``, 0-based). Defaults to the
        Delaunay triangulation of ``p``.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("triangle_grid is only supported in two space dimensions")

    if t is None:
        from scipy.spatial import Delaunay
        t = Delaunay(p).simplices
    else:
        t = np.asarray(t, dtype=np.int64)
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError("triangle_grid: T must be an n-by-3 triangle list")
        if t.size and (t.max() >= p.shape[0] or t.min() < 0):
            raise ValueError("triangle_grid: T references invalid points")

    G = tessellation_grid(p, t)
    G["type"] = ["triangleGrid"]
    return G
