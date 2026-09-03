"""Python port of MRST's ``cartGrid.m`` (mrst-2026a/core/gridprocessing)."""

from __future__ import annotations

import numpy as np

from .tensor_grid import tensor_grid


def cart_grid(celldim, physdim=None) -> dict:
    """Construct a 1D/2D/3D Cartesian grid in physical space.

    Parameters
    ----------
    celldim:
        Number of cells in each coordinate direction (length 1, 2 or 3).
    physdim:
        Physical size of the domain along each direction. Defaults to
        ``celldim`` (unit cell size), matching MRST's ``cartGrid``.
    """
    celldim = np.asarray(celldim, dtype=float).ravel()
    if not np.all(celldim > 0):
        raise ValueError("celldim must be positive")

    physdim = celldim if physdim is None else np.asarray(physdim, dtype=float).ravel()

    dim = celldim.size
    x = np.linspace(0.0, physdim[0], int(celldim[0]) + 1)
    if dim == 1:
        G = tensor_grid(x)
    elif dim == 2:
        y = np.linspace(0.0, physdim[1], int(celldim[1]) + 1)
        G = tensor_grid(x, y)
    elif dim == 3:
        y = np.linspace(0.0, physdim[1], int(celldim[1]) + 1)
        z = np.linspace(0.0, physdim[2], int(celldim[2]) + 1)
        G = tensor_grid(x, y, z)
    else:
        raise ValueError(f"Cannot create grid with {dim} dimensions: only 1, 2 or 3 is valid.")

    G["type"] = list(G["type"]) + ["cartGrid"]
    return G
