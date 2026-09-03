"""Partition a Cartesian grid.

1:1 Python translation of MRST multiscale/coarsegrid/partitionCartGrid.m
"""

import numpy as np


def partition_cart_grid(cart_dims, part_dims):
    """Partition a Cartesian grid into coarse blocks.

    Parameters
    ----------
    cart_dims : tuple
        (nx, ny, nz) fine-grid cell dimensions.
    part_dims : tuple
        (cnx, cny, cnz) coarse-grid block dimensions.

    Returns
    -------
    ndarray
        Partition vector of size [nx*ny*nz] (1-based).
    """
    cart_dims = np.atleast_1d(cart_dims).ravel()
    part_dims = np.atleast_1d(part_dims).ravel()

    assert len(cart_dims) == len(part_dims)
    assert len(cart_dims) in (2, 3)
    assert np.all(cart_dims > 0) and np.all(part_dims > 0)

    if len(cart_dims) == 2:
        cart_dims = np.append(cart_dims, 1)
        part_dims = np.append(part_dims, 1)

    nx, ny, nz = int(cart_dims[0]), int(cart_dims[1]), int(cart_dims[2])
    cnx, cny, cnz = int(part_dims[0]), int(part_dims[1]), int(part_dims[2])

    xC = np.ceil(np.tile(np.arange(1, nx + 1), ny * nz).reshape(-1, 1) / (nx / cnx)).ravel()
    yC = np.ceil(np.tile(np.repeat(np.arange(1, ny + 1), nx), nz).reshape(-1, 1) / (ny / cny)).ravel()
    zC = np.ceil(np.repeat(np.arange(1, nz + 1), nx * ny).reshape(-1, 1) / (nz / cnz)).ravel()

    p = xC + cnx * ((yC - 1) + cny * (zC - 1))
    return p.astype(int)
