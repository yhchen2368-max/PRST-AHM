"""Partition grid uniformly in logical space.

1:1 Python translation of MRST multiscale/coarsegrid/partitionUI.m
"""

import numpy as np


def partition_ui(G, coarse_dim):
    """Partition a corner-point grid uniformly in index space.

    Parameters
    ----------
    G : dict
        Grid with cartDims and cells.indexMap.
    coarse_dim : tuple
        Number of coarse blocks in each direction (length 2 or 3).

    Returns
    -------
    ndarray
        Partition vector (1..PROD(coarse_dim)).
    """
    if "cartDims" not in G:
        raise ValueError("Grid must have 'cartDims' to use partitionUI")

    coarse_dim = np.atleast_1d(coarse_dim).ravel()

    if len(coarse_dim) != G.get("griddim", len(G["cartDims"])) or np.any(coarse_dim < 1):
        raise ValueError("coarse_dim must match grid dimensions")

    assert np.all(coarse_dim > 0)
    assert np.all(coarse_dim <= G["cartDims"])

    cart_dims = np.array(G["cartDims"], dtype=int)
    index_map = np.asarray(G["cells"]["indexMap"], dtype=int)

    # Convert linear index to IJK subscripts
    ijk = np.column_stack(np.unravel_index(index_map - 1, tuple(cart_dims)))
    M = ijk.max(axis=0) - ijk.min(axis=0) + 1

    nc = G["cells"]["num"]
    block_ix = np.zeros(nc, dtype=int)

    for d in range(len(coarse_dim) - 1, -1, -1):
        B = int(coarse_dim[d])
        block_ix = _lb_lin_dist(ijk[:, d] - ijk[:, d].min(), int(M[d]), B) + B * block_ix

    return block_ix + 1


def _lb_lin_dist(f, M, B):
    """Load-balanced linear distribution."""
    L = M // B
    R = M % B
    f = np.asarray(f, dtype=int)
    return np.maximum(np.floor(f / (L + 1)), np.floor((f - R) / L)).astype(int)
