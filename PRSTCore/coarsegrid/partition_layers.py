"""Partition grid uniformly in (I,J) direction, non-uniformly in K.

1:1 Python translation of MRST multiscale/coarsegrid/partitionLayers.m
"""

import numpy as np


def partition_layers(G, coarse_dim, L):
    """Partition a corner-point grid with layer-based grouping.

    Parameters
    ----------
    G : dict
        Grid with cells.ijkMap or cartDims + cells.indexMap.
    coarse_dim : list of 2 ints
        Number of coarse blocks in I and J directions.
    L : list of int
        Run-length-encoded layer boundaries, e.g., [1, 32, 76, ...].

    Returns
    -------
    ndarray
        Partition vector (1..N), length = G.cells.num.
    """
    if not _grid_ok(G):
        raise ValueError("Grid is not a valid grid_structure")

    if len(coarse_dim) != 2:
        raise ValueError("coarse_dim must be length 2")

    # Get IJK indices
    if "ijkMap" in G["cells"]:
        ijk = np.asarray(G["cells"]["ijkMap"], dtype=float)
        M = ijk.max(axis=0).astype(int)
    else:
        ijk_cols = np.unravel_index(
            np.asarray(G["cells"]["indexMap"], dtype=int) - 1,
            tuple(G["cartDims"]),
        )
        ijk = np.column_stack(ijk_cols).astype(float)
        M = np.array(G["cartDims"], dtype=int)

    # Layer-based partition in K
    L_arr = np.asarray(L, dtype=int).ravel()
    # rldecode: repeat [0,1,2,...,len(L)-1] with counts = diff(L)
    diffs = np.diff(L_arr)
    l_num = np.repeat(np.arange(len(diffs)), diffs)  # 0-based layer index
    k_idx = (ijk[:, 2] - 1).astype(int)
    k_idx = np.clip(k_idx, 0, len(l_num) - 1)
    block_ix = l_num[k_idx].astype(int)

    # Uniform partition in J, I
    for d in [1, 0]:  # J then I
        B = coarse_dim[d]
        block_ix = _lb_lin_dist(ijk[:, d].astype(int) - 1, M[d], B) + B * block_ix

    return block_ix + 1


def _lb_lin_dist(f, M, B):
    """Load-balanced linear distribution.

    Maps index set (0..M-1) to blocks (0..B-1).
    """
    L = M // B
    R = M % B
    f = np.asarray(f, dtype=int)
    result = np.maximum(np.floor(f / (L + 1)), np.floor((f - R) / L))
    return result.astype(int)


def _grid_ok(G):
    """Check if G is a valid grid structure."""
    if G is None or not isinstance(G, dict):
        return False
    if "cells" not in G or G["cells"] is None:
        return False
    c = G["cells"]
    if "ijkMap" in c and len(c["ijkMap"]) == c["num"]:
        return True
    if "cartDims" in G and "indexMap" in c:
        return True
    return False
