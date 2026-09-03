"""Invert partition to create block-to-cell mapping.

1:1 Python translation of MRST multiscale/coarsegrid/utils/invertPartition.m
"""

import numpy as np


def invert_partition(p):
    """Create block-to-cell mapping from cell-to-block partition.

    Parameters
    ----------
    p : ndarray
        Partition vector (cell -> block).

    Returns
    -------
    b2c_pos : ndarray
        Indirection map of size [max(p) + 1].
    b2c : ndarray
        Inverse cell map: cells of block b are b2c[b2c_pos[b]:b2c_pos[b+1]].
    locno : ndarray
        Local cell numbers within each block.
    """
    p = np.asarray(p, dtype=int)
    nblocks = p.max()
    ncells = len(p)

    # Sort cells by block
    order = np.argsort(p)
    sorted_p = p[order]

    # Count cells per block
    counts = np.bincount(sorted_p, minlength=nblocks + 1)[1:]
    b2c_pos = np.concatenate([[0], np.cumsum(counts)])
    b2c = order

    # Local numbering
    locno = np.zeros(ncells, dtype=int)
    for b in range(1, nblocks + 1):
        cells_in_block = order[b2c_pos[b - 1]:b2c_pos[b]]
        locno[cells_in_block] = np.arange(len(cells_in_block))

    return b2c_pos, b2c, locno
