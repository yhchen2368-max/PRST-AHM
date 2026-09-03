"""Construct face partition from cell partition.

1:1 Python translation of MRST multiscale/coarsegrid/cellPartitionToFacePartition.m
"""

import numpy as np


def cell_partition_to_face_partition(g, p, all_boundary_faces=False):
    """Define unique partition ID for each pair of cell partition IDs.

    Parameters
    ----------
    g : dict
        Grid with faces.neighbors, faces.num.
    p : ndarray
        Cell partition vector.
    all_boundary_faces : bool
        Whether to assign unique IDs to all boundary faces.

    Returns
    -------
    ndarray
        Face partition (one integer per face). Zero for interior-block faces.
    """
    nf = g["faces"]["num"]
    nbrs = np.asarray(g["faces"]["neighbors"])
    # p is 1-based partition; nbrs are 0-based (0 = boundary)
    p_ext = np.concatenate([[0], p])

    # Build coarse block pair for each face
    pairs = np.zeros((nf, 2), dtype=int)
    for f in range(nf):
        n1, n2 = nbrs[f, 0], nbrs[f, 1]
        pairs[f, 0] = p_ext[n1] if n1 > 0 else 0
        pairs[f, 1] = p_ext[n2] if n2 > 0 else 0
    pairs = np.sort(pairs, axis=1)

    # Sort pairs and run-length encode
    idx = np.lexsort((pairs[:, 1], pairs[:, 0]))
    sorted_pairs = pairs[idx]
    _, n = np.unique(sorted_pairs, axis=0, return_counts=True)

    pf = np.zeros(nf, dtype=int)
    pf[idx] = np.repeat(np.arange(1, len(n) + 1), n)

    if all_boundary_faces:
        boundary = np.any(nbrs == 0, axis=1)
        pf[boundary] = np.arange(pf.max() + 1, pf.max() + boundary.sum() + 1)

    return pf
