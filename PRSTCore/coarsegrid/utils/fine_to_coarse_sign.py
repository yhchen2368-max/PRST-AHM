"""Compute sign change between fine faces and coarse faces.

1:1 Python translation of MRST multiscale/coarsegrid/utils/fineToCoarseSign.m
"""

import numpy as np


def fine_to_coarse_sign(cg):
    """Compute sign change between fine and coarse faces.

    Parameters
    ----------
    cg : dict
        Coarse grid with parent, faces.connPos, faces.fconn, faces.neighbors,
        partition.

    Returns
    -------
    ndarray
        Sign (+1 or -1) for each fine face in cg.faces.fconn.
    """
    parent = cg["parent"]
    faceno = np.repeat(np.arange(cg["faces"]["num"]),
                       np.diff(cg["faces"]["connPos"]))
    # p is 1-based partition; neighbor cell IDs are 0-based (0 = boundary)
    p_ext = np.concatenate([[0], cg["partition"]])
    fconn = np.asarray(cg["faces"]["fconn"])
    c1 = parent["faces"]["neighbors"][fconn, 0]
    b1 = cg["faces"]["neighbors"][faceno, 0]
    c1_coarse = np.array([p_ext[ci] if ci > 0 else 0 for ci in c1])
    sgn = 2 * (c1_coarse == b1).astype(int) - 1
    return sgn
