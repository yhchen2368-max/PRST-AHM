"""Extract fine-grid faces constituting coarse grid faces.

1:1 Python translation of MRST multiscale/coarsegrid/subFaces.m
"""

import numpy as np


def sub_faces(g, cg):
    """Extract fine-grid faces belonging to each coarse face.

    Parameters
    ----------
    g : dict
        Fine grid.
    cg : dict
        Coarse grid with faces.fconn, faces.connPos.

    Returns
    -------
    nsub : ndarray
        Number of fine faces per coarse face.
    sub : ndarray
        Fine face indices in condensed storage format.
    """
    nsub = np.diff(cg["faces"]["connPos"])
    sub = np.asarray(cg["faces"]["fconn"])
    return nsub, sub
