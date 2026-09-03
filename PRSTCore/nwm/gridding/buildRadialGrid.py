"""Port of MRST ``buildRadialGrid``: build a 2D radial grid from point and
dimension definitions."""

import numpy as np

from .._core import tessellationGrid
from .makeConnListFromMat import makeConnListFromMat


def buildRadialGrid(p, nA, nR):
    """Build the 2D radial grid from point and dimension definitions.

    Parameters
    ----------
    p : ndarray
        2D node coordinates, obeying the logical numbering (angularly cycle
        fastest, then radially).
    nA : int
        Angular cell dimension.
    nR : int
        Radial cell dimension.

    Returns
    -------
    G : dict
        The 2D radial grid.  Each cell has four faces with face types:
        face 1: Radial -, face 2: Angular +, face 3: Radial +, face 4:
        Angular -.
    t : list of 1D arrays
        Connectivity list.
    """
    p = np.asarray(p, dtype=float)
    np_ = p.shape[0]
    nd = np.arange(np_, dtype=np.int64).reshape(nA, nR + 1, order='F')
    nd = np.vstack([nd, nd[0]])
    t = makeConnListFromMat(nd)
    G = tessellationGrid(p, t)
    G['radDims'] = [nA, nR]
    a, r = np.unravel_index(np.arange(G['cells']['num']), (nA, nR), order='F')
    G['radIndices'] = np.column_stack([a, r])
    G['type'] = list(G.get('type', [])) + ['buildRadialGrid']
    return G, t
