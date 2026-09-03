"""Port of MRST ``getConnListAndBdyNodeWR2D``: get the connectivity list and
boundary nodes of the 2D well region (WR), composed of a Cartesian region
and two half-radial regions in the xy plane."""

import numpy as np

from .makeConnListFromMat import makeConnListFromMat


def getConnListAndBdyNodeWR2D(p, ny, na):
    """Get connectivity list and boundary nodes of the 2D well region.

    Parameters
    ----------
    p : list of dicts
        Points of the WR region (one entry per well node), as returned by
        ``pointsSingleWellNode``.
    ny : int
        The number of Cartesian cells in the Y direction.
    na : int
        The number of angular cells in the radial region.

    Returns
    -------
    t : list of 1D arrays
        Connectivity list of the whole well region.
    tC : list of 1D arrays
        Connectivity list of the Cartesian region.
    bn : ndarray
        Indices of the outer boundary nodes of the whole well region.
    bnC : ndarray
        Indices of the outer boundary nodes of the Cartesian region.
    """
    # Number of Cartesian nodes
    nny = ny + 1
    nnx = len(p)
    nnxy = nny * nnx

    # Number of radial nodes
    nr = ny // 2
    nnra = nr * (na - 1)

    # Cartesian node indices, reshaped to 2D (column-major)
    ndC = np.arange(nny * nnx, dtype=np.int64).reshape(nny, nnx, order='F')

    # Connectivity list for the radial nodes, Heel
    ndR1 = (nnxy - 1 + np.arange(1, nnra + 1, dtype=np.int64)).reshape(nr, na - 1, order='F')
    ndR1 = np.hstack([np.arange(nr, dtype=np.int64).reshape(-1, 1),
                      ndR1,
                      np.arange(nny - 1, nr, -1, dtype=np.int64).reshape(-1, 1)])
    ndR1 = np.vstack([ndR1, np.full((1, ndR1.shape[1]), nr, dtype=np.int64)])

    # Connectivity list for the radial nodes, Toe
    ndR2 = (nnxy + nnra - 1 + np.arange(1, nnra + 1, dtype=np.int64)).reshape(nr, na - 1, order='F')
    d = (nnx - 1) * nny
    ndR2 = np.hstack([(np.arange(nr, dtype=np.int64) + d).reshape(-1, 1),
                      ndR2,
                      (np.arange(nny - 1, nr, -1, dtype=np.int64) + d).reshape(-1, 1)])
    ndR2 = np.vstack([ndR2, np.full((1, ndR2.shape[1]), nr + d, dtype=np.int64)])

    # Connectivity list for the Cartesian nodes
    tC = makeConnListFromMat(ndC)
    tR1 = makeConnListFromMat(ndR1)
    tR2 = makeConnListFromMat(ndR2)

    # Combine the connectivity lists
    t = tC + tR1 + tR2
    for k in range(len(t)):
        if len(t[k]) != len(np.unique(t[k])):
            t[k] = np.unique(t[k])

    # Get boundary nodes, with radial region
    bn = np.concatenate([ndC[0, :],
                         ndR2[0, 1:-1],
                         ndC[-1, ::-1],
                         ndR1[0, -2:0:-1]])

    # Get boundary nodes, without radial region
    bnC = np.concatenate([ndC[0, :],
                          ndC[1:-1, -1],
                          ndC[-1, ::-1],
                          ndC[-2:0:-1, 0]])
    return t, tC, bn, bnC
