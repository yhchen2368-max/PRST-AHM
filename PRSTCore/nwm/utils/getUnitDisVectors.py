"""Port of MRST ``getUnitDisVectors``: unit distance vectors of cells in a
Corner-point or Cartesian grid, from X- face centre to X+ face centre (and
likewise for Y and Z)."""

import numpy as np


def _meanRows(x):
    if x.shape[0] == 0:
        return np.full(x.shape[1], np.nan)
    return np.sum(x, axis=0) / x.shape[0]


def getUnitDisVectors(G, cfCentersAll, cells):
    """Return ``(ux, uy, uz)`` unit distance vectors for the given cells."""
    cells = np.asarray(cells, dtype=np.int64).ravel()
    facePos = G['cells']['facePos']

    cellFacesDir = [G['cells']['faces'][facePos[x]:facePos[x + 1], 1] for x in cells]
    cfCenters = [cfCentersAll[facePos[x]:facePos[x + 1]] for x in cells]

    cfCentersDir = [None] * 6
    for d in range(1, 7):
        tmp = [y[x == d] for x, y in zip(cellFacesDir, cfCenters)]
        tmp = [_meanRows(x) for x in tmp]
        cfCentersDir[d - 1] = np.array(tmp)

    # Unit distance vectors
    ux = cfCentersDir[1] - cfCentersDir[0]
    ux = ux / np.sqrt(np.sum(ux ** 2, axis=1))[:, None]
    uy = cfCentersDir[3] - cfCentersDir[2]
    uy = uy / np.sqrt(np.sum(uy ** 2, axis=1))[:, None]
    uz = cfCentersDir[5] - cfCentersDir[4]
    uz = uz / np.sqrt(np.sum(uz ** 2, axis=1))[:, None]
    return ux, uy, uz
