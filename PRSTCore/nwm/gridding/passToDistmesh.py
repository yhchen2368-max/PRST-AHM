"""Port of MRST ``passToDistmesh``: generate the parameters passed to
``distmesh_2d_nwm``."""

import numpy as np

from .._core import ddiff, dpoly, inpolygon, mergeOptions
from .distmesh_2d_nwm import distmesh_2d_nwm


def pointsInCircle(p, pc, r):
    """Return whether each point of ``p`` lies inside the circle ``(pc, r)``."""
    d = p - pc
    d = np.sqrt(np.sum(d ** 2, axis=1))
    return d <= r


def passToDistmesh(pIB, pOB, multiplier, maxIter, **kwargs):
    """Generate parameters passed to ``distmesh_2d_nwm``.

    Returns ``(pdis, fd)`` where ``fd`` is the signed distance function of
    the region between the inner boundary ``pIB`` and the outer boundary
    ``pOB``.
    """
    opt = mergeOptions({'pIBRadius': None}, **kwargs)

    pIB = np.asarray(pIB, dtype=float)
    pOB = np.asarray(pOB, dtype=float)
    pIB = np.vstack([pIB, pIB[0]])
    pOB = np.vstack([pOB, pOB[0]])

    assert np.all(inpolygon(pIB[:, 0], pIB[:, 1], pOB[:, 0], pOB[:, 1])), \
        'The well region boundary is outside the VOI region, please ' \
        'enlarge the VOI boundary'

    fdI = lambda p: dpoly(p, pIB)
    fdO = lambda p: dpoly(p, pOB)
    fd = lambda p: ddiff(fdO(p), fdI(p))

    lIB = pIB[1:] - pIB[:-1]
    lIB = np.sqrt(np.sum(lIB ** 2, axis=1))
    lIBave = np.mean(lIB)

    lOB = pOB[1:] - pOB[:-1]
    lOB = np.sqrt(np.sum(lOB ** 2, axis=1))
    lOBave = np.mean(lOB)

    a = multiplier
    fh = lambda p: np.minimum(a * fdI(p) + lIBave, lOBave)
    h0 = lIBave

    pfix = np.vstack([pIB[:-1], pOB[:-1]])

    x_min = np.min(pOB[:, 0])
    x_max = np.max(pOB[:, 0])
    y_min = np.min(pOB[:, 1])
    y_max = np.max(pOB[:, 1])
    bbox = np.array([[x_min, y_min], [x_max, y_max]])

    print('    * Dist Mesh iteration information: ')
    pdis, _ = distmesh_2d_nwm(fd, fh, h0, bbox, maxIter, pfix, False)

    # Remove points too close to the boundary
    lIBmax = np.max(lIB)
    lOBmax = np.max(lOB)

    pdisG = pdis[pfix.shape[0]:, :]
    idx = np.abs(fdI(pdisG)) <= np.abs(fdO(pdisG))

    pdisI = pdisG[idx, :]
    if opt['pIBRadius'] is None:
        idxI = np.abs(fdI(pdisI)) > lIBmax / 2
    else:
        idxI1 = np.abs(fdI(pdisI)) > 0.1
        in_ = np.zeros((pdisI.shape[0], pIB.shape[0] - 1), dtype=bool)
        for i in range(pIB.shape[0] - 1):
            in_[:, i] = pointsInCircle(pdisI, pIB[i], opt['pIBRadius'][i])
        idxI2 = ~np.all(in_, axis=1)
        idxI = idxI1 & idxI2
    pdisI = pdisI[idxI, :]

    pdisO = pdisG[~idx, :]
    idxO = np.abs(fdO(pdisO)) > lOBmax / 2
    pdisO = pdisO[idxO, :]

    pdis = np.vstack([pfix, pdisI, pdisO])
    return pdis, fd
