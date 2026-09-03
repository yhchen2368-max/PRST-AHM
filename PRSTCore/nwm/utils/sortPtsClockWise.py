"""Port of MRST ``sortPtsClockWise``: sort the points of each element
(specified by the connectivity list ``t``) in clockwise order."""

import numpy as np


def sortPtsClockWise(p, t):
    """Sort the points in clockwise order for each element of ``t``.

    ``p`` is the 2D point set; ``t`` is a list of node-id arrays.  The list
    ``t`` is updated in place and also returned.
    """
    def fTheta(x, y):
        return 2 * np.pi * np.double(np.sign(np.arctan2(y, x)) < 0) + np.arctan2(y, x)

    for k in range(len(t)):
        t0 = np.asarray(t[k], dtype=np.int64)
        xy = p[t0, :] - np.mean(p[t0, :], axis=0)
        theta = fTheta(xy[:, 0], xy[:, 1])
        i = np.argsort(theta)[::-1]  # descending
        t[k] = t0[i]
    return t
