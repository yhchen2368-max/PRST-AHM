"""Port of MRST ``pointsSingleWellNode``: generate the 2D well-region (WR)
points corresponding to a single well node.

The WR is composed of a Cartesian region and two half-radial regions in the
xy plane, used to connect the HW grid.
"""

import numpy as np


def _fTheta(x, y):
    return 2 * np.pi * np.double(np.sign(np.arctan2(y, x)) < 0) + np.arctan2(y, x)


def pointsSingleWellNode(pW, ly, ny, na, ii):
    """Generate the 2D well-region (WR) points for single (0-based) well node
    ``ii``.

    Returns a dict with fields ``'cart'`` and ``'rad'`` (2D coordinates).
    """
    ny = np.atleast_1d(np.asarray(ny, dtype=int))
    ly = np.atleast_1d(np.asarray(ly, dtype=float))
    if ny.size == 1:
        y0 = np.linspace(ly[0] / 2, 0, ny[0] // 2 + 1)
        y0 = y0[:-1]
    else:
        yI = np.linspace(ly[0] / 2, 0, ny[0] // 2 + 1)
        yI = yI[:-1]
        yO = np.logspace(np.log10(ly[0] / 2 + ly[1] / 2), np.log10(ly[0] / 2),
                         ny[1] // 2 + 1)
        yO = yO[:-1]
        y0 = np.concatenate([yO, yI])

    pW = np.asarray(pW, dtype=float)[:, :2]
    p2 = pW[ii]
    if ii == 0:
        p3 = pW[ii + 1]
        ang = _fTheta(p3[0] - p2[0], p3[1] - p2[1])
        g1 = 0.5 * np.pi
        g2 = 1.5 * np.pi
    elif ii == pW.shape[0] - 1:
        p1 = pW[ii - 1]
        ang = _fTheta(p2[0] - p1[0], p2[1] - p1[1])
        g1 = 0.5 * np.pi
        g2 = -0.5 * np.pi
    else:
        p1 = pW[ii - 1]
        p3 = pW[ii + 1]
        p12 = (p1 + p2) / 2
        p23 = (p2 + p3) / 2
        ang = _fTheta(p23[0] - p12[0], p23[1] - p12[1])

    # Rotating mapping
    M = np.array([[np.cos(ang), np.sin(ang)],
                  [-np.sin(ang), np.cos(ang)]])

    # Cartesian points
    y = np.concatenate([y0, [0], -y0[::-1]])
    xy = np.column_stack([np.zeros_like(y), y]) @ M
    xy = xy + p2

    # Radial points
    if ii == 0 or ii == pW.shape[0] - 1:
        g = np.linspace(g1, g2, na + 1)
        g = g[1:-1]
        r = y0
        xr = r[:, None] * np.cos(g)[None, :]
        yr = r[:, None] * np.sin(g)[None, :]
        xyr = np.column_stack([xr.ravel(order='F'), yr.ravel(order='F')]) @ M
        xyr = xyr + p2
    else:
        xyr = np.empty((0, 2))

    return {'cart': xy, 'rad': xyr}
