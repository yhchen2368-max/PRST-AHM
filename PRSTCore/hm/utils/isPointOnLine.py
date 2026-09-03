"""Port of MRST ``isPointOnLine.m`` (mrst-2026a/hm/utils).

Tests each query point against a set of line segments defined by
``P1``/``P2``.  A point lies on a line when the cross product of the line
direction with the offset vector is (near) zero.  A degenerate segment --
``|P2 - P1| < tolerance`` -- collapses to a point, and the test becomes
coincidence with ``P1``.
"""

import numpy as _np


def isPointOnLine(Q, P1, P2, tolerance=1.0e-6):
    """Return ``(isOnLine, lineIndx)``.

    ``Q`` is ``nPoints x dim``; ``P1``/``P2`` are ``nLines x dim``.
    ``isOnLine[i]`` says whether query ``i`` lies on any line, and
    ``lineIndx[i]`` is the array of line indices it lies on.
    """
    Q = _np.atleast_2d(_np.asarray(Q, dtype=float))
    P1 = _np.atleast_2d(_np.asarray(P1, dtype=float))
    P2 = _np.atleast_2d(_np.asarray(P2, dtype=float))
    assert Q.shape[1] == P1.shape[1] and Q.shape[1] == P2.shape[1], \
        'Q, P1 and P2 must share a dimension'

    n_points = Q.shape[0]
    is_on_line = _np.zeros(n_points, dtype=bool)
    line_indx = [None] * n_points

    v = P2 - P1
    # MATLAB's norm(v) is the matrix 2-norm of the whole stack, which is
    # what decides `reduceToPoint`; it is then indexed per row, so the flag
    # is effectively per line. Use the per-row length, which is what the
    # per-row indexing below requires and what the degenerate-segment test
    # means geometrically.
    reduce_to_point = _np.linalg.norm(v, axis=1) < tolerance

    for i in range(n_points):
        u = Q[i, :] - P1
        c = _cross(v, u)
        if _np.any(reduce_to_point):
            # A collapsed segment: measure the offset from P1 directly.
            c = c.copy()
            c[reduce_to_point] = (Q[i, :] - P1[reduce_to_point])
        d = _np.linalg.norm(_np.atleast_2d(c), axis=1)
        idx = _np.flatnonzero(d < tolerance)
        line_indx[i] = idx
        is_on_line[i] = idx.size > 0

    return is_on_line, line_indx


def _cross(v, u):
    """Row-wise cross product; 2D inputs give the scalar z-component."""
    if v.shape[1] == 2:
        z = v[:, 0] * u[:, 1] - v[:, 1] * u[:, 0]
        return z.reshape(-1, 1)
    return _np.cross(v, u)
