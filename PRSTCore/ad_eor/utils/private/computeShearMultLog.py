"""Port of MRST ``computeShearMultLog.m``.

PLYSHLOG shear multiplier: for each cell whose water velocity/shear-rate
exceeds the table's minimum entry and whose viscosity multiplier exceeds 1,
finds the log-log intersection of the (velocity, effective-multiplier)
curve with the target multiplier line and exponentiates back.

Ported as a direct per-cell loop, matching the ``.m`` source (only cells
with active shear are visited, so this stays cheap in practice).
"""

import numpy as _np


def computeShearMultLog(fluid, vW, muWMultf):
    vW = _np.asarray(vW, dtype=float).ravel()
    muWMultf = _np.asarray(muWMultf, dtype=float).ravel()

    plyshlog = fluid['plyshlog']
    refConcentration = _np.asarray(plyshlog['refcondition'], dtype=float).ravel()[0]
    refViscMult = float(_np.asarray(fluid['muWMult'](refConcentration)).ravel()[0])

    plyshlogTable = _np.asarray(plyshlog['data'][0], dtype=float)
    waterVel = plyshlogTable[:, 0]
    refM = plyshlogTable[:, 1]

    # Convert the table using the reference condition.
    refM = (refViscMult * refM - 1.0) / (refViscMult - 1.0)

    minWaterVel = float(_np.min(waterVel))

    iShear = _np.flatnonzero((vW > minWaterVel) & (muWMultf > 1.0))

    P = muWMultf[iShear]

    V = _np.log(waterVel)
    vW0 = _np.log(vW[iShear])

    def f(x, y, x0):
        return x + y - x0

    z = _np.ones(iShear.size)

    for i in range(iShear.size):
        Z = (1.0 + (P[i] - 1.0) * refM) / P[i]
        Z = _np.log(Z)

        sign = f(V, Z, vW0[i])

        temp = sign[:-1] * sign[1:]
        j = _np.flatnonzero(temp <= 0)

        assert j.size <= 1, 'more than one intersection point found'

        if j.size == 1:
            j0 = int(j[0])
            _, zi = _find_intersection(
                _np.array([[V[j0], Z[j0]], [V[j0 + 1], Z[j0 + 1]]]),
                _np.array([[0.0, vW0[i]], [vW0[i], 0.0]]))
            z[i] = zi
        else:
            # Out of the table range; since small values are already
            # handled, this must be a large value.
            assert vW0[i] - Z[-1] > V[-1]
            z[i] = Z[-1]

    z = _np.exp(z)

    zSh = _np.ones(vW.size)
    zSh[iShear] = z
    return zSh


def _find_intersection(l1, l2):
    """Intersection of line segment ``l1`` (2x2: [x1 y1; x2 y2]) with the
    straight line through ``l2`` (2x2: two points on the line)."""
    x1, y1 = l1[0]
    x2, y2 = l1[1]
    x3, y3 = l2[0]
    x4, y4 = l2[1]

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    assert d != 0

    x = ((x3 - x4) * (x1 * y2 - y1 * x2) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    y = ((y3 - y4) * (x1 * y2 - y1 * x2) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d

    assert min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
    return x, y
