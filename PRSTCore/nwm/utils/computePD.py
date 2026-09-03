"""Port of MRST ``computePD``: dimensionless pressure for a well arbitrarily
located inside a rectangular box (infinite-series solution)."""

import numpy as np


def computePD(x, y, a, b, xw, yw):
    """Dimensionless pressure at ``(x, y)`` for a well located inside a
    rectangular box of size ``a x b``; the well is ``xw`` from the right
    boundary and ``yw`` from the lower boundary."""
    N = 10
    nn = np.arange(-N, N + 1)
    p1 = np.pi / b * (x - 2 * nn * a)
    p2 = np.pi / b * (x - 2 * nn * a - 2 * xw)

    q1 = np.pi / b * y
    q2 = np.pi / b * (y + 2 * yw)
    s1 = np.cosh(p1) - np.cos(q2)
    s2 = np.cosh(p2) - np.cos(q1)
    s3 = np.cosh(p1) - np.cos(q1)
    s4 = np.cosh(p2) - np.cos(q2)

    f = np.log((s1 * s2) / (s3 * s4))
    return float(np.sum(f))
