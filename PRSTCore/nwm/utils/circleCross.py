"""Port of MRST ``circleCross``: intersection points of two circles."""

import numpy as np


def circleCross(x1, y1, r1, x2, y2, r2):
    """Compute the intersection points of two circles, ``2 x 2`` array
    (``nan`` if the circles do not intersect)."""
    if y1 != y2:
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        k2 = -(x1 - x2) / (y1 - y2)
        b2 = (r1 ** 2 - r2 ** 2 - x1 ** 2 + x2 ** 2 - y1 ** 2 + y2 ** 2) / (2 * (y2 - y1))

        if abs(r2 - r1) < d < r2 + r1:
            delta = (-b2 ** 2 + r2 ** 2 + k2 ** 2 * r2 ** 2 - 2 * b2 * k2 * x2
                     - k2 ** 2 * x2 ** 2 + 2 * b2 * y2 + 2 * k2 * x2 * y2 - y2 ** 2)
            xx1 = (-b2 * k2 + x2 + k2 * y2 - np.sqrt(delta)) / (1 + k2 ** 2)
            yy1 = k2 * xx1 + b2
            xx2 = (-b2 * k2 + x2 + k2 * y2 + np.sqrt(delta)) / (1 + k2 ** 2)
            yy2 = k2 * xx2 + b2
            p = np.array([[xx1, yy1], [xx2, yy2]])
        else:
            p = np.full((2, 2), np.nan)
    else:
        p0 = np.array([[x1, y1], [x2, y2]])
        R = np.array([r1, r2])
        idx = np.argsort(p0[:, 0])
        p0 = p0[idx]
        R = R[idx]
        x1, x2, y1, y2, r1, r2 = (p0[0, 0], p0[1, 0], p0[0, 1], p0[1, 1],
                                  R[0], R[1])
        d = abs(x2 - x1)
        delta = (-d + r1 - r2) * (-d - r1 + r2) * (-d + r1 + r2) * (d + r1 + r2)
        if delta > 0:
            a = np.sqrt(delta) / d
            dx = (d ** 2 - r2 ** 2 + r1 ** 2) / (2 * d)
            p = np.array([[x1 + dx, y1 + a / 2], [x1 + dx, y1 - a / 2]])
        else:
            p = np.full((2, 2), np.nan)
    return p
