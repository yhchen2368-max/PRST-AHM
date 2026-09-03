"""Port of MRST ``tri_area`` (from the ``hfm`` module): triangle area given
the coordinates of its vertices using Heron's formula."""

import numpy as np


def tri_area(P1, P2, P3):
    """Calculate the triangle area given the coordinates of its vertices
    ``P1``, ``P2`` and ``P3`` using Heron's formula."""
    diff1 = P1 - P2
    diff2 = P1 - P3
    diff3 = P3 - P2
    a = np.linalg.norm(diff1)
    b = np.linalg.norm(diff2)
    c = np.linalg.norm(diff3)

    # sort the elements
    v = np.sort([a, b, c])
    a, b, c = v[2], v[1], v[0]

    temp = b + c
    v1 = a + temp   # 2s
    temp = a - b
    v2 = c - temp   # 2*(s-a)
    v3 = c + temp   # 2*(s-b)
    temp = b - c
    v4 = a + temp   # 2*(s-c)
    area = 0.25 * np.sqrt(abs(v1 * v2 * v3 * v4))
    return area
