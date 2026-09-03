"""Port of MRST ``polyintersect`` (Gerben J. de Boer, Deltares): finds all
intersections of two polygons analytically over the (extended) line pieces and
discards those outside the support points of each segment.

Only the computational core is ported (no debug figures).
"""

import numpy as np


def polyintersect(x1, y1, x2, y2, *args):
    """Return ``(xr, yr)`` - all intersections between the line pieces of
    polygons ``(x1, y1)`` and ``(x2, y2)``.

    Optional trailing arguments may be ``'debug', 0/1`` / ``'disp', 0/1``
    key/value pairs; stray positional numbers are ignored (legacy MRST
    call signature ``polyintersect(..., 2)``).
    """
    OPT = {'debug': 0, 'disp': 0}
    i = 0
    while i < len(args):
        if isinstance(args[i], str):
            key = args[i].lower()
            if key in ('debug', 'disp'):
                OPT[key] = args[i + 1]
                i += 2
            else:
                raise ValueError(f'Invalid string argument: {args[i]}')
        else:
            i += 1

    x1 = np.asarray(x1, dtype=float).ravel()
    y1 = np.asarray(y1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()

    with np.errstate(divide='ignore', invalid='ignore'):
        # y = a*x + b, one line piece per corner interval
        a1 = np.diff(y1) / np.diff(x1)          # Inf for vertical lines
        b1 = y1[:-1] - x1[:-1] * a1             # NaN/-Inf for vertical lines
        n1 = a1.size
        a2 = np.diff(y2) / np.diff(x2)
        b2 = y2[:-1] - x2[:-1] * a2
        n2 = a2.size

    cross_x = []
    cross_y = []
    eps = 1e-4

    for imesh1 in range(n1):
        for imesh2 in range(n2):
            with np.errstate(divide='ignore', invalid='ignore'):
                local_x = (b2[imesh2] - b1[imesh1]) / (a1[imesh1] - a2[imesh2])
                local_y = a1[imesh1] * local_x + b1[imesh1]

            if np.isinf(local_x) or np.isnan(local_x):
                # Two parallel vertical lines: no crossing
                if np.isinf(a1[imesh1]) and np.isinf(a2[imesh2]):
                    local_x = np.nan
                    local_y = np.nan
                # Two parallel (non-vertical) lines: no crossing
                elif a1[imesh1] == a2[imesh2]:
                    local_x = np.nan
                    local_y = np.nan
                # One vertical line in polygon 1
                elif np.isinf(a1[imesh1]):
                    local_x = x1[imesh1]
                    local_y = a2[imesh2] * local_x + b2[imesh2]
                # One vertical line in polygon 2
                elif np.isinf(a2[imesh2]):
                    local_x = x2[imesh2]
                    local_y = a1[imesh1] * local_x + b1[imesh1]

            # Discard crossings outside the line pieces
            valid = not (np.isnan(local_x) or np.isnan(local_y))
            if valid:
                if local_x < min(x1[imesh1], x1[imesh1 + 1]) - eps:
                    valid = False
                elif local_x > max(x1[imesh1], x1[imesh1 + 1]) + eps:
                    valid = False
                elif local_y < min(y1[imesh1], y1[imesh1 + 1]) - eps:
                    valid = False
                elif local_y > max(y1[imesh1], y1[imesh1 + 1]) + eps:
                    valid = False
                elif local_x < min(x2[imesh2], x2[imesh2 + 1]) - eps:
                    valid = False
                elif local_x > max(x2[imesh2], x2[imesh2 + 1]) + eps:
                    valid = False
                elif local_y < min(y2[imesh2], y2[imesh2 + 1]) - eps:
                    valid = False
                elif local_y > max(y2[imesh2], y2[imesh2 + 1]) + eps:
                    valid = False

            if valid:
                cross_x.append(local_x)
                cross_y.append(local_y)

    return np.array(cross_x), np.array(cross_y)
