"""Port of MRST ``tabulate_NWM``: count the number of identical subscripts in
``u`` (equivalent to MATLAB ``tabulate``), implemented via ``accumarray``.

NOTE: unlike the MATLAB original (which returns 1-based labels), this port
returns 0-based labels so that the result is consistent with the 0-based
grid indices used throughout PRSTCore.
"""

import numpy as np


def tabulate_NWM(u):
    """Return an ``m x 2`` array ``[label, count]`` with 0-based labels."""
    u = np.asarray(u, dtype=np.int64).ravel()
    if u.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    v = np.bincount(u, minlength=int(u.max()) + 1)
    return np.column_stack([np.arange(0, v.size, dtype=np.int64), v])
