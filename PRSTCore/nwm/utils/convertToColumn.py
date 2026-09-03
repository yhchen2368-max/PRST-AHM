"""Port of MRST ``convertToColumn``: force an array into a column vector."""

import numpy as np


def convertToColumn(y):
    """Return ``y`` as a column vector (transpose if it is a row vector)."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    if y.shape[0] < y.shape[1]:
        return y.T
    return y
