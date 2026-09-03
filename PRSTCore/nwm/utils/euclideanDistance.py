"""Port of MRST ``euclideanDistance``: pairwise Euclidean distances between
two point sets (equivalent to MATLAB ``pdist2(X, Y, 'euclidean')``)."""

import numpy as np


def euclideanDistance(X, Y):
    """``D[i, j] = ||X[i, :] - Y[j, :]||_2``."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    D = np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * (X @ Y.T)
    D = np.abs(D)
    D = np.sqrt(D)
    return D
