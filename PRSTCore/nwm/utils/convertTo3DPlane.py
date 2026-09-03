"""Port of MRST ``convertTo3DPlane``: map points from a horizontal (xy) plane
back to the fully 3D plane using the transformation matrices of
``convertToXYPlane``."""

import numpy as np


def convertTo3DPlane(p, T, R):
    """Convert points ``p`` (N x 3, horizontal plane) back to the 3D plane.

    ``T`` and ``R`` are the transformation matrices returned by
    ``convertToXYPlane``.
    """
    p = np.asarray(p, dtype=float)
    was_1d = p.ndim == 1
    if was_1d:
        p = p.reshape(1, -1)
    p_extend = np.column_stack([p, np.ones(p.shape[0])])
    p = p_extend @ np.linalg.inv(T @ R)
    p = p[:, :3]
    return p[0] if was_1d else p
