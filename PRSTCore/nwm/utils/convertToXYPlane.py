"""Port of MRST ``convertToXYPlane``: map points from a fully 3D plane to the
horizontal xy plane (and back via ``convertTo3DPlane``)."""

import numpy as np

from .._core import faceNormals, mergeOptions


def convertToXYPlane(pts1, n1, pts2, **kwargs):
    """Convert points ``pts1`` and ``pts2`` from a fully 3D plane to the
    horizontal xy plane.  The plane is specified by ``pts1[n1[0]]``,
    ``pts1[n1[1]]`` and ``pts1[n1[2]]``.

    Returns ``(pts1, pts2, T, R, nor_z)``.
    """
    opt = mergeOptions({'normalZ': None}, **kwargs)
    n1 = np.asarray(n1, dtype=np.int64)
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    if pts2.ndim == 1:
        pts2 = pts2.reshape(1, -1)

    # Get normal z
    nz = opt['normalZ']
    if nz is not None and np.asarray(nz).size > 0:
        nor_z = np.asarray(nz, dtype=float).ravel()
    else:
        pFacez = pts1[n1[0:3]]
        nor_z = faceNormals(pFacez)

    # Get normal x
    nor_x = pts1[n1[1]] - pts1[n1[0]]
    nor_x = nor_x / np.linalg.norm(nor_x, 2)

    # Get normal y
    pFacey = np.vstack([nor_x, nor_z, np.zeros(3)])
    nor_y = faceNormals(pFacey)

    # Base point is the origin
    x0, y0, z0 = 0.0, 0.0, 0.0

    # Shift
    T = np.eye(4)
    T[3, [0, 1, 2]] = [-x0, -y0, -z0]

    # Rotate
    R = np.zeros((4, 4))
    R[:3, :3] = np.column_stack([nor_x, nor_y, nor_z])
    R[3, 3] = 1.0

    # Transform
    pts1_extend = np.column_stack([pts1, np.ones(pts1.shape[0])])
    pts1 = pts1_extend @ T @ R
    pts1 = pts1[:, :3]

    pts2_extend = np.column_stack([pts2, np.ones(pts2.shape[0])])
    pts2 = pts2_extend @ T @ R
    pts2 = pts2[:, :3]

    return pts1, pts2, T, R, nor_z
