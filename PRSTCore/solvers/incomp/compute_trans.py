"""Python port of MRST's ``computeTrans.m`` (mrst-2026a/core/solvers).

Half-transmissibilities: one value per row of ``G['cells']['faces']``
(i.e. per half-face), computed from the two-point flux formula

    T_hf = (C . K . N) / (C . C)

where ``C`` is the vector from the cell centroid to its face's centroid,
``N`` is the outward-pointing face normal (length == face area, as
guaranteed by :func:`PRSTCore.gridprocessing.compute_geometry`), and ``K``
is the cell's permeability tensor.
"""

from __future__ import annotations

import numpy as np


def _perm_tensor(perm: np.ndarray, griddim: int) -> np.ndarray:
    """Expand ``rock['perm']`` (ncell x {1, griddim, ntensor}) to a full
    ``(ncell, griddim, griddim)`` symmetric tensor, matching MRST's
    ``permTensor.m`` column conventions (scalar / diagonal / full symmetric)."""
    perm = np.asarray(perm, dtype=float)
    if perm.ndim == 1:
        perm = perm.reshape(-1, 1)
    nc, ncol = perm.shape
    K = np.zeros((nc, griddim, griddim))

    if ncol == 1:
        for d in range(griddim):
            K[:, d, d] = perm[:, 0]
    elif ncol == griddim:
        for d in range(griddim):
            K[:, d, d] = perm[:, d]
    elif griddim == 2 and ncol == 3:
        K[:, 0, 0], K[:, 0, 1], K[:, 1, 1] = perm[:, 0], perm[:, 1], perm[:, 2]
        K[:, 1, 0] = K[:, 0, 1]
    elif griddim == 3 and ncol == 6:
        K[:, 0, 0], K[:, 0, 1], K[:, 0, 2] = perm[:, 0], perm[:, 1], perm[:, 2]
        K[:, 1, 1], K[:, 1, 2] = perm[:, 3], perm[:, 4]
        K[:, 2, 2] = perm[:, 5]
        K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = K[:, 0, 1], K[:, 0, 2], K[:, 1, 2]
    else:
        raise ValueError(f"Unsupported rock['perm'] shape {perm.shape} for griddim={griddim}")
    return K


def compute_trans(G: dict, rock: dict, *, fix_negative: bool = True) -> np.ndarray:
    """Port of MRST ``computeTrans.m`` (``K_system='xyz'`` path only)."""
    griddim = int(G["griddim"])
    nc = G["cells"]["num"]

    face_pos = G["cells"]["facePos"]
    cell_faces = np.asarray(G["cells"]["faces"])
    cf = cell_faces[:, 0]
    cellNo = np.repeat(np.arange(nc), np.diff(face_pos))

    cell_centroids = np.asarray(G["cells"]["centroids"])
    face_centroids = np.asarray(G["faces"]["centroids"])
    face_normals = np.asarray(G["faces"]["normals"])
    neighbors = np.asarray(G["faces"]["neighbors"])

    C = face_centroids[cf] - cell_centroids[cellNo]
    sgn = np.where(neighbors[cf, 0] == cellNo, 1.0, -1.0)
    N = face_normals[cf] * sgn[:, None]

    K = _perm_tensor(np.asarray(rock["perm"], dtype=float), griddim)
    Khf = K[cellNo]  # (nhf, griddim, griddim)

    T = np.einsum("hi,hij,hj->h", C, Khf, N) / np.sum(C * C, axis=1)

    if fix_negative:
        T = np.abs(T)

    if "ntg" in rock and cell_faces.shape[1] == 2:
        dim_tag = np.ceil(cell_faces[:, 1] / 2.0).astype(int)
        ntg = np.asarray(rock["ntg"], dtype=float).ravel()[cellNo]
        mult = np.where(dim_tag == 3, 1.0, ntg)
        T = T * mult

    return T
