"""Python port of MRST's ``tensorGrid.m`` (mrst-2026a/core/gridprocessing).

Builds the topology (cells/faces/nodes connectivity) of an axis-aligned
grid with variable cell sizes along each coordinate direction.  Geometry
(volumes, centroids, areas, normals) is added separately by
:func:`PRSTCore.gridprocessing.compute_geometry.compute_geometry`, mirroring
MRST's split between ``tensorGrid``/``cartGrid`` and ``computeGeometry``.

Index convention (differs from MRST, which is 1-based):
  - cells/faces/nodes are 0-based.
  - ``G['faces']['neighbors']`` uses ``-1`` (not ``0``) to mark "no cell on
    this side" (a boundary face), matching the convention already used
    elsewhere in PRSTCore (e.g. ``ad_core/operators.py``).
  - ``G['cells']['faces'][:, 1]`` keeps MRST's 1-6 direction tags
    (W, E, S, N, T, B) unchanged, since these are direction labels rather
    than array indices.
"""

from __future__ import annotations

import numpy as np


def _flat(a: np.ndarray) -> np.ndarray:
    """Column-major (Fortran-order) flatten, matching MATLAB's ``reshape``."""
    return a.ravel(order="F")


def _tensor_grid_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    sx = x.size - 1
    numC = sx
    numN = sx + 1
    numF = numN

    left = np.arange(-1, sx, dtype=np.int64)
    right = np.concatenate([np.arange(0, sx, dtype=np.int64), np.array([-1], dtype=np.int64)])
    neighbors = np.column_stack([left, right])

    face_pos = np.arange(0, (numC + 1) * 2, 2, dtype=np.int64)
    node_pos = np.arange(0, numF + 1, dtype=np.int64)

    cell_faces_idx = np.empty(numC * 2, dtype=np.int64)
    cell_faces_idx[0::2] = np.arange(numC, dtype=np.int64)
    cell_faces_idx[1::2] = np.arange(1, numC + 1, dtype=np.int64)
    cell_faces_tag = np.tile(np.array([1, 2], dtype=np.int64), numC)
    cell_faces = np.column_stack([cell_faces_idx, cell_faces_tag])

    return {
        "cells": {
            "num": int(numC),
            "facePos": face_pos,
            "indexMap": np.arange(numC, dtype=np.int64),
            "faces": cell_faces,
        },
        "faces": {
            "num": int(numF),
            "nodePos": node_pos,
            "neighbors": neighbors,
            "tag": np.zeros(numF, dtype=np.int64),
            "nodes": np.arange(numF, dtype=np.int64),
        },
        "nodes": {"num": int(numN), "coords": x.reshape(-1, 1)},
        "cartDims": np.array([sx], dtype=np.int64),
        "type": ["tensorGrid"],
    }


def _tensor_grid_2d(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    sx, sy = x.size - 1, y.size - 1

    numC = sx * sy
    numN = (sx + 1) * (sy + 1)
    numFX = (sx + 1) * sy
    numFY = sx * (sy + 1)
    numF = numFX + numFY

    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack([_flat(X), _flat(Y)])

    N = np.arange(numN, dtype=np.int64).reshape((sx + 1, sy + 1), order="F")

    NF1 = _flat(N[0 : sx + 1, 0:sy])
    NF2 = _flat(N[0 : sx + 1, 1 : sy + 1])
    faceNodesX = np.vstack([NF1, NF2]).ravel(order="F")

    NF1 = _flat(N[0:sx, 0 : sy + 1])
    NF2 = _flat(N[1 : sx + 1, 0 : sy + 1])
    # Reversed (NF2 first) so that computeGeometry derives normals pointing
    # in the positive-i direction -- mirrors the note in MRST's tensorGrid.m.
    faceNodesY = np.vstack([NF2, NF1]).ravel(order="F")

    faceNodes = np.concatenate([faceNodesX, faceNodesY])

    foffset = 0
    FX = foffset + np.arange(numFX, dtype=np.int64).reshape((sx + 1, sy), order="F")
    foffset += numFX
    FY = foffset + np.arange(numFY, dtype=np.int64).reshape((sx, sy + 1), order="F")

    F1 = _flat(FX[0:sx, :])
    F2 = _flat(FX[1 : sx + 1, :])
    F3 = _flat(FY[:, 0:sy])
    F4 = _flat(FY[:, 1 : sy + 1])

    cell_faces_idx = np.vstack([F1, F3, F2, F4]).ravel(order="F")
    cell_faces_tag = np.tile(np.array([1, 3, 2, 4], dtype=np.int64), numC)
    cell_faces = np.column_stack([cell_faces_idx, cell_faces_tag])

    C = np.full((sx + 2, sy + 2), -1, dtype=np.int64)
    C[1 : sx + 1, 1 : sy + 1] = np.arange(numC, dtype=np.int64).reshape((sx, sy), order="F")

    NX1 = _flat(C[0 : sx + 1, 1 : sy + 1])
    NX2 = _flat(C[1 : sx + 2, 1 : sy + 1])
    NY1 = _flat(C[1 : sx + 1, 0 : sy + 1])
    NY2 = _flat(C[1 : sx + 1, 1 : sy + 2])

    neighbors = np.concatenate(
        [np.column_stack([NX1, NX2]), np.column_stack([NY1, NY2])], axis=0
    )

    face_pos = np.arange(0, (numC + 1) * 4, 4, dtype=np.int64)
    node_pos = np.arange(0, (numF + 1) * 2, 2, dtype=np.int64)

    return {
        "cells": {
            "num": int(numC),
            "facePos": face_pos,
            "indexMap": np.arange(numC, dtype=np.int64),
            "faces": cell_faces,
        },
        "faces": {
            "num": int(numF),
            "nodePos": node_pos,
            "neighbors": neighbors,
            "tag": np.zeros(numF, dtype=np.int64),
            "nodes": faceNodes,
        },
        "nodes": {"num": int(numN), "coords": coords},
        "cartDims": np.array([sx, sy], dtype=np.int64),
        "type": ["tensorGrid"],
    }


def _tensor_grid_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, depthz: np.ndarray | None = None) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    sx, sy, sz = x.size - 1, y.size - 1, z.size - 1

    if depthz is None:
        depthz_arr = np.zeros((sx + 1, sy + 1))
    else:
        depthz_arr = np.asarray(depthz, dtype=float).reshape((sx + 1, sy + 1), order="F")

    numC = sx * sy * sz
    numN = (sx + 1) * (sy + 1) * (sz + 1)
    numFX = (sx + 1) * sy * sz
    numFY = sx * (sy + 1) * sz
    numFZ = sx * sy * (sz + 1)
    numF = numFX + numFY + numFZ

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Z = Z + depthz_arr[:, :, None]
    coords = np.column_stack([_flat(X), _flat(Y), _flat(Z)])

    N = np.arange(numN, dtype=np.int64).reshape((sx + 1, sy + 1, sz + 1), order="F")

    NF1 = _flat(N[0 : sx + 1, 0:sy, 0:sz])
    NF2 = _flat(N[0 : sx + 1, 1 : sy + 1, 0:sz])
    NF3 = _flat(N[0 : sx + 1, 1 : sy + 1, 1 : sz + 1])
    NF4 = _flat(N[0 : sx + 1, 0:sy, 1 : sz + 1])
    faceNodesX = np.vstack([NF1, NF2, NF3, NF4]).ravel(order="F")

    NF1 = _flat(N[0:sx, 0 : sy + 1, 0:sz])
    NF2 = _flat(N[0:sx, 0 : sy + 1, 1 : sz + 1])
    NF3 = _flat(N[1 : sx + 1, 0 : sy + 1, 1 : sz + 1])
    NF4 = _flat(N[1 : sx + 1, 0 : sy + 1, 0:sz])
    faceNodesY = np.vstack([NF1, NF2, NF3, NF4]).ravel(order="F")

    NF1 = _flat(N[0:sx, 0:sy, 0 : sz + 1])
    NF2 = _flat(N[1 : sx + 1, 0:sy, 0 : sz + 1])
    NF3 = _flat(N[1 : sx + 1, 1 : sy + 1, 0 : sz + 1])
    NF4 = _flat(N[0:sx, 1 : sy + 1, 0 : sz + 1])
    faceNodesZ = np.vstack([NF1, NF2, NF3, NF4]).ravel(order="F")

    faceNodes = np.concatenate([faceNodesX, faceNodesY, faceNodesZ])

    foffset = 0
    FX = foffset + np.arange(numFX, dtype=np.int64).reshape((sx + 1, sy, sz), order="F")
    foffset += numFX
    FY = foffset + np.arange(numFY, dtype=np.int64).reshape((sx, sy + 1, sz), order="F")
    foffset += numFY
    FZ = foffset + np.arange(numFZ, dtype=np.int64).reshape((sx, sy, sz + 1), order="F")

    F1 = _flat(FX[0:sx, :, :])
    F2 = _flat(FX[1 : sx + 1, :, :])
    F3 = _flat(FY[:, 0:sy, :])
    F4 = _flat(FY[:, 1 : sy + 1, :])
    F5 = _flat(FZ[:, :, 0:sz])
    F6 = _flat(FZ[:, :, 1 : sz + 1])

    cell_faces_idx = np.vstack([F1, F2, F3, F4, F5, F6]).ravel(order="F")
    cell_faces_tag = np.tile(np.array([1, 2, 3, 4, 5, 6], dtype=np.int64), numC)
    cell_faces = np.column_stack([cell_faces_idx, cell_faces_tag])

    C = np.full((sx + 2, sy + 2, sz + 2), -1, dtype=np.int64)
    C[1 : sx + 1, 1 : sy + 1, 1 : sz + 1] = np.arange(numC, dtype=np.int64).reshape(
        (sx, sy, sz), order="F"
    )

    NX1 = _flat(C[0 : sx + 1, 1 : sy + 1, 1 : sz + 1])
    NX2 = _flat(C[1 : sx + 2, 1 : sy + 1, 1 : sz + 1])
    NY1 = _flat(C[1 : sx + 1, 0 : sy + 1, 1 : sz + 1])
    NY2 = _flat(C[1 : sx + 1, 1 : sy + 2, 1 : sz + 1])
    NZ1 = _flat(C[1 : sx + 1, 1 : sy + 1, 0 : sz + 1])
    NZ2 = _flat(C[1 : sx + 1, 1 : sy + 1, 1 : sz + 2])

    neighbors = np.concatenate(
        [
            np.column_stack([NX1, NX2]),
            np.column_stack([NY1, NY2]),
            np.column_stack([NZ1, NZ2]),
        ],
        axis=0,
    )

    face_pos = np.arange(0, (numC + 1) * 6, 6, dtype=np.int64)
    node_pos = np.arange(0, (numF + 1) * 4, 4, dtype=np.int64)

    return {
        "cells": {
            "num": int(numC),
            "facePos": face_pos,
            "indexMap": np.arange(numC, dtype=np.int64),
            "faces": cell_faces,
        },
        "faces": {
            "num": int(numF),
            "nodePos": node_pos,
            "neighbors": neighbors,
            "tag": np.zeros(numF, dtype=np.int64),
            "nodes": faceNodes,
        },
        "nodes": {"num": int(numN), "coords": coords},
        "cartDims": np.array([sx, sy, sz], dtype=np.int64),
        "type": ["tensorGrid"],
    }


def tensor_grid(x, y=None, z=None, depthz=None) -> dict:
    """Port of MRST ``tensorGrid.m``.

    ``tensor_grid(x)`` builds a 1D grid, ``tensor_grid(x, y)`` a 2D grid,
    ``tensor_grid(x, y, z)`` a 3D grid.  ``x``/``y``/``z`` are vectors of
    cell-vertex coordinates (so a grid with ``n`` cells along an axis needs
    ``n + 1`` coordinates along that axis).
    """
    if y is None:
        G = _tensor_grid_1d(x)
    elif z is None:
        G = _tensor_grid_2d(x, y)
    else:
        G = _tensor_grid_3d(x, y, z, depthz=depthz)
    G["griddim"] = int(G["cartDims"].size)
    return G
