"""Python port of MRST's ``computeGeometry.m`` (mrst-2026a/core/gridprocessing).

Adds ``cells.volumes``/``cells.centroids`` and ``faces.areas``/
``faces.normals``/``faces.centroids`` to a topology-only grid produced by
:func:`PRSTCore.gridprocessing.tensor_grid.tensor_grid` or
:func:`PRSTCore.gridprocessing.cart_grid.cart_grid`.

Follows MRST's algorithm exactly (not a shortcut cuboid formula), so it is
correct for non-uniform and node-perturbed (non-orthogonal) grids as well:

  - Each face is triangulated using an auxiliary centre point (the
    unweighted average of its node coordinates); the area-weighted sum of
    the sub-triangle normals gives the face normal, and the area-weighted
    average of sub-triangle centroids gives the face centroid.
  - Each cell is decomposed into sub-tetrahedra, one per face sub-triangle,
    using an auxiliary interior point (the unweighted average of the cell's
    face centroids); the signed sum of sub-tetrahedron volumes gives the
    cell volume, and the volume-weighted average of sub-tetrahedron
    centroids gives the cell centroid.

Only ``G.griddim`` in ``{1, 2, 3}`` with node coordinates embedded in the
same number of physical dimensions are supported (i.e. MRST's ``geom_1d``,
``geom_2d2`` and ``geom_3d`` paths). The 2D-surface-embedded-in-3D and
hinge-node/CpGeometry paths are not implemented.
"""

from __future__ import annotations

import numpy as np


def _weighted_group_average(group_id: np.ndarray, values: np.ndarray, weights: np.ndarray, ngroups: int):
    """Grouped weighted average; mirrors MRST's ``averageCoordinates`` helper.

    Returns ``(average, weight_sum)`` where ``average[g]`` is the
    weight-weighted mean of ``values`` rows belonging to group ``g``, and
    ``weight_sum[g]`` is the (possibly signed) sum of weights in group ``g``.
    """
    weights = np.asarray(weights, dtype=float)
    wsum = np.bincount(group_id, weights=weights, minlength=ngroups)
    values = np.atleast_2d(values.T).T if values.ndim == 1 else values
    out = np.zeros((ngroups, values.shape[1]))
    for d in range(values.shape[1]):
        out[:, d] = np.bincount(group_id, weights=weights * values[:, d], minlength=ngroups)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = out / wsum[:, None]
    return avg, wsum


def _group_sum(group_id: np.ndarray, values: np.ndarray, ngroups: int) -> np.ndarray:
    if values.ndim == 1:
        return np.bincount(group_id, weights=values, minlength=ngroups)
    out = np.zeros((ngroups, values.shape[1]))
    for d in range(values.shape[1]):
        out[:, d] = np.bincount(group_id, weights=values[:, d], minlength=ngroups)
    return out


def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """``[0..counts[0]-1, 0..counts[1]-1, ...]`` concatenated, vectorized."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    idx = np.arange(total)
    group_starts_cum = np.repeat(np.cumsum(counts) - counts, counts)
    return idx - group_starts_cum


def _circular_next(counts: np.ndarray) -> np.ndarray:
    """Per-group circular "next row" pointer into a flat, group-contiguous array."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    nxt = np.arange(1, total + 1)
    group_ends = np.cumsum(counts) - 1
    group_starts = group_ends - counts + 1
    nxt[group_ends] = group_starts
    return nxt


def _face_geom3d_all(G: dict):
    node_pos = np.asarray(G["faces"]["nodePos"])
    face_nodes = np.asarray(G["faces"]["nodes"])
    coords = np.asarray(G["nodes"]["coords"], dtype=float)
    nf = G["faces"]["num"]

    counts = np.diff(node_pos)
    face_no = np.repeat(np.arange(nf), counts)

    p_centers, _ = _weighted_group_average(face_no, coords[face_nodes], np.ones(face_nodes.size), nf)
    p_centers_row = p_centers[face_no]

    nxt = _circular_next(counts)
    a = coords[face_nodes]
    b = coords[face_nodes[nxt]]

    sub_normals = np.cross(b - a, p_centers_row - a) / 2.0
    sub_areas = np.linalg.norm(sub_normals, axis=1)
    sub_centroids = (a + b + p_centers_row) / 3.0

    fN = _group_sum(face_no, sub_normals, nf)
    fN_row = fN[face_no]
    sub_signs = np.sign(np.sum(sub_normals * fN_row, axis=1))

    fC, fA = _weighted_group_average(face_no, sub_centroids, sub_areas, nf)

    degenerate = ~(fA > 0)
    if np.any(degenerate):
        fC[degenerate] = p_centers[degenerate]

    return fA, fN, fC, sub_centroids, sub_normals, sub_signs


def _geom_3d(G: dict):
    nc = G["cells"]["num"]
    fA, fN, fC, subC, subN, subSigns = _face_geom3d_all(G)

    face_pos = np.asarray(G["cells"]["facePos"])
    cell_faces = np.asarray(G["cells"]["faces"])
    counts_cf = np.diff(face_pos)
    hf_cell = np.repeat(np.arange(nc), counts_cf)
    hf_face = cell_faces[:, 0]

    cCenter, _ = _weighted_group_average(hf_cell, fC[hf_face], np.ones(hf_face.size), nc)

    node_pos = np.asarray(G["faces"]["nodePos"])
    tri_counts = node_pos[hf_face + 1] - node_pos[hf_face]
    hf_row = np.arange(hf_face.size)
    tri_hf_row = np.repeat(hf_row, tri_counts)
    tri_cell = hf_cell[tri_hf_row]
    tri_face = hf_face[tri_hf_row]

    starts = node_pos[hf_face]
    tri_sub_idx = np.repeat(starts, tri_counts) + _ragged_arange(tri_counts)

    relSubC = subC[tri_sub_idx] - cCenter[tri_cell]
    neighbors0 = np.asarray(G["faces"]["neighbors"])[tri_face, 0]
    cfsign = 2.0 * (neighbors0 == tri_cell).astype(float) - 1.0
    outNormals = subN[tri_sub_idx] * (subSigns[tri_sub_idx] * cfsign)[:, None]
    tVolumes = (1.0 / 3.0) * np.sum(relSubC * outNormals, axis=1)
    tCentroids = (3.0 / 4.0) * relSubC

    relCentroid, volume = _weighted_group_average(tri_cell, tCentroids, tVolumes, nc)
    centroid = relCentroid + cCenter
    return fA, fN, fC, volume, centroid


def _geom_2d(G: dict):
    nc = G["cells"]["num"]
    face_nodes = np.asarray(G["faces"]["nodes"]).reshape(-1, 2)
    coords = np.asarray(G["nodes"]["coords"], dtype=float)

    n1 = coords[face_nodes[:, 0]]
    n2 = coords[face_nodes[:, 1]]
    edge_vec = n2 - n1
    faceAreas = np.sqrt(np.sum(edge_vec**2, axis=1))
    faceCentroids = (n1 + n2) / 2.0
    faceNormals = np.column_stack([edge_vec[:, 1], -edge_vec[:, 0]])

    face_pos = np.asarray(G["cells"]["facePos"])
    cell_faces = np.asarray(G["cells"]["faces"])
    counts_cf = np.diff(face_pos)
    hf_cell = np.repeat(np.arange(nc), counts_cf)
    hf_face = cell_faces[:, 0]

    cCenter, _ = _weighted_group_average(hf_cell, faceCentroids[hf_face], np.ones(hf_face.size), nc)

    neighbors = np.asarray(G["faces"]["neighbors"])
    flip = neighbors[hf_face, 1] == hf_cell
    a_node = np.where(flip, face_nodes[hf_face, 1], face_nodes[hf_face, 0])
    b_node = np.where(flip, face_nodes[hf_face, 0], face_nodes[hf_face, 1])

    cc = cCenter[hf_cell]
    a = coords[a_node] - cc
    b = coords[b_node] - cc
    subArea = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2.0
    subCentroid = (cc + 2.0 * faceCentroids[hf_face]) / 3.0

    cellCentroids, cellVolumes = _weighted_group_average(hf_cell, subCentroid, subArea, nc)
    return faceAreas, faceNormals, faceCentroids, cellVolumes, cellCentroids


def _geom_1d(G: dict):
    coords = np.asarray(G["nodes"]["coords"], dtype=float).reshape(-1)
    nf = G["faces"]["num"]
    nc = G["cells"]["num"]

    faceAreas = np.ones(nf)
    faceNormals = np.ones((nf, 1))
    faceCentroids = coords.reshape(-1, 1)
    cellVolumes = np.diff(coords)

    face_pos = np.asarray(G["cells"]["facePos"])
    cell_faces = np.asarray(G["cells"]["faces"])
    counts_cf = np.diff(face_pos)
    cellNo = np.repeat(np.arange(nc), counts_cf)

    node_of_face = coords[cell_faces[:, 0]]
    cellCentroids = (np.bincount(cellNo, weights=node_of_face, minlength=nc) / 2.0).reshape(-1, 1)
    return faceAreas, faceNormals, faceCentroids, cellVolumes, cellCentroids


def compute_geometry(G: dict) -> dict:
    """Port of MRST ``computeGeometry.m``. Returns a new grid dict with
    geometry fields added; ``G`` itself is not mutated."""
    griddim = int(G["griddim"])
    if griddim == 1:
        fA, fN, fC, cV, cC = _geom_1d(G)
    elif griddim == 2:
        fA, fN, fC, cV, cC = _geom_2d(G)
    elif griddim == 3:
        fA, fN, fC, cV, cC = _geom_3d(G)
    else:
        raise ValueError(f"Unable to compute geometry for {griddim} dimensions")

    out = dict(G)
    out["faces"] = dict(G["faces"])
    out["cells"] = dict(G["cells"])
    out["faces"]["areas"] = fA
    out["faces"]["normals"] = fN
    out["faces"]["centroids"] = fC
    out["cells"]["volumes"] = cV
    out["cells"]["centroids"] = cC
    out["type"] = list(G.get("type", [])) + ["computeGeometry"]
    return out
