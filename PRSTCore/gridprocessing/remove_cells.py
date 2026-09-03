"""Python port of MRST's ``removeCells.m`` (mrst-2026a/core/gridprocessing).

Removes cells from a grid and renumbers the surviving cells/faces/nodes.
Faces between a removed cell and a kept cell become new boundary faces of
the kept cell; faces entirely between two removed cells are dropped, along
with any node no longer referenced by a surviving face.

Works for any grid produced by :mod:`PRSTCore.gridprocessing` (regular
``cart_grid``/``tensor_grid`` output, corner-point grids from
``process_grdecl``, or general polyhedral grids), before or after
:func:`PRSTCore.gridprocessing.compute_geometry.compute_geometry` -- if
geometry fields are present they are index-dropped along with the topology
(kept cells/faces do not need their geometry recomputed).
"""

from __future__ import annotations

import numpy as np


def remove_cells(G: dict, cells) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Port of MRST ``removeCells.m``.

    Parameters
    ----------
    G : dict
        Grid (topology-only or with geometry) with 0-based indices and -1
        marking "no neighbor"/boundary, per this package's convention.
    cells : array-like of int, or boolean mask of length ``G['cells']['num']``
        Cells to remove.

    Returns
    -------
    (H, cellmap, facemap, nodemap)
        ``H`` is the reduced grid. ``cellmap[i]``/``facemap[i]``/``nodemap[i]``
        give the *original* (``G``) index of new entity ``i`` in ``H`` --
        e.g. ``cellmap[3] == 7`` means ``H``'s cell 3 was ``G``'s cell 7.
    """
    nc = G["cells"]["num"]
    nf = G["faces"]["num"]
    nn = G["nodes"]["num"]

    cells = np.asarray(cells)
    if cells.size == 0:
        return G, np.arange(nc), np.arange(nf), np.arange(nn)

    if cells.dtype == bool:
        if cells.size != nc:
            raise ValueError("Boolean cell mask must have length G['cells']['num']")
        remove_mask = cells
    else:
        remove_mask = np.zeros(nc, dtype=bool)
        remove_mask[cells.astype(np.int64)] = True
    keep_mask = ~remove_mask

    cellmap = np.full(nc, -1, dtype=np.int64)
    cellmap[keep_mask] = np.arange(int(np.count_nonzero(keep_mask)))

    face_pos = G["cells"]["facePos"]
    cell_faces = G["cells"]["faces"]
    counts = np.diff(face_pos)
    hf_cell = np.repeat(np.arange(nc), counts)
    hf_keep = keep_mask[hf_cell]
    kept_half_faces = cell_faces[hf_keep].copy()
    new_face_pos = np.concatenate([[0], np.cumsum(counts[keep_mask])]).astype(np.int64)

    neighbors = np.asarray(G["faces"]["neighbors"], dtype=np.int64)
    a, b = neighbors[:, 0], neighbors[:, 1]
    new_a = np.where(a >= 0, cellmap[np.clip(a, 0, nc - 1)], -1)
    new_b = np.where(b >= 0, cellmap[np.clip(b, 0, nc - 1)], -1)
    face_removed = (new_a < 0) & (new_b < 0)

    facemap = np.full(nf, -1, dtype=np.int64)
    facemap[~face_removed] = np.arange(int(np.count_nonzero(~face_removed)))
    new_neighbors = np.column_stack([new_a, new_b])[~face_removed]

    kept_half_faces[:, 0] = facemap[kept_half_faces[:, 0]]
    if np.any(kept_half_faces[:, 0] < 0):
        raise ValueError("In remove_cells: too many faces removed")

    has_face_nodes = "nodes" in G["faces"]
    if has_face_nodes:
        node_pos = G["faces"]["nodePos"]
        face_nodes = G["faces"]["nodes"]
        fcounts = np.diff(node_pos)
        fn_face = np.repeat(np.arange(nf), fcounts)
        kept_face_nodes_pre = face_nodes[(~face_removed)[fn_face]]
        new_node_pos = np.concatenate([[0], np.cumsum(fcounts[~face_removed])]).astype(np.int64)

        referenced = np.zeros(nn, dtype=bool)
        referenced[kept_face_nodes_pre] = True
        nodemap = np.full(nn, -1, dtype=np.int64)
        nodemap[referenced] = np.arange(int(np.count_nonzero(referenced)))

        new_face_nodes = nodemap[kept_face_nodes_pre]
        if np.any(new_face_nodes < 0):
            raise ValueError("In remove_cells: too many nodes removed")
        new_coords = np.asarray(G["nodes"]["coords"])[referenced]
    else:
        referenced = np.zeros(nn, dtype=bool)
        nodemap = np.full(nn, -1, dtype=np.int64)
        new_face_nodes = None
        new_node_pos = None
        new_coords = np.asarray(G["nodes"]["coords"])

    H = dict(G)
    H["cells"] = dict(G["cells"])
    H["faces"] = dict(G["faces"])
    H["nodes"] = dict(G["nodes"])

    H["cells"]["num"] = int(np.count_nonzero(keep_mask))
    H["cells"]["facePos"] = new_face_pos
    H["cells"]["faces"] = kept_half_faces
    if "indexMap" in G["cells"]:
        H["cells"]["indexMap"] = np.asarray(G["cells"]["indexMap"])[keep_mask]

    H["faces"]["num"] = int(np.count_nonzero(~face_removed))
    H["faces"]["neighbors"] = new_neighbors
    if "tag" in G["faces"]:
        H["faces"]["tag"] = np.asarray(G["faces"]["tag"])[~face_removed]
    if has_face_nodes:
        H["faces"]["nodePos"] = new_node_pos
        H["faces"]["nodes"] = new_face_nodes

    H["nodes"]["num"] = int(np.count_nonzero(referenced)) if has_face_nodes else G["nodes"]["num"]
    H["nodes"]["coords"] = new_coords

    if "volumes" in G["cells"]:
        H["cells"]["volumes"] = np.asarray(G["cells"]["volumes"])[keep_mask]
        H["cells"]["centroids"] = np.asarray(G["cells"]["centroids"])[keep_mask]
    if "areas" in G["faces"]:
        H["faces"]["areas"] = np.asarray(G["faces"]["areas"])[~face_removed]
        H["faces"]["centroids"] = np.asarray(G["faces"]["centroids"])[~face_removed]
        H["faces"]["normals"] = np.asarray(G["faces"]["normals"])[~face_removed]

    H["type"] = list(G.get("type", [])) + ["removeCells"]

    return H, np.flatnonzero(keep_mask), np.flatnonzero(~face_removed), np.flatnonzero(referenced)
