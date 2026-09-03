"""Form a coarse grid from a fine-grid cell partition.

The topology follows MRST's ``generateCoarseGrid`` data model:

* cell and face identifiers stored in topology arrays are 1-based;
* boundary/exterior neighbor is stored as 0;
* ``faces.fconn`` stores the 0-based fine-face rows that constitute each
  coarse face, so NumPy arrays can be indexed directly.
"""

from collections import defaultdict

import numpy as np


def generate_coarse_grid(G, pv, pf=None):
    """Generate coarse grid topology from a fine grid and partition.

    Parameters
    ----------
    G : dict
        Fine grid with ``cells`` and ``faces`` topology.
    pv : ndarray
        1-based fine-cell to coarse-block partition.
    pf : ndarray, optional
        Optional face partition. Faces with different positive ``pf`` values
        are kept as separate coarse faces even when they connect the same
        coarse block pair.
    """
    p = np.asarray(pv, dtype=int).ravel()
    nc = int(G["cells"]["num"])
    nf = int(G["faces"]["num"])
    if p.size != nc:
        raise ValueError("partition must have one entry per fine-grid cell")
    if p.min(initial=1) < 1:
        raise ValueError("partition must contain positive coarse block numbers")

    nbrs = np.asarray(G["faces"]["neighbors"], dtype=int)
    if nbrs.shape[0] != nf or nbrs.shape[1] < 2:
        raise ValueError("G.faces.neighbors must have shape (G.faces.num, 2)")

    indicator = _indicator(G, pf)
    groups = _coarse_face_groups(nbrs, p, indicator)

    coarse_neighbors = []
    conn_pos = [0]
    fconn = []
    face_tags = []
    for group in groups:
        coarse_neighbors.append(group["neighbors"])
        fconn.extend(group["faces"])
        conn_pos.append(len(fconn))
        face_tags.append(group["tag"])

    coarse_neighbors = np.asarray(coarse_neighbors, dtype=int).reshape((-1, 2))
    ncf = coarse_neighbors.shape[0]
    conn_pos = np.asarray(conn_pos, dtype=int)
    fconn = np.asarray(fconn, dtype=int)

    cell_faces, face_pos = _transpose_connections(coarse_neighbors, face_tags, int(p.max()))

    CG = {
        "cells": {
            "num": int(p.max()),
            "facePos": np.asarray(face_pos, dtype=int),
            "faces": np.asarray(cell_faces, dtype=int).reshape((-1, 2)),
        },
        "faces": {
            "num": int(ncf),
            "neighbors": coarse_neighbors,
            "connPos": conn_pos,
            "fconn": fconn,
        },
        "partition": p.copy(),
        "parent": G,
        "griddim": G.get("griddim", 3),
        "type": ["generateCoarseGrid"],
    }

    if "nnc" in G and isinstance(G["nnc"], dict) and "cells" in G["nnc"]:
        cells = np.asarray(G["nnc"].get("cells", []), dtype=int)
        if cells.size:
            p_ext = np.concatenate([[0], p])
            cn = p_ext[cells]
            cn = cn[cn[:, 0] != cn[:, 1]]
            if cn.size:
                # Stable unique rows.
                seen = set()
                rows = []
                for row in cn.tolist():
                    key = tuple(row)
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
                CG["nnc"] = {"cells": np.asarray(rows, dtype=int)}

    return CG


def _indicator(G, pf):
    """Return per-face grouping indicators similar to MRST's indicator()."""
    nf = int(G["faces"]["num"])
    cols = []

    # Inherit external face tags from G.cells.faces second column, if present.
    inherited = np.zeros(nf, dtype=int)
    cf = np.asarray(G.get("cells", {}).get("faces", np.empty((0, 0))), dtype=int)
    if cf.ndim == 2 and cf.shape[1] > 1 and cf.size:
        nbrs = np.asarray(G["faces"]["neighbors"], dtype=int)
        ext = np.any(nbrs == 0, axis=1)
        for row in cf:
            face_id = int(row[0])
            if face_id <= 0:
                continue
            face_ix = face_id - 1
            if 0 <= face_ix < nf and ext[face_ix]:
                inherited[face_ix] = int(row[1])
    cols.append(inherited)

    if pf is not None:
        pf_arr = np.asarray(pf, dtype=int).ravel()
        if pf_arr.size != nf:
            raise ValueError("face partition must have one entry per fine-grid face")
        cols.append(pf_arr)

    return np.column_stack(cols)


def _coarse_face_groups(nbrs, p, indicator):
    p_ext = np.concatenate([[0], p])
    groups = {}
    order = []

    for f, (c1, c2) in enumerate(nbrs[:, :2]):
        b1 = int(p_ext[c1]) if c1 > 0 else 0
        b2 = int(p_ext[c2]) if c2 > 0 else 0
        if b1 == b2 and b1 != 0:
            continue

        pair = tuple(sorted((b1, b2)))
        # Domain boundary: keep exterior in second column as MRST-style
        # coarse grid convention where possible.
        neighbors = (pair[1], 0) if pair[0] == 0 else pair
        tag = tuple(int(v) for v in indicator[f])
        key = pair + tag
        if key not in groups:
            groups[key] = {"neighbors": neighbors, "faces": [], "tag": tag[0] if tag else 0}
            order.append(key)
        groups[key]["faces"].append(int(f))

    return [groups[k] for k in order]


def _transpose_connections(coarse_neighbors, face_tags, ncoarse):
    by_cell = [[] for _ in range(ncoarse)]
    for face_ix, (c1, c2) in enumerate(coarse_neighbors, start=1):
        tag = int(face_tags[face_ix - 1]) if face_tags else 0
        if c1 > 0:
            by_cell[c1 - 1].append([face_ix, tag])
        if c2 > 0 and c2 != c1:
            by_cell[c2 - 1].append([face_ix, tag])

    faces = []
    face_pos = [0]
    for rows in by_cell:
        faces.extend(rows)
        face_pos.append(len(faces))
    return faces, face_pos
