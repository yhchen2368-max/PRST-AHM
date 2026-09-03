"""Python port of MRST's ``processGRDECL.m`` (mrst-2026a/core/gridprocessing).

Converts a corner-point (``COORD``/``ZCORN``/``ACTNUM``) grid specification
into a topology-only grid dict in the same ``cells.facePos``/``faces.nodePos``/
``nodes.coords`` format that :func:`PRSTCore.gridprocessing.cart_grid.cart_grid`
and :func:`PRSTCore.gridprocessing.tensor_grid.tensor_grid` produce, so that
:func:`PRSTCore.gridprocessing.compute_geometry.compute_geometry` -- already
verified bit-identical to MRST -- can be applied uniformly::

    G = compute_geometry(process_grdecl(grdecl))

The hard geometric work (splitting faulted, non-matching pillar interfaces
into their actual polygonal connections; bridging degenerate/pinched-out
cells) is delegated to ``PRSTCore.deckformat.grid.init_eclipse_grid``'s
``_build_corner_point_nodes``/``_cp_mex_topology`` -- a faithful, previously
validated (against Norne/SPE9) port of MRST's default ``processgrid_mex``
topology builder -- rather than reimplemented here. This module only adapts
that (nodes, face-node-lists, neighbors, tags, index_map) output into the
CSR-based grid_structure convention used elsewhere in
:mod:`PRSTCore.gridprocessing`.

Known gap vs. MRST's ``processGRDECL``: explicit ``NNC`` keyword connections
(deck-specified cell pairs independent of geometry) are not added; the
*geometric* fault/pinch connectivity (non-matching pillar interfaces,
collapsed zero-thickness layers) is handled via the reused topology builder.
"""

from __future__ import annotations

import numpy as np

from PRSTCore.deckformat.grid.init_eclipse_grid import _cp_mex_topology


def process_grdecl(grdecl: dict) -> dict:
    """Port of MRST ``processGRDECL.m``.

    Parameters
    ----------
    grdecl : dict
        Must contain ``'cartDims'`` (length-3), ``'COORD'`` (``6*(nx+1)*(ny+1)``
        values), ``'ZCORN'`` (``8*nx*ny*nz`` values, Fortran/ECLIPSE order),
        and optionally ``'ACTNUM'`` (``nx*ny*nz`` 0/1 flags; all-active if
        omitted).

    Returns
    -------
    dict
        Topology-only grid (no ``cells.volumes``/``faces.areas``/etc. --
        call :func:`compute_geometry` on the result for those).
    """
    dims = grdecl.get('cartDims')
    if dims is None or len(dims) < 3:
        raise ValueError("process_grdecl requires grdecl['cartDims']")
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    nfull = nx * ny * nz

    coord = np.asarray(grdecl['COORD'], dtype=float).ravel()
    zcorn = np.asarray(grdecl['ZCORN'], dtype=float).ravel(order='F')
    expected_coord = 6 * (nx + 1) * (ny + 1)
    expected_zcorn = 8 * nfull
    if coord.size != expected_coord or zcorn.size != expected_zcorn:
        raise ValueError(
            'Invalid corner-point dimensions: expected %d COORD and %d ZCORN values, got %d and %d'
            % (expected_coord, expected_zcorn, coord.size, zcorn.size)
        )

    actnum = grdecl.get('ACTNUM')
    actnum = np.ones(nfull, dtype=int) if actnum is None else np.asarray(actnum, dtype=int).ravel()
    if actnum.size != nfull:
        raise ValueError('ACTNUM length does not match cartDims')
    active = actnum.astype(bool)

    nodes, faces_list, neighbors, tags, index_map = _cp_mex_topology(
        coord, zcorn, active.reshape((nx, ny, nz), order='F'), nx, ny, nz
    )

    nc = int(index_map.size)
    nf = len(faces_list)
    neighbors = np.asarray(neighbors, dtype=np.int64).reshape(-1, 2)
    tags = np.asarray(tags, dtype=np.int64).ravel()

    counts = np.array([len(f) for f in faces_list], dtype=np.int64)
    node_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    flat_nodes = (
        np.concatenate([np.asarray(f, dtype=np.int64) for f in faces_list])
        if faces_list else np.zeros(0, dtype=np.int64)
    )

    # Half-faces, one row per (face, owning cell): MRST direction-tag
    # convention is 1-6 (W,E,S,N,T,B); the axis (1/2/3 here) maps to it as
    # 2*axis for the neighbors[:,0] ("from") side and 2*axis-1 for the
    # neighbors[:,1] ("to") side (matching cart_grid/tensor_grid's own
    # convention, verified against MRST there). Geometry itself does not
    # depend on this tag; only compute_trans's net-to-gross axis lookup
    # (ceil(tag/2)) does, and that's satisfied by either half of the pair.
    a, b = neighbors[:, 0], neighbors[:, 1]
    has_a, has_b = a >= 0, b >= 0
    hf_face = np.concatenate([np.nonzero(has_a)[0], np.nonzero(has_b)[0]])
    hf_cell = np.concatenate([a[has_a], b[has_b]])
    hf_tag = np.concatenate([2 * tags[has_a], 2 * tags[has_b] - 1])

    order = np.argsort(hf_cell, kind='stable')
    hf_face, hf_cell, hf_tag = hf_face[order], hf_cell[order], hf_tag[order]

    counts_per_cell = np.bincount(hf_cell, minlength=nc) if hf_cell.size else np.zeros(nc, dtype=np.int64)
    face_pos = np.concatenate([[0], np.cumsum(counts_per_cell)]).astype(np.int64)
    cell_faces = np.column_stack([hf_face, hf_tag]).astype(np.int64) if hf_face.size else np.zeros((0, 2), dtype=np.int64)

    return {
        'cells': {
            'num': nc,
            'facePos': face_pos,
            'indexMap': np.asarray(index_map, dtype=np.int64),
            'faces': cell_faces,
        },
        'faces': {
            'num': nf,
            'nodePos': node_pos,
            'neighbors': neighbors,
            'tag': tags,
            'nodes': flat_nodes,
        },
        'nodes': {'num': int(nodes.shape[0]), 'coords': np.asarray(nodes, dtype=float)},
        'cartDims': np.array([nx, ny, nz], dtype=np.int64),
        'griddim': 3,
        'type': ['processGRDECL'],
    }
