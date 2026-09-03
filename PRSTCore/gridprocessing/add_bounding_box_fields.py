"""Port of MRST ``addBoundingBoxFields.m``
(mrst-2026a/core/utils/gridtools).

Adds the axis-aligned extent of each face (and optionally each cell) to
the grid, as ``G['faces']['bbox']`` / ``G['cells']['bbox']``. The extent
is the span of the entity's node coordinates along each axis, so it is a
cheap proxy for size used to reject candidates before doing exact
geometry.
"""

import numpy as _np


def add_bounding_box_fields(G, cells=False, faces=True):
    """Return ``G`` with the requested bbox fields added."""
    coords = _np.asarray(G['nodes']['coords'], dtype=float)
    gdim = coords.shape[1]
    griddim = int(G.get('griddim', gdim))

    if cells:
        cno = _cell_nodes(G)
        G['cells']['bbox'] = _spans(cno[:, 0], coords[cno[:, 2], :],
                                    int(G['cells']['num']), gdim, griddim)

    if faces:
        npos = _np.asarray(G['faces']['nodePos'], dtype=int).ravel()
        nodes = _np.asarray(G['faces']['nodes'], dtype=int).ravel()
        fno = _np.repeat(_np.arange(int(G['faces']['num'])), _np.diff(npos))
        G['faces']['bbox'] = _spans(fno, coords[nodes, :],
                                    int(G['faces']['num']), gdim, griddim)
    return G


def _spans(owner, points, n, gdim, griddim):
    """Per-owner max minus min, one column per axis."""
    bbox = _np.full((n, griddim), _np.nan)
    for d in range(gdim):
        hi = _np.full(n, -_np.inf)
        lo = _np.full(n, _np.inf)
        _np.maximum.at(hi, owner, points[:, d])
        _np.minimum.at(lo, owner, points[:, d])
        with _np.errstate(invalid='ignore'):
            span = hi - lo
        # An owner with no nodes has -inf - inf; leave it NaN, as MATLAB's
        # preallocated NaN does.
        span[~_np.isfinite(span)] = _np.nan
        bbox[:, d] = span
    return bbox


def _cell_nodes(G):
    """Port of ``cellNodes``: (cell, face, node) triples.

    MRST returns them sorted by cell; only the cell and node columns are
    read here, so the face column is filled in but not relied upon.
    """
    facePos = _np.asarray(G['cells']['facePos'], dtype=int).ravel()
    cfaces = _np.asarray(G['cells']['faces'], dtype=int)
    if cfaces.ndim > 1:
        cfaces = cfaces[:, 0]
    nodePos = _np.asarray(G['faces']['nodePos'], dtype=int).ravel()
    fnodes = _np.asarray(G['faces']['nodes'], dtype=int).ravel()

    cells = _np.repeat(_np.arange(int(G['cells']['num'])), _np.diff(facePos))
    counts = _np.diff(nodePos)[cfaces]
    out_cells = _np.repeat(cells, counts)
    out_faces = _np.repeat(cfaces, counts)
    out_nodes = _np.concatenate([fnodes[nodePos[f]:nodePos[f + 1]]
                                 for f in cfaces]) if cfaces.size \
        else _np.zeros(0, dtype=int)
    return _np.column_stack([out_cells, out_faces, out_nodes])
