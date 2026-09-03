"""Port of MRST ``computeVelocTPFA.m``.

Builds the per-axis operator that reconstructs an approximate cell-centred
velocity vector from face fluxes: ``v_c = 1/V * sum_f (x_f - x_c) u_f``.
Exact for linear pressure fields and first-order accurate on K-orthogonal
(e.g. TPFA-consistent) grids; large errors are expected in well cells.

Requires generic half-face connectivity (``G['cells']['facePos']``,
``G['cells']['faces']``) as MRST's unstructured grid provides. PRSTCore's
Cartesian black-oil grid (``deckformat/grid/init_eclipse_grid.py``) does not
currently build this; callers on such a grid should use a simpler
axis-aligned reconstruction from the internal-connection list instead (see
``ad_eor.properties.CapillaryNumber`` for the caller-supplied ``sqVeloc``
convention this implies).
"""

import numpy as _np

try:
    import scipy.sparse as _sp
except Exception:  # pragma: no cover
    _sp = None


def computeVelocTPFA(G, intInx):
    """Returns ``veloc``: a list of ``dim`` callables ``veloc[i](v)`` that
    map face fluxes ``v`` (ordered over internal faces, matching ``intInx``)
    to the cell-valued ``i``-th velocity component."""
    cells = G['cells']
    faces = G['faces']
    face_pos = _np.asarray(cells['facePos'], dtype=_np.int64).ravel()
    cell_faces = _np.asarray(cells['faces'], dtype=_np.int64)
    if cell_faces.ndim == 2:
        cell_faces_col = cell_faces[:, 0]
    else:
        cell_faces_col = cell_faces
    nc = int(cells['num'])
    cellNo = _np.repeat(_np.arange(nc), _np.diff(face_pos))
    dim = _np.asarray(G['nodes']['coords'], dtype=float).shape[1]

    intInx = _np.asarray(intInx, dtype=bool).ravel()
    nhf = cell_faces_col.size
    nf = int(faces['num'])
    nif = int(_np.count_nonzero(intInx))

    fromIntfacesToFaces = _sp.csr_matrix(
        (_np.ones(nif), (_np.flatnonzero(intInx), _np.arange(nif))), shape=(nf, nif))
    neighbors = _np.asarray(faces['neighbors'], dtype=_np.int64)
    sgn = 2.0 * (cellNo == neighbors[cell_faces_col, 0]).astype(float) - 1.0
    fromFacesToHalffaces = _sp.csr_matrix(
        (sgn, (_np.arange(nhf), cell_faces_col)), shape=(nhf, nf))
    fromIntfacesToHalffaces = fromFacesToHalffaces @ fromIntfacesToFaces

    vol = _np.asarray(cells['volumes'], dtype=float).ravel()
    face_centroids = _np.asarray(faces['centroids'], dtype=float)
    cell_centroids = _np.asarray(cells['centroids'], dtype=float)
    C = face_centroids[cell_faces_col, :] - cell_centroids[cellNo, :]

    sumHalffaces = _sp.csr_matrix(
        (_np.ones(nhf), (cellNo, _np.arange(nhf))), shape=(nc, nhf))

    veloc = []
    for i in range(dim):
        Ci = C[:, i]

        def _v(v, Ci=Ci):
            return (1.0 / vol) * (sumHalffaces @ (Ci * (fromIntfacesToHalffaces @ v)))
        veloc.append(_v)
    return veloc
