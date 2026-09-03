"""Port of MRST ``computeSqVelocTPFA.m``.

Approximates the squared cell-centred velocity magnitude from face fluxes
by squaring per-face flux/area, reconstructing a vector via the
face-centroid-minus-cell-centroid direction, and taking a distance-weighted
average over half-faces. See ``computeVelocTPFA`` for the connectivity
requirements this relies on.
"""

import numpy as _np

try:
    import scipy.sparse as _sp
except Exception:  # pragma: no cover
    _sp = None


def computeSqVelocTPFA(G, intInx):
    """Returns ``sqVeloc(v)``: cell-valued squared velocity magnitude for
    face fluxes ``v`` ordered over internal faces (matching ``intInx``)."""
    cells = G['cells']
    faces = G['faces']
    face_pos = _np.asarray(cells['facePos'], dtype=_np.int64).ravel()
    cell_faces = _np.asarray(cells['faces'], dtype=_np.int64)
    cell_faces_col = cell_faces[:, 0] if cell_faces.ndim == 2 else cell_faces
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

    sumHalffaces = _sp.csr_matrix(
        (_np.ones(nhf), (cellNo, _np.arange(nhf))), shape=(nc, nhf))
    face_areas = _np.asarray(faces['areas'], dtype=float).ravel()
    wSumHalffaces = sumHalffaces @ _sp.diags(1.0 / (face_areas[cell_faces_col] ** 2))

    face_centroids = _np.asarray(faces['centroids'], dtype=float)
    cell_centroids = _np.asarray(cells['centroids'], dtype=float)
    C = _np.abs(face_centroids[cell_faces_col, :] - cell_centroids[cellNo, :])
    Csum = sumHalffaces @ C
    C = C / Csum[cellNo, :]

    D = [_sp.diags(C[:, i]) for i in range(dim)]

    def sqVeloc(v):
        hf_v = fromIntfacesToHalffaces @ v
        hf_sq_v = hf_v ** 2
        out = 0.0
        for i in range(dim):
            out = out + wSumHalffaces @ (D[i] @ hf_sq_v)
        return out

    return sqVeloc
