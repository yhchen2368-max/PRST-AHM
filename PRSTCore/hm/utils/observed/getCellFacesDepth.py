"""Port of MRST ``getCellFacesDepth.m`` (mrst-2026a/hm/utils/observed).

Top and bottom depth of each cell, taken as the shallowest and deepest of
its face centroids -- the depth interval a logged measurement is averaged
over.
"""

import numpy as _np


def getCellFacesDepth(G, c):
    """Return ``(top, bottom)``, one entry per cell in ``c``."""
    cells = _np.atleast_1d(_np.asarray(c, dtype=int)).ravel()
    face_pos = _np.asarray(G['cells']['facePos'], dtype=int).ravel()
    cell_faces = _np.asarray(G['cells']['faces'], dtype=int)
    if cell_faces.ndim == 1:
        cell_faces = cell_faces.reshape(-1, 1)
    centroids = _np.asarray(G['faces']['centroids'], dtype=float)

    top = _np.zeros(cells.size)
    bottom = _np.zeros(cells.size)
    for i, cell in enumerate(cells):
        faces = cell_faces[face_pos[cell]:face_pos[cell + 1], 0]
        z = centroids[faces, 2]
        top[i] = z.min()
        bottom[i] = z.max()
    return top, bottom
