"""Port of MRST ``getDZ``: thickness (DZ) of a grid cell from its Z- and Z+
face centres (requires the face direction indicator in ``cells.faces[:, 1]``)."""

import numpy as np


def getDZ(G, c):
    """Compute DZ of (0-based) cell ``c`` in grid ``G``."""
    fPos = np.arange(G['cells']['facePos'][c], G['cells']['facePos'][c + 1])
    faces = G['cells']['faces'][fPos, 0]
    faceDir = G['cells']['faces'][fPos, 1]
    fCenter6 = G['faces']['centroids'][faces[faceDir == 6]]
    fCenter5 = G['faces']['centroids'][faces[faceDir == 5]]
    dz = np.linalg.norm(fCenter6 - fCenter5, 2)
    return float(dz)
