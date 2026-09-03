"""Port of MRST ``computeCentroids``: centroid of a 2D polygon (area-weighted
average of the sub-triangles with an auxiliary interior point)."""

import numpy as np

from .tri_area import tri_area


def computeCentroids(p):
    """Compute the centroid of the 2D polygon given by points ``p``."""
    p = np.asarray(p, dtype=float)
    p0 = np.sum(p, axis=0) / p.shape[0]
    areas = np.zeros(p.shape[0])
    pmids = np.zeros((p.shape[0], 2))
    p = np.vstack([p, p[0]])
    for i in range(len(areas)):
        pts = np.vstack([p[i:i + 2], p0])
        areas[i] = tri_area(pts[0], pts[1], pts[2])
        pmids[i] = np.sum(pts, axis=0) / 3
    pmid = np.zeros(2)
    pmid[0] = np.sum(pmids[:, 0] * areas) / np.sum(areas)
    pmid[1] = np.sum(pmids[:, 1] * areas) / np.sum(areas)
    return pmid
