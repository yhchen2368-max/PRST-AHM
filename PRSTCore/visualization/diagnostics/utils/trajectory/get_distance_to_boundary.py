"""MRST trajectory ``getDistanceToBoundary.m`` counterpart."""

import numpy as np


def get_distance_to_boundary(G, p, vec, faceIx=None):
    """Approximate ray distance to boundary faces.

    For structured PRSTCore grids without face-node geometry this falls back
    to the grid bounding box.  If face centroids/normals are available, the
    closest forward intersection with face planes is returned.
    """
    p = np.asarray(p, dtype=float).ravel()
    vec = np.asarray(vec, dtype=float).ravel()
    norm = float(np.linalg.norm(vec))
    if norm < np.sqrt(np.finfo(float).eps) * max(float(np.linalg.norm(p)), 1.0):
        return np.inf, None, None
    direction = vec / norm
    faces = G.get("faces", {})
    centroids = np.asarray(faces.get("centroids", []), dtype=float)
    normals = np.asarray(faces.get("normals", []), dtype=float)
    if faceIx is not None and centroids.size and normals.size:
        ix = np.asarray(faceIx, dtype=int).ravel()
        if ix.size and np.min(ix) >= 1 and np.max(ix) <= centroids.shape[0]:
            ix = ix - 1
        best = (np.inf, None)
        for f in ix:
            n = normals[int(f)]
            den = float(np.dot(direction, n))
            if abs(den) < np.finfo(float).eps:
                continue
            t = float(np.dot(centroids[int(f)] - p, n) / den)
            if t >= 0.0 and t < best[0]:
                best = (t, int(f))
        if best[1] is not None:
            return best[0], best[1], True
    coords = np.asarray(G.get("nodes", {}).get("coords", G.get("cells", {}).get("centroids", [])), dtype=float)
    if coords.size == 0:
        return np.inf, None, None
    lo = np.min(coords, axis=0)
    hi = np.max(coords, axis=0)
    tmin = -np.inf
    tmax = np.inf
    for dim in range(min(3, direction.size)):
        if abs(direction[dim]) < np.finfo(float).eps:
            if p[dim] < lo[dim] or p[dim] > hi[dim]:
                return np.inf, None, False
            continue
        t1 = (lo[dim] - p[dim]) / direction[dim]
        t2 = (hi[dim] - p[dim]) / direction[dim]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    inside = tmin <= 0.0 <= tmax
    dist = tmax if inside else tmin
    return float(dist), None, bool(inside)


getDistanceToBoundary = get_distance_to_boundary

