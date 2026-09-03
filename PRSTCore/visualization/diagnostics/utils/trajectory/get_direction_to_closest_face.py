"""MRST trajectory ``getDirectionToClosestFace.m`` counterpart."""

import numpy as np


def get_direction_to_closest_face(G, pnt, faceIx):
    centroids = np.asarray(G["faces"]["centroids"], dtype=float)
    faces = np.asarray(faceIx, dtype=int).ravel()
    if faces.size and np.min(faces) >= 1 and np.max(faces) <= centroids.shape[0]:
        faces = faces - 1
    pnt = np.asarray(pnt, dtype=float).ravel()
    d = centroids[faces] - pnt
    ix = int(np.argmin(np.sum(d * d, axis=1)))
    return d[ix], centroids[faces[ix]], int(faces[ix])


getDirectionToClosestFace = get_direction_to_closest_face

