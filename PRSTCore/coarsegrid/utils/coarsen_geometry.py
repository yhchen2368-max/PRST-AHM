"""Coarsen geometry for coarse grid.

1:1 Python translation of MRST multiscale/coarsegrid/utils/coarsenGeometry.m
"""

import numpy as np
from scipy import sparse


def coarsen_geometry(cg):
    """Add geometry (centroids, volumes) to a coarse grid.

    Parameters
    ----------
    cg : dict
        Coarse grid with parent, partition, cells.num.

    Returns
    -------
    dict
        Updated coarse grid with cells.volumes, cells.centroids.
    """
    parent = cg["parent"]
    nc = parent["cells"]["num"]
    ncoarse = cg["cells"]["num"]

    p = cg["partition"]
    centroids = np.asarray(parent["cells"]["centroids"])
    volumes = np.asarray(parent["cells"]["volumes"])

    nd = centroids.shape[1]

    # Accumulate volume-weighted centroids and total volumes
    M = sparse.csr_matrix((volumes, (p - 1, np.arange(nc))), shape=(ncoarse, nc))
    coarse_vol = M.sum(axis=1).A1
    coarse_centroids = (M @ centroids) / coarse_vol[:, np.newaxis]

    cg["cells"]["volumes"] = coarse_vol
    cg["cells"]["centroids"] = coarse_centroids

    # Coarse face geometry
    if "faces" in cg and "connPos" in cg["faces"] and "fconn" in cg["faces"]:
        ncf = cg["faces"]["num"]
        fine_areas = np.asarray(parent["faces"]["areas"])
        fine_normals = np.asarray(parent["faces"]["normals"])
        fine_centroids_f = np.asarray(parent["faces"]["centroids"])

        conn_pos = np.asarray(cg["faces"]["connPos"])
        fconn = np.asarray(cg["faces"]["fconn"])

        coarse_normals = np.zeros((ncf, nd))
        coarse_centroids_fc = np.zeros((ncf, nd))

        sgn = _fine_to_coarse_sign(cg)

        for i in range(ncf):
            start, end = conn_pos[i], conn_pos[i + 1]
            ff = fconn[start:end]
            sg = sgn[start:end]
            total_area = np.sum(fine_areas[ff])
            if total_area > 1e-12:
                coarse_normals[i] = np.sum(fine_normals[ff] * sg[:, np.newaxis], axis=0)
                coarse_centroids_fc[i] = np.average(fine_centroids_f[ff], axis=0,
                                                    weights=fine_areas[ff])

        cg["faces"]["normals"] = coarse_normals
        cg["faces"]["centroids"] = coarse_centroids_fc
        cg["faces"]["areas"] = np.linalg.norm(coarse_normals, axis=1)

    return cg


def _fine_to_coarse_sign(cg):
    """Internal helper matching fineToCoarseSign."""
    parent = cg["parent"]
    faceno = np.repeat(np.arange(cg["faces"]["num"]),
                       np.diff(cg["faces"]["connPos"]))
    # p is 1-based partition; neighbor cell IDs are 0-based (0 = boundary)
    p_ext = np.concatenate([[0], cg["partition"]])
    fconn = np.asarray(cg["faces"]["fconn"])
    c1 = parent["faces"]["neighbors"][fconn, 0]
    b1 = cg["faces"]["neighbors"][faceno, 0]
    # Get coarse block for c1
    c1_coarse = np.array([p_ext[ci] if ci > 0 else 0 for ci in c1])
    sgn = 2 * (c1_coarse == b1).astype(int) - 1
    return sgn
