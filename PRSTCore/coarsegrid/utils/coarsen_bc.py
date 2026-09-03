"""Coarsen boundary conditions to coarse grid.

1:1 Python translation of MRST multiscale/coarsegrid/utils/coarsenBC.m
"""

import numpy as np


def coarsen_bc(cg, bc):
    """Map fine-grid boundary conditions to coarse grid.

    Parameters
    ----------
    cg : dict
        Coarse grid with parent, faces.connPos, faces.fconn, faces.num.
    bc : dict
        Fine-grid BC with 'face', 'type', 'value', optional 'sat'.

    Returns
    -------
    dict or None
        Coarse-grid boundary conditions.
    """
    if bc is None or len(bc.get("face", [])) == 0:
        return None

    parent = cg["parent"]
    nf_fine = parent["faces"]["num"]
    nf_coarse = cg["faces"]["num"]

    # Map fine faces to coarse faces
    cf = np.zeros(nf_fine, dtype=int)
    center_face = cg["faces"]["fconn"][cg["faces"]["connPos"][:nf_coarse]]
    cf[center_face] = np.arange(1, nf_coarse + 1)

    bc_faces = np.atleast_1d(bc["face"]).astype(int)
    coarse_faces_present = np.unique(cf[bc_faces - 1])
    coarse_faces_present = coarse_faces_present[coarse_faces_present > 0]

    bc_coarse = {"face": [], "type": [], "value": []}
    areas = parent["faces"]["areas"]

    for cf_idx in coarse_faces_present:
        fine_faces_in_coarse = np.where(cf == cf_idx)[0]
        bc_in_face = np.intersect1d(bc_faces - 1, fine_faces_in_coarse, return_indices=True)

        if len(bc_in_face[0]) == 0:
            continue

        types = np.atleast_1d(bc["type"])[bc_in_face[1]]
        values = np.atleast_1d(bc["value"])[bc_in_face[1]]
        fine_areas = areas[bc_faces[bc_in_face[1]] - 1]

        btype = types[0].lower() if isinstance(types[0], str) else str(types[0])

        if btype == "pressure":
            # Sample at center face
            val = float(values[0])
        elif btype == "flux":
            # Accumulate (with sign from fineToCoarseSign)
            from .fine_to_coarse_sign import fine_to_coarse_sign
            sgn = fine_to_coarse_sign(cg)
            fconn = np.asarray(cg["faces"]["fconn"])
            conn_pos = np.asarray(cg["faces"]["connPos"])
            start, end = conn_pos[cf_idx - 1], conn_pos[cf_idx]
            coarse_sgn = sgn[start:end]
            val = float(np.sum(values * coarse_sgn[:len(values)]))
        else:
            val = float(np.average(values, weights=fine_areas))

        bc_coarse["face"].append(int(cf_idx))
        bc_coarse["type"].append(btype)
        bc_coarse["value"].append(val)

    return bc_coarse
