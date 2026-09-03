"""Upscale a schedule to a coarser model.

1:1 Python translation of MRST autodiff/ad-core/upscale/upscaleSchedule.m
"""

import copy

import numpy as np


def upscale_schedule(model, schedule, well_upscale_method="recompute",
                      bc_upscale_method="linear"):
    """Convert a fine schedule to a coarse schedule.

    Parameters
    ----------
        model : dict or object
        Coarse model with G.partition, G.faces, rock.
    schedule : dict
        Fine schedule with 'step' and 'control'.
    well_upscale_method : str
        'recompute', 'sum', 'harmonic', or 'mean'.
    bc_upscale_method : str
        'linear', 'idw', 'mean', or 'nearest'.

    Returns
    -------
    dict
        Coarse schedule.
    """
    new_controls = []
    for ctrl in schedule["control"]:
        new_ctrl = copy.deepcopy(ctrl)

        # Wells
        if "W" in ctrl:
            W_coarse = [_handle_well(model, w, well_upscale_method) for w in ctrl["W"]]
            new_ctrl["W"] = W_coarse

        # Boundary conditions
        if "bc" in ctrl:
            new_ctrl["bc"] = _handle_bc(model, ctrl["bc"], bc_upscale_method)

        # Sources
        if "src" in ctrl:
            new_ctrl["src"] = _handle_src(model, ctrl["src"])

        new_controls.append(new_ctrl)

    return {
        "step": {
            "val": np.asarray(schedule["step"]["val"], dtype=float),
            "control": np.asarray(schedule["step"]["control"], dtype=int),
        },
        "control": new_controls,
    }


def _handle_well(model, W, well_upscale_method):
    """Upscale a single well."""
    G = _model_get(model, "G")
    rock = _model_get(model, "rock")
    p = np.asarray(G["partition"])
    wc = dict(W)

    # Map fine cells to coarse cells
    fine_cells = np.atleast_1d(W.get("cells", [0])).astype(int)
    pc = p[fine_cells - 1] if np.all(fine_cells > 0) else np.ones(1, dtype=int)

    # Unique stable (preserve order)
    unique_cells, first_ind, new_map = _unique_stable(pc)
    nc = len(unique_cells)

    wc["cells"] = unique_cells.tolist()
    counts = np.bincount(new_map, minlength=nc + 1)[:nc]
    counts = np.maximum(counts, 1)

    # Direction
    if "dir" in W:
        dval = W["dir"]
        wc["dir"] = dval if np.isscalar(dval) or isinstance(dval, str) else np.asarray(dval, dtype=object)[first_ind].tolist()

    # Radius
    if "r" in W:
        r_val = np.atleast_1d(W["r"])
        if len(r_val) > 1:
            wc["r"] = r_val[first_ind]
        elif len(r_val) == 1:
            wc["r"] = float(r_val[0])

    # Well index
    if well_upscale_method.lower() == "recompute":
        wc["WI"] = _compute_well_index(G, rock,
                                        wc.get("r", 0.1), unique_cells,
                                        wc.get("dir", "z"))
    elif well_upscale_method.lower() == "sum":
        wi = np.atleast_1d(W.get("WI", 0.0))
        if wi.size == 1 and new_map.size > 1:
            wi = np.full(new_map.size, wi[0])
        wc["WI"] = np.bincount(new_map, weights=wi, minlength=nc + 1)[:nc].tolist()
    elif well_upscale_method.lower() == "harmonic":
        wi = np.atleast_1d(W.get("WI", 0.0))
        if wi.size == 1 and new_map.size > 1:
            wi = np.full(new_map.size, wi[0])
        wi_inv = 1.0 / np.maximum(wi, 1e-15)
        wc["WI"] = (1.0 / (np.bincount(new_map, weights=wi_inv, minlength=nc + 1)[:nc] / counts)).tolist()
    elif well_upscale_method.lower() == "mean":
        wi = np.atleast_1d(W.get("WI", 0.0))
        if wi.size == 1 and new_map.size > 1:
            wi = np.full(new_map.size, wi[0])
        wc["WI"] = (np.bincount(new_map, weights=wi, minlength=nc + 1)[:nc] / counts).tolist()
    else:
        raise ValueError(f"Unknown upscale mode: {well_upscale_method}")

    # dZ
    wc["_base_WI"] = np.asarray(wc["WI"], dtype=float).ravel().copy()

    centroids = np.asarray(G["cells"]["centroids"])
    if centroids.shape[1] > 2:
        z = centroids[np.asarray(wc["cells"], dtype=int) - 1, 2]
    else:
        z = np.zeros(len(wc["cells"]))
    ref_depth = W.get("refDepth", 0.0)
    wc["dZ"] = (z - ref_depth).tolist()

    # cstatus
    wc["cstatus"] = np.ones(nc, dtype=bool).tolist()

    # Fine-to-coarse mapping
    wc["fperf"] = new_map.tolist()

    # Topology
    if "topo" in W:
        mp = np.concatenate([[0], new_map])
        newtopo = mp[np.array(W["topo"]) + 1]
        newtopo = np.sort(newtopo, axis=1)
        newtopo = np.unique(newtopo, axis=0)
        newtopo = newtopo[newtopo[:, 0] != newtopo[:, 1]]
        wc["topo"] = newtopo.tolist()

    wc["parentIndices"] = first_ind.tolist()

    return wc


def _compute_well_index(G, rock, r, cells, direction):
    """Compute a Peaceman-like well index per coarse completion.

    This is not yet MRST's full ``computeWellIndex`` for arbitrary grids, but
    it uses coarse-cell volume/permeability instead of the previous constants.
    """
    cells = np.atleast_1d(cells)
    perm = np.asarray(rock["perm"])
    if perm.ndim == 1:
        perm = perm.reshape(-1, 1)

    r_arr = np.atleast_1d(r).astype(float)
    if r_arr.size == 1 and cells.size > 1:
        r_arr = np.full(cells.size, r_arr[0])
    volumes = np.asarray(G.get("cells", {}).get("volumes", np.ones(G["cells"]["num"])), dtype=float).ravel()
    wi = np.zeros(cells.size, dtype=float)
    for ix, c in enumerate(cells):
        ci = int(c) - 1
        if ci < len(perm):
            if perm.shape[1] >= 2:
                k = float(np.sqrt(perm[ci, 0] * perm[ci, 1]))
            else:
                k = float(perm[ci, 0])
        else:
            k = 100e-15
        length = max(float(volumes[ci]) ** (1.0 / max(G.get("griddim", 3), 1)), 1e-12) if ci < volumes.size else 1.0
        rw = max(float(r_arr[min(ix, r_arr.size - 1)]), 1e-6)
        re = max(0.2 * length, rw * 1.0001)
        wi[ix] = 2.0 * np.pi * k * length / max(np.log(re / rw), 1e-12)
    return wi.tolist()


def _handle_bc(model, bc, method):
    """Upscale boundary conditions."""
    if bc is None or len(bc.get("face", [])) == 0:
        return None

    CG = _model_get(model, "G")
    G = CG["parent"]
    nfaces_coarse = CG["faces"]["num"]

    # Coarse face -> fine face map
    conn_coarse = np.repeat(np.arange(1, nfaces_coarse + 1),
                            np.diff(CG["faces"]["connPos"]))

    is_face_bc = np.zeros(G["faces"]["num"], dtype=bool)
    bc_faces = np.atleast_1d(bc["face"]).astype(int)
    is_face_bc[bc_faces - 1] = True

    coarse_face_no = np.zeros(G["faces"]["num"], dtype=int)
    coarse_face_no[CG["faces"]["fconn"]] = conn_coarse

    coarse_faces_bc = np.unique(conn_coarse[is_face_bc[CG["faces"]["fconn"]]])

    bc_coarse = {"face": [], "type": [], "value": [], "sat": []}
    for cf in coarse_faces_bc:
        is_current = coarse_face_no == cf
        act = is_current[bc_faces - 1] if len(bc_faces) > 0 else np.array([], dtype=bool)
        if not np.any(act):
            continue
        faces = bc_faces[act]
        areas = G["faces"]["areas"][faces - 1]
        values = np.atleast_1d(bc["value"])[act]
        types_list = np.atleast_1d(bc["type"])[act]

        btype = types_list[0]

        if method.lower() == "flux" or btype.lower() == "flux":
            bc_coarse["face"].append(int(cf))
            bc_coarse["type"].append("flux")
            bc_coarse["value"].append(float(np.sum(values)))
        elif method.lower() == "pressure" or btype.lower() == "pressure":
            bc_coarse["face"].append(int(cf))
            bc_coarse["type"].append("pressure")
            bc_coarse["value"].append(float(np.average(values, weights=areas)))
        else:
            # Default: area-weighted average
            bc_coarse["face"].append(int(cf))
            bc_coarse["type"].append(btype)
            bc_coarse["value"].append(float(np.average(values, weights=areas)))

    if len(bc_coarse["face"]) == 0:
        return None
    return bc_coarse


def _handle_src(model, src):
    """Upscale source terms (placeholder)."""
    if src is None:
        return None
    G = _model_get(model, "G")
    p = np.asarray(G["partition"])
    src_cells = np.atleast_1d(src.get("cells", [])).astype(int)
    pc = p[src_cells - 1] if len(src_cells) > 0 else np.array([])
    unique_cells = np.unique(pc)

    src_coarse = {"cells": unique_cells.tolist()}
    if "rate" in src:
        src_coarse["rate"] = np.bincount(pc, weights=np.atleast_1d(src["rate"]),
                                          minlength=G["cells"]["num"] + 1)
        src_coarse["rate"] = src_coarse["rate"][unique_cells].tolist()
    return src_coarse


def _model_get(model, name, default=None):
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)


def _unique_stable(values):
    values = np.asarray(values, dtype=int).ravel()
    mapping = {}
    unique = []
    first = []
    inv = np.zeros(values.size, dtype=int)
    for i, v in enumerate(values):
        key = int(v)
        if key not in mapping:
            mapping[key] = len(unique)
            unique.append(key)
            first.append(i)
        inv[i] = mapping[key]
    return np.asarray(unique, dtype=int), np.asarray(first, dtype=int), inv
