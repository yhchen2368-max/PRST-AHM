"""Upscale a fine model into a coarser version using a partition vector.

1:1 Python translation of MRST autodiff/ad-core/upscale/upscaleModelTPFA.m
"""

import copy

import numpy as np
from scipy import sparse

from PRSTCore.coarsegrid.generate_coarse_grid import generate_coarse_grid
from PRSTCore.coarsegrid.process_partition import compress_partition, process_partition
from PRSTCore.coarsegrid.utils.coarsen_geometry import coarsen_geometry


def upscale_model_tpfa(model, partition, validate_partition=True,
                        trans_from_rock=True, trans_coarse=None,
                        perm_coarse=None, neighborship=None,
                        poro_coarse=None, pv_coarse=None):
    """Create a coarse model from a fine model and partition.

    Parameters
    ----------
    model : dict
        Fine-scale model with G, rock, operators.
    partition : ndarray
        Partition vector (1..N), length = model.G.cells.num.
    validate_partition : bool
        Process and compress partition.
    trans_from_rock : bool
        Compute transmissibility from upscaled rock.
    trans_coarse : ndarray, optional
        Pre-computed coarse transmissibilities.
    perm_coarse : ndarray, optional
        Pre-computed coarse permeability.
    neighborship : ndarray, optional
        Coarse face neighbors.
    poro_coarse : ndarray, optional
        Pre-computed coarse porosity.

    Returns
    -------
    dict
        Coarse model.
    """
    G = _model_get(model, "G")
    rock = _model_get(model, "rock")

    # Handle grid
    CG = _get_grid(G, partition, validate_partition)

    # Handle rock
    rock_c = _get_rock(rock, CG, poro_coarse, perm_coarse)

    # Handle transmissibility
    Tc, Nc = _get_transmissibility(CG, rock_c, trans_coarse, neighborship,
                                    trans_from_rock, model)

    ops = _setup_operators_tpfa(CG, rock_c, Nc, Tc,
                                _coarse_pore_volume(model, CG, rock_c,
                                                    pv_coarse))
    model_c = _copy_model(model)
    _model_set(model_c, "G", CG)
    _model_set(model_c, "rock", rock_c)
    _model_set(model_c, "operators", ops)
    _model_set(model_c, "porevolume", ops["pv"])
    _model_set(model_c, "FlowDiscretization", None)

    return model_c


def _get_grid(G, partition, validate_partition):
    """Generate coarse grid from partition."""
    if isinstance(partition, dict):
        return partition  # Already a coarse grid

    p = np.asarray(partition, dtype=int)
    if validate_partition:
        p = process_partition(G, p)
        p = compress_partition(p)

    CG = generate_coarse_grid(G, p)
    CG = coarsen_geometry(CG)
    return CG


def _get_rock(rock, CG, poro_coarse, perm_coarse):
    """Upscale rock properties."""
    p = CG["partition"]
    vol = CG["parent"]["cells"].get("volumes", np.ones(len(p)))
    coarsevol = np.bincount(p, weights=vol, minlength=CG["cells"]["num"] + 1)[1:]
    counts = np.bincount(p, minlength=CG["cells"]["num"] + 1)[1:]
    counts = np.maximum(counts, 1)

    # Porosity
    if poro_coarse is None and "poro" in rock:
        poro = np.asarray(rock["poro"]).ravel()
        if "ntg" in rock:
            poro = poro * np.asarray(rock["ntg"]).ravel()
        poro_c = np.bincount(p, weights=poro * vol, minlength=CG["cells"]["num"] + 1)[1:] / np.maximum(coarsevol, 1e-12)
    elif poro_coarse is not None:
        poro_c = np.asarray(poro_coarse).ravel()
    else:
        poro_c = np.ones(CG["cells"]["num"])

    # Permeability
    if perm_coarse is None and "perm" in rock:
        perm = np.asarray(rock["perm"])
        if perm.ndim == 1:
            perm = perm.reshape(-1, 1)
        nK = perm.shape[1]
        perm_c = np.zeros((CG["cells"]["num"], nK))
        for i in range(nK):
            perm_c[:, i] = np.bincount(p, weights=1.0 / np.maximum(perm[:, i], 1e-15),
                                        minlength=CG["cells"]["num"] + 1)[1:]
        perm_c = counts.reshape(-1, 1) / np.maximum(perm_c, 1e-15)
    elif perm_coarse is not None:
        perm_c = np.asarray(perm_coarse)
        if perm_c.ndim == 1:
            perm_c = perm_c.reshape(-1, 1)
    else:
        perm_c = np.ones((CG["cells"]["num"], 1))

    rock_c = {"poro": poro_c, "perm": perm_c}

    # Regions
    if "regions" in rock:
        rock_c["regions"] = {}
        for t in ["saturation", "pvt"]:
            if t in rock["regions"]:
                rock_c["regions"][t] = _coarsen_region(CG, rock, t)

    return rock_c


def _coarsen_region(CG, rock, t):
    """Coarsen a region field."""
    reg_fine = np.asarray(rock["regions"][t])
    p = CG["partition"]
    ncoarse = CG["cells"]["num"]
    reg_coarse = np.zeros(ncoarse, dtype=reg_fine.dtype)
    for i in range(ncoarse):
        act = p == (i + 1)
        u = np.unique(reg_fine[act])
        reg_coarse[i] = u[0]
    return reg_coarse


def _get_transmissibility(CG, rock_c, trans_coarse, neighborship,
                           trans_from_rock, model):
    """Compute coarse transmissibilities."""
    if neighborship is None:
        N = CG["faces"].get("neighbors", np.zeros((CG["faces"]["num"], 2), dtype=int))
    else:
        N = np.asarray(neighborship)

    nF = N.shape[0]
    intx = np.all(N != 0, axis=1)
    nIF = intx.sum()

    if trans_coarse is None or len(trans_coarse) == 0:
        if trans_from_rock:
            Tc = _get_face_transmissibility(CG, rock_c)
        else:
            # Sum fine transmissibilities
            cts = np.repeat(np.arange(CG["faces"]["num"]),
                            np.diff(CG["faces"]["connPos"]))
            fine_T_all = _fine_face_transmissibility(model)
            if fine_T_all is not None:
                Tc = np.bincount(cts,
                                 weights=fine_T_all[CG["faces"]["fconn"]],
                                 minlength=CG["faces"]["num"] + 1)[:CG["faces"]["num"]]
            else:
                Tc = np.ones(nF)
    elif len(trans_coarse) == nF:
        Tc = np.asarray(trans_coarse).ravel()
    elif len(trans_coarse) == nIF:
        Tc_dummy = _compute_trans(CG, rock_c)
        Tc = np.zeros(nF)
        Tc[intx] = np.asarray(trans_coarse).ravel()
        Tc[~intx] = Tc_dummy[~intx]
    else:
        raise ValueError(f"Number of transmissibility entries ({len(trans_coarse)}) "
                         f"does not match face count ({nF}) or interface count ({nIF})")

    return Tc, N


def _get_face_transmissibility(CG, rock_c):
    """Compute face transmissibility from rock (harmonic average)."""
    nF = CG["faces"]["num"]
    N = CG["faces"].get("neighbors", np.zeros((nF, 2), dtype=int))
    perm = np.asarray(rock_c["perm"])
    if perm.ndim == 1:
        perm = perm.reshape(-1, 1)
    kx = perm[:, 0] if perm.shape[1] >= 1 else np.ones(CG["cells"]["num"])

    Tc = np.ones(nF)
    for f in range(nF):
        c1, c2 = N[f, 0], N[f, 1]
        if c1 > 0 and c2 > 0:
            k1, k2 = kx[c1 - 1], kx[c2 - 1]
            if k1 > 0 and k2 > 0:
                Tc[f] = 1.0 / (1.0 / k1 + 1.0 / k2)
    return Tc


def _compute_trans(CG, rock_c):
    """Compute transmissibility from rock (simplified)."""
    return _get_face_transmissibility(CG, rock_c)


def _coarse_pore_volume(model, CG, rock_c, pv_coarse):
    """The coarse pore volume, summed from the fine model's.

    MRST-0 takes ``accumarray(partition, model.operators.pv)`` rather
    than recomputing porosity times coarse bulk volume. The two are not
    the same: recomputing leaves the coarse model holding a different
    amount of fluid than the fine one it stands for, and a calibrated
    coarse model that cannot match the fine model's total pore volume
    cannot match its material balance either.

    Falls back to the geometric product only when the fine model has no
    pore volume to sum.
    """
    if pv_coarse is not None:
        return np.asarray(pv_coarse, dtype=float).ravel()

    ops = _model_get(model, "operators", {}) or {}
    pv_fine = ops.get("pv") if isinstance(ops, dict) \
        else getattr(ops, "pv", None)
    partition = np.asarray(CG["partition"], dtype=int)
    ncoarse = int(CG["cells"]["num"])
    if pv_fine is not None and np.size(pv_fine) == partition.size:
        pv_fine = np.asarray(pv_fine, dtype=float).ravel()
        return np.bincount(partition, weights=pv_fine,
                           minlength=ncoarse + 1)[1:]

    return np.asarray(CG["cells"]["volumes"], dtype=float).ravel() \
        * np.asarray(rock_c["poro"], dtype=float).ravel()


def _setup_operators_tpfa(CG, rock_c, N, T, pv):
    """Setup operators for TPFA.

    Connections with zero transmissibility are dropped as well as
    external ones -- MRST-0's ``ix = Tc ~= 0`` -- since a zero-weight
    connection contributes nothing but still costs a Jacobian entry.
    """
    N_all = np.asarray(N, dtype=int)
    T_all = np.asarray(T, dtype=float).ravel()
    internal = np.all(N_all != 0, axis=1) & (T_all != 0)
    return {
        "N": N_all[internal],
        "T": T_all[internal],
        "T_all": T_all,
        "pv": np.asarray(pv, dtype=float).ravel(),
    }


def _copy_model(model):
    if isinstance(model, dict):
        return dict(model)
    return copy.copy(model)


def _model_get(model, name, default=None):
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)


def _model_set(model, name, value):
    if isinstance(model, dict):
        model[name] = value
    else:
        setattr(model, name, value)


def _fine_face_transmissibility(model):
    """Map fine-grid face rows to transmissibility values.

    MRST stores ``operators.T_all`` per grid face.  PRSTCore fine models often
    only store internal ``operators.N/T``.  This reconstructs a per-face
    array when the internal-neighbor ordering is consistent with the grid.
    """
    ops = _model_get(model, "operators", {}) or {}
    G = _model_get(model, "G", {}) or {}
    if "T_all" in ops:
        return np.asarray(ops["T_all"], dtype=float).ravel()
    if "T" not in ops or "N" not in ops or "faces" not in G:
        return None

    nbrs = np.asarray(G["faces"].get("neighbors", []), dtype=int)
    if nbrs.ndim != 2 or nbrs.shape[1] < 2:
        return None
    internal_faces = np.where(np.all(nbrs[:, :2] != 0, axis=1))[0]
    N = np.asarray(ops["N"], dtype=int)
    T = np.asarray(ops["T"], dtype=float).ravel()
    if N.shape[0] != T.size:
        return None

    out = np.zeros(int(G["faces"]["num"]), dtype=float)
    if internal_faces.size == T.size:
        grid_N = nbrs[internal_faces, :2]
        if np.array_equal(grid_N, N):
            out[internal_faces] = T
            return out
        if np.array_equal(np.sort(grid_N, axis=1), np.sort(N, axis=1)):
            out[internal_faces] = T
            return out

    # Fallback: map by unordered cell pair.
    pair_to_t = {tuple(row): float(t) for row, t in zip(np.sort(N, axis=1), T)}
    filled = 0
    for fi in internal_faces:
        key = tuple(np.sort(nbrs[fi, :2]))
        if key in pair_to_t:
            out[fi] = pair_to_t[key]
            filled += 1
    return out if filled else None
