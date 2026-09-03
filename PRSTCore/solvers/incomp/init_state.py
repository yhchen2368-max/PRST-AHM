"""Python port of MRST's ``initResSol.m`` / ``initWellSol.m`` (mrst-2026a/core/solvers)."""

from __future__ import annotations

import numpy as np


def init_res_sol(G: dict, p0, s0=1.0) -> dict:
    """Port of MRST ``initResSol.m``.

    ``p0`` and ``s0`` are broadcast to every cell if given as scalars (or a
    single row for ``s0`` with multiple phase columns). Returns
    ``{'pressure', 'flux', 's'}``.
    """
    nc = G["cells"]["num"]
    nf = G["faces"]["num"]

    s0 = np.atleast_2d(np.asarray(s0, dtype=float))
    if s0.shape[0] == 1:
        s0 = np.tile(s0, (nc, 1))
    elif s0.shape[0] != nc:
        raise ValueError(f"Initial saturation must have 1 or {nc} rows, got {s0.shape[0]}")

    p0 = np.asarray(p0, dtype=float)
    if p0.size == 1:
        p0 = np.full(nc, float(p0))
    elif p0.size != nc:
        raise ValueError(f"Initial pressure must have 1 or {nc} entries, got {p0.size}")

    return {"pressure": p0, "flux": np.zeros(nf), "s": s0 if s0.shape[1] > 1 else s0[:, 0]}


def init_well_sol(wells, p0: float) -> list[dict]:
    """Port of MRST ``initWellSol.m``: uniform bhp, zero perforation rates."""
    return [
        {"flux": np.zeros(np.asarray(w["cells"]).size), "pressure": float(p0)}
        for w in wells
    ]


def init_state(G: dict, W, p0, s0=None) -> dict:
    """Port of MRST ``initState.m`` (mrst-2026a/core/solvers).

    The reservoir solution plus, when wells are given, their well
    solution. When ``s0`` is supplied its column count must match the
    number of phases each well declares in ``compi`` -- a mismatch there
    means the wells and the fluid disagree about how many phases exist,
    which produces nonsense rather than an error further downstream.
    """
    state = init_res_sol(G, p0) if s0 is None else init_res_sol(G, p0, s0)

    if W is not None and len(W) > 0:
        state["wellSol"] = init_well_sol(W, p0)
        if s0 is not None:
            try:
                compi = np.vstack([np.atleast_1d(np.asarray(w["compi"],
                                                            dtype=float))
                                   for w in W])
            except Exception as exc:
                raise ValueError(
                    "Well compositions are inconsistently specified") from exc
            ncomp = compi.shape[1]
            nphase = np.atleast_2d(np.asarray(s0, dtype=float)).shape[1]
            if ncomp != nphase:
                raise AssertionError(
                    "Number of phases in well definition (%d) does not match "
                    "number of phases in reservoir fluids (%d)\nConsider "
                    "using the 'Comp_i' option of function 'add_well'."
                    % (ncomp, nphase))
    return state
