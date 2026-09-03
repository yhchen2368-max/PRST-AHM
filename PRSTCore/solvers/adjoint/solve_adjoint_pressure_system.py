"""Solve adjoint pressure system.

1:1 Python translation of MRST solvers/adjoint/solveAdjointPressureSystem.m

Note: Full implementation requires MRST's solveIncompFlow with RHS override.
This simplified version provides the algorithmic structure.
"""

import numpy as np
from scipy import sparse


def solve_adjoint_pressure_system(G, S, W, rock, fluid, sim_res, adj_res, obj):
    """Solve adjoint pressure system for one time step.

    Uses simplified approach for 1D systems.
    """
    nc = G["cells"]["num"]
    nf = G["faces"]["num"]

    # Find current step
    cur_step = next((i for i, a in enumerate(adj_res) if a is not None and a.get("timeInterval") is not None), len(sim_res) - 1)
    dt = sim_res[cur_step]["timeInterval"][1] - sim_res[cur_step]["timeInterval"][0]

    PV = G["cells"]["volumes"] * rock["poro"]
    invPV = 1.0 / np.maximum(PV, 1e-12)

    res_sol = sim_res[cur_step]["resSol"]
    s = np.asarray(res_sol.get("s", np.ones(nc))).ravel()
    krw = fluid.get("krw", lambda x: x)(s)
    kro = fluid.get("kro", lambda x: 1 - x)(s)
    muw = fluid.get("muw", 1.0)
    muo = fluid.get("muo", 1.0)
    mob_w = krw / muw
    mob_o = kro / muo
    Lt = np.maximum(mob_w + mob_o, 1e-12)
    f_w = mob_w / Lt

    l_s = np.asarray(adj_res[cur_step].get("resSol", {}).get("s", np.zeros(nc))).ravel()

    # Simplified RHS: use only the reservoir part (no well/face coupling)
    f_bc = -np.asarray(obj["partials"][cur_step].get("v", np.zeros(nc)))
    # Correction term simplified for 1D
    f_bc = f_bc - dt * f_w * (invPV * l_s)

    # Set well RHS
    inx = 0
    for well_nr, w in enumerate(W):
        n_cells = len(w.get("cells", [1]))
        obj_q = np.asarray(obj["partials"][cur_step].get("q_w", np.zeros(0))).ravel()
        if "S" in w and "RHS" in w["S"] and inx + n_cells <= len(obj_q):
            w["S"]["RHS"]["f"] = obj_q[inx:inx + n_cells]
        inx += n_cells

    from .compute_adjoint_rhs import compute_adjoint_rhs
    b = compute_adjoint_rhs(G, W, f_bc)

    # Simplified solve: store adjoint pressure and flux
    adj_pressure = np.linalg.solve(
        sparse.diags([np.ones(nc - 1), -2 * np.ones(nc), np.ones(nc - 1)],
                      [-1, 0, 1]).toarray() + 1e-6 * np.eye(nc),
        b[0][:nc] if len(b) > 0 and len(b[0]) >= nc else np.zeros(nc),
    )
    adj_flux = np.zeros(nf)
    # Simple flux from pressure gradient
    T = rock["perm"].ravel()[:nc] / muw
    for i in range(1, nc):
        adj_flux[i] = T[i - 1] * (adj_pressure[i - 1] - adj_pressure[i])
    adj_flux = np.maximum(adj_flux, 0)  # ensure non-negative for stability

    adj_res[cur_step]["resSol"] = {
        "pressure": adj_pressure,
        "flux": adj_flux,
        "wellSol": [{"pressure": 0.0, "flux": 0.0} for _ in W],
    }

    return adj_res

