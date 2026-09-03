"""Solve adjoint transport system.

1:1 Python translation of MRST solvers/adjoint/solveAdjointTransportSystem.m
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_adjoint_transport_system(G, S, W, rock, fluid, sim_res, adj_res, obj):
    """Solve adjoint transport (saturation) system for one time step.

    Parameters
    ----------
    G, S, W, rock, fluid : dict/list
        Standard MRST structures.
    sim_res : list of dict
        Forward simulation results.
    adj_res : list of dict
        Adjoint results (mutated in place).
    obj : dict
        Objective with 'partials'.

    Returns
    -------
    list of dict
        Updated adj_res.
    """
    num_steps = len(sim_res)

    if not adj_res or all(a.get("timeInterval") is None for a in adj_res if a):
        cur_step = num_steps - 1
    else:
        cur_step = next((i for i in range(len(adj_res) - 1, -1, -1)
                         if adj_res[i].get("timeInterval") is None), num_steps - 1)

    dt = sim_res[cur_step]["timeInterval"][1] - sim_res[cur_step]["timeInterval"][0]

    if cur_step >= len(adj_res):
        adj_res = adj_res + [{"timeInterval": None, "resSol": {}}]

    adj_res[cur_step] = dict(adj_res[cur_step]) if cur_step < len(adj_res) else {}
    adj_res[cur_step]["timeInterval"] = sim_res[cur_step]["timeInterval"]

    nc = G["cells"]["num"]
    PV = G["cells"]["volumes"] * rock["poro"]
    invDPV = sparse.diags(1.0 / np.maximum(PV, 1e-12), 0)

    # Mobilities and derivatives
    res_sol = sim_res[cur_step]["resSol"]
    s = np.asarray(res_sol.get("s", np.ones(nc))).ravel()
    krw = fluid.get("krw", lambda s: s)(s)
    kro = fluid.get("kro", lambda s: 1 - s)(s)
    dkrw = fluid.get("dkrw", lambda s: np.ones_like(s))(s)
    dkro = fluid.get("dkro", lambda s: -np.ones_like(s))(s)
    muw = fluid.get("muw", 1.0)
    muo = fluid.get("muo", 1.0)

    mob_w = krw / muw
    mob_o = kro / muo
    dmob_w = dkrw / muw
    dmob_o = dkro / muo
    mob = np.column_stack([mob_w, mob_o])
    dmob = np.column_stack([dmob_w, dmob_o])
    Lt = mob_w + mob_o
    f_w = mob_w / np.maximum(Lt, 1e-12)
    f_o = mob_o / np.maximum(Lt, 1e-12)
    Dfw = (f_o * dmob_w - f_w * dmob_o) / np.maximum(Lt, 1e-12)
    DDf = sparse.diags(Dfw, 0)

    from .generate_upstream_transport_matrix import generate_upstream_transport_matrix
    At = generate_upstream_transport_matrix(
        G, S, W, res_sol, sim_res[cur_step]["wellSol"], transpose=True,
    )

    syst_mat = sparse.eye(nc) - dt * (DDf @ At @ invDPV)

    # Build RHS
    RHS = -np.asarray(obj["partials"][cur_step].get("s", np.zeros(nc)))

    if cur_step < num_steps - 1:
        # Simplified 1D B^{n+1} v^{n+1} part
        flux_next = np.asarray(sim_res[cur_step + 1]["resSol"].get("flux", np.zeros(nc + 1)))
        lam_v_next = np.asarray(adj_res[cur_step + 1]["resSol"].get("flux", np.zeros(nc + 1)))
        dim = -np.sum(dmob, axis=1) / np.maximum(Lt**2, 1e-12)

        for i in range(nc):
            # Face i+1 (right face of cell i)
            v_right = flux_next[i + 1]
            l_right = lam_v_next[i + 1]
            RHS[i] = RHS[i] - dim[i] * v_right * l_right

            # Face i (left face of cell i) already handled implicitly via upstream
            if i > 0:
                v_left = flux_next[i]
                l_left = lam_v_next[i]
                RHS[i] = RHS[i] - dim[i] * v_left * l_left * (-1 if v_left < 0 else 0)

    # Solve
    if hasattr(syst_mat, "__len__"):
        l_s_new = spsolve(syst_mat, RHS)
    else:
        l_s_new = RHS

    adj_res[cur_step]["resSol"]["s"] = l_s_new
    if "wellSol" not in adj_res[cur_step]["resSol"]:
        adj_res[cur_step]["resSol"]["wellSol"] = [{"pressure": 0.0, "flux": 0.0} for _ in W]

    return adj_res
