"""Compute gradient from adjoint results.

1:1 Python translation of MRST solvers/adjoint/computeGradient.m
"""

import numpy as np
from scipy.linalg import block_diag
from .controls2wells import controls2wells


def compute_gradient(W, adj_res, schedule, controls):
    """Compute gradient of objective w.r.t. control variables.

    Parameters
    ----------
    W : list of dict
        Well structures.
    adj_res : list of dict
        Adjoint results with resSol.wellSol.pressure/flux.
    schedule : list of dict
        Schedule.
    controls : dict
        Controls structure.

    Returns
    -------
    list of ndarray
        Gradient for each control step.
    """
    bhp_wells = [i for i, w in enumerate(W) if str(w.get("type", "")).lower() == "bhp"]
    rate_wells = [i for i, w in enumerate(W) if str(w.get("type", "")).lower() == "rate"]

    # Build D_w (block diagonal of well D matrices)
    D_blocks = [np.atleast_2d(w.get("S", {}).get("D", np.eye(1))) for w in W]
    Dw = block_diag(*[np.atleast_2d(d) for d in D_blocks])
    DwD = Dw[:, bhp_wells] if bhp_wells else Dw

    # Projector for equality constraints
    ec = controls.get("linEqConst")
    if ec is not None:
        A_eq = np.atleast_2d(ec["A"])
        projector = np.eye(A_eq.shape[1]) - A_eq.T @ np.linalg.solve(A_eq @ A_eq.T, A_eq)
    else:
        projector = None

    A_N_list, b_N_list, A_D_list, b_D_list = controls2wells(W, schedule, controls)

    grad = []
    for k in range(len(A_N_list)):
        adj_well_pres = np.array([adj_res[k + 1]["resSol"]["wellSol"][i].get("pressure", 0)
                                   for i in range(len(W))])
        adj_well_flux = np.array([adj_res[k + 1]["resSol"]["wellSol"][i].get("flux", 0)
                                   for i in range(len(W))])

        l_p = adj_well_pres[rate_wells] if rate_wells else np.array([])
        l_q = adj_well_flux

        if len(l_p) > 0 and A_N_list[k].size > 0:
            grad_k = A_N_list[k].T @ l_p + A_D_list[k].T @ DwD.T @ l_q
        else:
            grad_k = A_D_list[k].T @ DwD.T @ l_q

        if projector is not None:
            grad_k = projector @ grad_k

        grad.append(grad_k)

    if controls.get("numControlSteps", len(schedule)) == 1:
        grad_mat = np.column_stack(grad)
        avg = np.mean(grad_mat, axis=1)
        grad = [avg] * len(grad)

    return grad
