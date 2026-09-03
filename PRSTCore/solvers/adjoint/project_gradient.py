"""Project gradient according to box and linear constraints.

1:1 Python translation of MRST solvers/adjoint/projectGradient.m
"""

import numpy as np


def project_gradient(controls, du, const_tol=1e-3, max_const_its=100, verbose_level=0):
    """Project gradient according to box, linear inequality, linear equality constraints.

    Parameters
    ----------
    controls : dict
        Controls structure.
    du : ndarray
        Raw gradient (numControls x numSteps).
    const_tol : float
        Constraint satisfaction tolerance.
    max_const_its : int
        Max iterations.
    verbose_level : int
        Verbosity.

    Returns
    -------
    ndarray
        Projected gradient.
    """
    num_steps = du.shape[1] if du.ndim > 1 else 1
    pdu = du.copy().reshape(-1, num_steps)
    u_init = np.array([w["values"] for w in controls["well"]]).T
    norm_u = np.linalg.norm(u_init, np.inf)

    tol = max(norm_u * const_tol / 2, 1e-12)

    min_max = np.array([w["minMax"] for w in controls["well"]])
    box_min = min_max[:, 0] + tol
    box_max = min_max[:, 1] - tol

    Aeq = None
    Beq = None
    if controls.get("linEqConst") is not None:
        Aeq = np.atleast_2d(controls["linEqConst"]["A"])
        Beq = np.atleast_1d(controls["linEqConst"]["b"])

    Aineq = None
    Bineq = None
    if controls.get("linIneqConst") is not None:
        Aineq = np.atleast_2d(controls["linIneqConst"]["A"])
        Bineq = np.atleast_1d(controls["linIneqConst"]["b"])

    for it in range(max_const_its):
        pdu_prev = pdu.copy()

        # Box constraints
        u_cur = u_init + pdu
        violated_min = u_cur < box_min[:, np.newaxis]
        violated_max = u_cur > box_max[:, np.newaxis]
        pdu[violated_min] = 0
        pdu[violated_max] = 0

        # Inequality constraints (simplified)
        if Aineq is not None:
            for col in range(num_steps):
                active = Aineq @ (u_init[:, col] + pdu[:, col]) > Bineq - tol
                if np.any(active):
                    A_act = Aineq[active]
                    P = np.eye(A_act.shape[1]) - A_act.T @ np.linalg.solve(
                        A_act @ A_act.T + np.eye(len(A_act)) * 1e-12, A_act)
                    pdu[:, col] = P @ pdu[:, col]

        # Equality constraints
        if Aeq is not None:
            P_eq = np.eye(Aeq.shape[1]) - Aeq.T @ np.linalg.solve(
                Aeq @ Aeq.T + np.eye(len(Aeq)) * 1e-12, Aeq)
            pdu = P_eq @ pdu

        if np.linalg.norm(pdu - pdu_prev) < tol:
            break

    return pdu
