"""Optimize with bound constraints using L-BFGS + QP trust-region.

1:1 Python translation of MRST optimizeBoundConstrained.m
"""

import numpy as np


def optimize_bound_constrained(u0, f, maximize=True, lb=0.0, ub=1.0,
                                step_init=None, max_initial_update=0.05,
                                grad_tol=1e-3, obj_change_tol=5e-4,
                                max_it=25, line_search_max_it=5,
                                wolfe1=1e-4, wolfe2=0.9,
                                lbfgs_num=10, lbfgs_strategy="dynamic",
                                plot_evolution=False, log_plot=False,
                                output_hessian=False, **kwargs):
    """Bound-constrained L-BFGS optimization with QP trust-region.

    This is a simplified version; full QP subspace handling is complex.

    Parameters
    ----------
    u0 : ndarray
        Initial guess in [lb, ub].
    f : callable
        Objective function returning (value, gradient).
    maximize : bool
        If True, maximize; if False, minimize.
    lb, ub : float or ndarray
        Lower and upper bounds.

    Returns
    -------
    v : float
        Optimal objective value.
    u : ndarray
        Optimal control vector.
    history : list of dict
        Iteration history.
    """
    from PRSTCore.optimization.optim.limited_memory_hessian import LimitedMemoryHessian
    from PRSTCore.optimization.optim.line_search import line_search

    u = np.clip(np.asarray(u0, dtype=float).ravel(), lb, ub)
    # line_search/LimitedMemoryHessian both assume standard minimize
    # semantics (descent direction, sufficient-decrease Wolfe condition).
    # Maximizing f is minimizing -f, so wrap f once here rather than
    # threading a sign flag through those two modules.
    sign = -1 if maximize else 1

    def f_internal(u_):
        v_, g_ = f(u_)
        return sign * v_, sign * np.asarray(g_, dtype=float).ravel()

    v, g = f_internal(u)

    if step_init is None or np.isnan(step_init) or step_init <= 0:
        step = max_initial_update / max(np.max(np.abs(g)), 1e-8)
    else:
        step = step_init

    Hi = LimitedMemoryHessian(
        init_scale=step, init_strategy=lbfgs_strategy, m=lbfgs_num, sign=1
    )

    history = []
    for it in range(max_it):
        # Compute search direction
        d = Hi.dot(-g)
        d = np.clip(d, -1.0, 1.0)

        pg_norm = np.linalg.norm(
            np.where((u <= lb + 1e-12) & (d < 0), 0,
                     np.where((u >= ub - 1e-12) & (d > 0), 0, d)),
            ord=np.inf,
        )
        if pg_norm < grad_tol:
            break

        ls_opt = {
            "wolfe1": wolfe1,
            "wolfe2": wolfe2,
            "safeguardFac": kwargs.get("safeguardFac", 1e-5),
            "stepIncreaseTol": kwargs.get("stepIncreaseTol", 10),
            "lineSearchMaxIt": line_search_max_it,
            "maxStep": 1.0,
        }

        u_new, v_new, g_new, ls_info = line_search(u, v, g, d, f_internal, ls_opt)
        u_new = np.clip(u_new, lb, ub)
        v_new, g_new = f_internal(u_new)

        s = u_new - u
        y = g_new - g
        if s.dot(y) > 1e-12:
            Hi.update(s, y)

        u, v, g = u_new, v_new, g_new
        history.append({"val": sign * v, "u": u.copy(), "pg": pg_norm})

        if len(history) > 1 and abs(v - sign * history[-2]["val"]) < obj_change_tol:
            break

    return sign * v, u, history
