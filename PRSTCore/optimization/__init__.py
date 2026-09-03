"""MRST optimization Python package.

1:1 translation of autodiff/optimization/ MATLAB code.

This package merges the flat PRSTCore.optimization module with the
subpackage structure matching MRST's autodiff/optimization/ layout.
"""

import numpy as np
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.optimization.utils.parameters import update_setup_from_scaled_parameters


def _finite_difference_gradient(fun, x, epsilon=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    f0 = fun(x)
    for i in range(x.size):
        h = epsilon * max(1.0, abs(x[i]))
        dx = np.zeros_like(x)
        dx[i] = h
        grad[i] = (fun(x + dx) - f0) / h
    return grad


def _finite_difference_jacobian(fun, x, epsilon=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = fun(x)
    f0 = np.atleast_1d(f0)
    J = np.zeros((f0.size, x.size), dtype=float)
    for i in range(x.size):
        h = epsilon * max(1.0, abs(x[i]))
        dx = np.zeros_like(x)
        dx[i] = h
        fi = np.atleast_1d(fun(x + dx))
        J[:, i] = (fi - f0) / h
    return J


def evaluate_match(pvec, obj, setup, parameters, states_ref, **kwargs):
    opt = {
        "Gradient": "AdjointAD",
        "objScaling": 1.0,
        "NonlinearSolver": None,
        "Verbose": False,
    }
    opt.update(kwargs)
    pvec = np.asarray(pvec, dtype=float)
    setup_new = update_setup_from_scaled_parameters(setup, parameters, pvec)
    well_sols, states = simulate_schedule_ad(
        setup_new["state0"], setup_new["model"], setup_new["schedule"],
        NonLinearSolver=opt["NonlinearSolver"], Verbose=opt["Verbose"],
    )
    result = obj(setup_new["model"], states, setup_new["schedule"], states_ref, False, None, None)
    misfit_vals = np.concatenate([np.atleast_1d(x) for x in result])
    misfit_val = -np.sum(misfit_vals) / opt["objScaling"]
    grad = None
    if opt["Gradient"] != "none":
        def scalar_obj(u):
            return evaluate_match(u, obj, setup, parameters, states_ref,
                                  Gradient="none", objScaling=opt["objScaling"],
                                  NonlinearSolver=opt["NonlinearSolver"],
                                  Verbose=opt["Verbose"])[0]
        grad = _finite_difference_gradient(scalar_obj, pvec)
    return misfit_val, grad, well_sols, states


def evaluate_match_summands(pvec, obj, setup, parameters, states_ref, obj_scaling=1.0,
                             nonlinear_solver=None, verbose=False, enforce_bounds=True,
                             accumulate_residuals=None):
    pvec = np.asarray(pvec, dtype=float)
    if enforce_bounds:
        pvec = np.clip(pvec, 0.0, 1.0)
    setup_new = update_setup_from_scaled_parameters(setup, parameters, pvec)
    well_sols, states = simulate_schedule_ad(
        setup_new["state0"], setup_new["model"], setup_new["schedule"],
        NonLinearSolver=nonlinear_solver, Verbose=verbose,
    )
    misfit_vals = np.concatenate([
        np.atleast_1d(x) for x in obj(setup_new["model"], states,
                                       setup_new["schedule"], states_ref,
                                       False, None, None)
    ])
    J = _finite_difference_jacobian(
        lambda u: np.concatenate([
            np.atleast_1d(x) for x in obj(setup_new["model"], states,
                                           setup_new["schedule"], states_ref,
                                           False, None, None)
        ]), pvec,
    )
    return misfit_vals, J, setup_new


def unit_box_bfgs(pinit, objh, maximize=True, step_init=None,
                   max_initial_update=0.05, grad_tol=1e-3,
                   obj_change_tol=5e-4, max_it=25, output_hessian=False,
                   log_plot=False, **kwargs):
    u = np.clip(np.asarray(pinit, dtype=float), 0.0, 1.0)
    v, g, *_ = objh(u)
    if g is None:
        raise NotImplementedError("Gradient required for BFGS")
    if step_init is None or step_init <= 0 or np.isnan(step_init):
        step = max_initial_update / max(np.max(np.abs(g)), 1e-8)
    else:
        step = step_init
    n = u.size
    H = np.eye(n)
    history = []
    for it in range(max_it):
        if np.linalg.norm(g, ord=np.inf) < grad_tol:
            break
        direction = H @ g
        if not maximize:
            direction = -direction
        alpha = step
        u_new = np.clip(u + alpha * direction, 0.0, 1.0)
        v_new, g_new, *_ = objh(u_new)
        if (maximize and v_new < v) or (not maximize and v_new > v):
            alpha *= 0.5
            u_new = np.clip(u + alpha * direction, 0.0, 1.0)
            v_new, g_new, *_ = objh(u_new)
        s = u_new - u
        y = g_new - g
        if s.dot(y) > 1e-12:
            rho = 1.0 / s.dot(y)
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        u, v, g = u_new, v_new, g_new
        history.append({"val": v, "u": u.copy(), "pg": np.linalg.norm(g, ord=np.inf), "alpha": alpha})
        if abs(v - history[-1]["val"]) < obj_change_tol:
            break
    return v, u, history


def unit_box_lm(pinit, residual_func, max_iter=10, damping=1e-3, **kwargs):
    p = np.clip(np.asarray(pinit, dtype=float), 0.0, 1.0)
    for it in range(max_iter):
        r, J, _ = residual_func(p)
        r = np.asarray(r, dtype=float)
        if J is None:
            J = _finite_difference_jacobian(lambda u: residual_func(u)[0], p)
        H = J.T @ J + damping * np.eye(J.shape[1])
        dp = np.linalg.solve(H, -J.T @ r)
        p = np.clip(p + dp, 0.0, 1.0)
    return p


# Re-export from subpackages
from .utils import *  # noqa: E402, F403
from .optim import *  # noqa: E402, F403
from .objectives import *  # noqa: E402, F403
