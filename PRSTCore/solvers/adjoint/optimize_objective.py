"""Run full optimization process using adjoint-based line search.

1:1 Python translation of MRST solvers/adjoint/optimizeObjective.m
"""

import numpy as np


def optimize_objective(G, S, W, rock, fluid, res_sol_init, schedule, controls,
                        objective_function, grad_tol=1e-3, obj_change_tol=1e-4,
                        const_tol=1e-10, step_size=-1.0, max_it=20,
                        max_const_it=100, verbose_level=0, plot_progress=True):
    """Run optimization using adjoint gradients and line search.

    Parameters
    ----------
    G, S, W, rock, fluid : dict/list
        Standard structures.
    res_sol_init : dict
        Initial reservoir solution.
    schedule : list of dict
    controls : dict
    objective_function : callable
    grad_tol : float
        Gradient norm stopping criterion.
    obj_change_tol : float
        Objective change stopping criterion.
    step_size : float
        Initial step size (-1 for auto).
    max_it : int
        Max iterations.
    verbose_level : int
    plot_progress : bool

    Returns
    -------
    sim_res, schedule, controls, output : tuple
    """
    from .run_schedule import run_schedule
    from .run_adjoint import run_adjoint
    from .compute_gradient import compute_gradient
    from .line_search_agr import line_search_agr

    if step_size <= 0:
        step_size = 1.0

    obj_values = []
    iteration_num = 0
    norm_grad = np.inf
    obj_change = np.inf

    sim_res = run_schedule(res_sol_init, G, S, W, rock, fluid, schedule)

    while (norm_grad >= grad_tol and obj_change >= obj_change_tol
           and iteration_num < max_it):
        iteration_num += 1

        if verbose_level >= 0:
            print(f"\n********** STARTING ITERATION {iteration_num:3d} ****************")
            print(f"Current stepsize: {step_size:.5f}")

        if iteration_num == 1:
            obj = objective_function(G, S, W, rock, fluid, sim_res)
            obj_value = obj.get("val", 0.0) if isinstance(obj, dict) else float(obj)
            obj_values.append(obj_value)
            if verbose_level >= 0:
                print(f"Initial function value: {obj_value:.3f}")

        # Adjoint solve
        adj_res = run_adjoint(sim_res, G, S, W, rock, fluid, schedule, controls,
                               objective_function)

        # Compute gradient
        grad = compute_gradient(W, adj_res, schedule, controls)
        grad_mat = np.column_stack(grad) if isinstance(grad, list) else grad
        norm_grad = np.linalg.norm(grad_mat, np.inf)

        if verbose_level >= 0:
            print(f"Gradient norm: {norm_grad:.6e}")

        # Line search
        sim_res_new, schedule_new, controls_new, ls_out = line_search_agr(
            sim_res, G, S, W, rock, fluid, schedule, controls, grad_mat,
            objective_function, step_size,
        )

        obj_new = ls_out["value"]
        obj_change = abs(obj_new - obj_values[-1]) / max(abs(obj_values[-1]), 1e-12)
        obj_values.append(obj_new)

        if ls_out["success"]:
            sim_res = sim_res_new
            schedule = schedule_new
            controls = controls_new

        if verbose_level >= 0:
            print(f"Iter {iteration_num}: obj={obj_new:.6f}, change={obj_change:.3e}")

    output = {"objValues": obj_values, "iterations": iteration_num}
    return sim_res, schedule, controls, output
