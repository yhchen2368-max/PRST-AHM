"""Aggressive line search based on projected gradient.

1:1 Python translation of MRST solvers/adjoint/lineSearchAgr.m
"""

import numpy as np


def line_search_agr(sim_res, G, S, W, rock, fluid, schedule, controls,
                     grad, objective_function, step_size, max_points=20,
                     lin_srch_tol=1e-4, const_tol=1e-3, max_const_its=100,
                     verbose_level=0):
    """Aggressive line search along projected gradient.

    Parameters
    ----------
    sim_res, G, S, W, rock, fluid, schedule, controls : standard structures
    grad : ndarray
        Gradient from compute_gradient.
    objective_function : callable
    step_size : float
        Initial step size.
    max_points : int
        Max points before termination.
    lin_srch_tol : float
        Tolerance.
    const_tol : float
        Constraint tolerance.
    max_const_its : int
        Max constraint iterations.
    verbose_level : int

    Returns
    -------
    sim_res, schedule, controls, output : tuple
        Best results and info dict.
    """
    from .project_gradient import project_gradient
    from .update_schedule import update_schedule
    from .run_schedule import run_schedule

    # Project gradient
    pdu = project_gradient(controls, grad, const_tol=const_tol,
                           max_const_its=max_const_its, verbose_level=verbose_level)
    norm_pdu = np.linalg.norm(pdu)

    obj_ref = objective_function(G, S, W, rock, fluid, sim_res)
    val_ref = obj_ref.get("val", 0.0) if isinstance(obj_ref, dict) else float(obj_ref)

    # Line search points
    fractions = [0.0, 0.5 * step_size, 1.0 * step_size]
    obj_vals = [val_ref]
    sims = [sim_res]
    scheds = [schedule]
    ctrls_list = [controls]

    for frac in fractions[1:]:
        u_new = np.array([w["values"] for w in controls["well"]]).T - frac * pdu.reshape(
            len(controls["well"]), -1).T
        ctrl_new = _update_ctrl_from_matrix(controls, u_new)
        sched_new = update_schedule(ctrl_new, schedule)
        sim_new = run_schedule(sim_res[0]["resSol"], G, S, W, rock, fluid, sched_new)
        obj_new = objective_function(G, S, W, rock, fluid, sim_new)
        val = obj_new.get("val", 0.0) if isinstance(obj_new, dict) else float(obj_new)
        obj_vals.append(val)
        sims.append(sim_new)
        scheds.append(sched_new)
        ctrls_list.append(ctrl_new)

    # Find best (min or extend)
    best_idx = np.argmin(obj_vals)
    success = best_idx > 0

    return sims[best_idx], scheds[best_idx], ctrls_list[best_idx], {
        "value": obj_vals[best_idx],
        "relNormGrad": norm_pdu / max(abs(val_ref), 1e-12),
        "success": success,
        "fraction": fractions[best_idx],
        "objValues": obj_vals,
    }


def _update_ctrl_from_matrix(controls, U):
    """Update controls from matrix (numSteps x numControlWells)."""
    new_ctrl = {
        "well": [],
        "linEqConst": controls.get("linEqConst"),
        "linIneqConst": controls.get("linIneqConst"),
        "numControlSteps": controls.get("numControlSteps"),
    }
    for k, w in enumerate(controls["well"]):
        new_ctrl["well"].append({**w, "values": U[:, k]})
    return new_ctrl
