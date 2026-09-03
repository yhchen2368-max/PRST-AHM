"""Compute numerical gradient via finite differences.

1:1 Python translation of MRST solvers/adjoint/computeNumericalGradient.m
"""

import numpy as np


def compute_numerical_gradient(sim_res, G, S, W, rock, fluid, schedule,
                                controls, objective_function):
    """Compute numerical gradient by perturbing controls.

    Parameters
    ----------
    sim_res, G, S, W, rock, fluid, schedule, controls : standard structures
    objective_function : callable

    Returns
    -------
    ndarray
        Numerical gradient (numControlWells x numSteps).
    """
    from .update_schedule import update_schedule
    from .run_schedule import run_schedule

    epsilon = 1e-5
    num_control_wells = len(controls["well"])
    num_steps = len(schedule)

    obj = objective_function(G, S, W, rock, fluid, sim_res)
    val_init = obj.get("val", 0.0) if isinstance(obj, dict) else float(obj)

    u_init = np.array([w["values"] for w in controls["well"]]).T.ravel()
    dim_u = len(u_init)
    epsilon = epsilon * np.linalg.norm(u_init)

    values = np.zeros(dim_u)
    for k in range(dim_u):
        e_k = np.zeros(dim_u)
        e_k[k] = 1.0
        u_cur = u_init + epsilon * e_k
        ctrl_new = _update_controls_vec(controls, u_cur, num_control_wells)
        sched_new = update_schedule(ctrl_new, schedule)
        sim_new = run_schedule(sim_res[0]["resSol"], G, S, W, rock, fluid, sched_new)
        obj_new = objective_function(G, S, W, rock, fluid, sim_new)
        values[k] = obj_new.get("val", 0.0) if isinstance(obj_new, dict) else float(obj_new)

    num_grad = (values - val_init) / epsilon
    return num_grad.reshape(num_control_wells, num_steps)


def _update_controls_vec(controls, u, num_control_wells):
    """Update controls from vector."""
    U = u.reshape(-1, num_control_wells)
    new_ctrl = {"well": [], "linEqConst": controls.get("linEqConst"),
                "linIneqConst": controls.get("linIneqConst")}
    for k, w in enumerate(controls["well"]):
        new_ctrl["well"].append({**w, "values": U[:, k]})
    return new_ctrl
