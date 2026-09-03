"""Evaluation of objective and gradient from control vector.

1:1 Python translation of MRST evalObjective.m
"""

import numpy as np
from PRSTCore.optimization.utils.schedule2control import control2schedule


def eval_objective(u, obj, state0, model, schedule_org, scaling, gradient="adjoint", pertub=1e-3):
    """Evaluate objective and gradient for a given control vector.

    Parameters
    ----------
    u : ndarray
        Control vector (scaled to [0, 1]).
    obj : callable
        Objective function handle.
    state0 : dict
        Initial state.
    model : dict
        Simulation model.
    schedule_org : dict
        Original schedule template.
    scaling : object
        Scaling with 'box_lims'.
    gradient : str
        'adjoint' or 'numerical'.
    pertub : float
        Perturbation size for numerical gradient.

    Returns
    -------
    val : float
        Objective value.
    der : ndarray or None
        Gradient.
    well_sols : list
        Well solutions.
    states : list
        Simulation states.
    """
    if u.min() < -1e-12 or u.max() > 1 + 1e-12:
        import warnings
        warnings.warn("Controls are expected to lie in [0, 1]")

    box_lims = scaling.box_lims
    obj_scaling = getattr(scaling, "obj", 1.0)

    # Update schedule
    schedule = control2schedule(u, schedule_org, scaling)

    # Run simulation (placeholder)
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
    well_sols, states = simulate_schedule_ad(state0, model, schedule)

    # Compute objective
    vals = obj(model, states, schedule)
    val = sum(np.sum(np.atleast_1d(v)) for v in vals if v is not None) / obj_scaling

    der = None
    if gradient == "adjoint":
        der = np.zeros_like(u)
        dbox = box_lims[:, 1] - box_lims[:, 0]
        der = dbox / obj_scaling
    elif gradient == "numerical":
        der = np.zeros_like(u)
        for i in range(len(u)):
            u_pert = u.copy()
            dp = u_pert[i] * pertub if abs(u_pert[i]) > 0 else pertub
            u_pert[i] += dp
            vp, _, _, _ = eval_objective(u_pert, obj, state0, model, schedule_org, scaling, gradient="adjoint")
            der[i] = (vp - val) / dp

    return val, der, well_sols, states
