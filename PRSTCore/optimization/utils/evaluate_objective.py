"""Evaluate objective (and gradient) for a control vector.

This is a standalone utility corresponding to the pattern used in MRST:
    f = @(u) evaluateObjective(u, obj, optSetup, optParams, ...);

Although not a standalone MRST function, this utility wraps the logic of:
1. Unpacking the control vector into a schedule
2. Running simulation
3. Computing objective value and gradient

It is also available at PRSTCore.solvers.adjoint.utils.eval_objective
"""

import numpy as np


def evaluate_objective(u, obj, setup, parameters,
                        enforce_bounds=True, obj_scaling=1.0,
                        nonlinear_solver=None, gradient="adjoint",
                        pertub=1e-3, **kwargs):
    """Evaluate objective and gradient for a control vector.

    Parameters
    ----------
    u : ndarray
        Control vector (scaled to [0,1]).
    obj : callable
        Objective function handle: f(model, states, schedule, ...).
    setup : dict
        Simulation setup with state0, model, schedule.
    parameters : list
        List of ModelParameter objects.
    enforce_bounds : bool
        Clip u to [0, 1].
    obj_scaling : float
        Scaling factor for objective.
    nonlinear_solver : object, optional
        Nonlinear solver.
    gradient : str
        'adjoint' or 'numerical'.
    pertub : float
        Perturbation size for numerical gradient.

    Returns
    -------
    val : float
        Scaled objective value.
    grad : ndarray or None
        Gradient (if computed).
    well_sols : list
        Well solutions.
    states : list
        Reservoir states.
    """
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

    if enforce_bounds:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)

    from PRSTCore.optimization.utils.parameters import update_setup_from_scaled_parameters

    # Apply parameter values to setup using the same model/operator hooks as
    # MRST-style model calibration utilities.
    setup_new = update_setup_from_scaled_parameters(setup, parameters, u)

    # Simulate
    well_sols, states = simulate_schedule_ad(
        setup_new["state0"], setup_new["model"], setup_new["schedule"],
        NonLinearSolver=nonlinear_solver,
    )

    # Evaluate objective
    vals = obj(setup_new["model"], states, setup_new["schedule"], False, [], [])
    if isinstance(vals, list):
        val = np.sum([np.sum(np.atleast_1d(v)) for v in vals if v is not None])
    else:
        val = float(vals)
    val = val / obj_scaling

    # Gradient
    grad = None
    if gradient != "none":
        grad = np.zeros_like(u)
        eps = pertub
        for i in range(len(u)):
            u_pert = u.copy()
            dp = eps * max(1.0, abs(u[i]))
            u_pert[i] += dp
            vp, _, _, _ = evaluate_objective(
                u_pert, obj, setup, parameters,
                enforce_bounds=False, obj_scaling=obj_scaling,
                nonlinear_solver=nonlinear_solver, gradient="none",
            )
            grad[i] = (vp - val) / dp

    return val, grad, well_sols, states


def _set_param_value(setup, param, pval):
    """Backward-compatible shim for older callers."""
    from PRSTCore.optimization.utils.parameters import update_setup_from_scaled_parameters

    tmp = update_setup_from_scaled_parameters(setup, [param], param.scale(pval))
    setup.update(tmp)
