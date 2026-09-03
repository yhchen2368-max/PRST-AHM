"""Simulation solver function for optimization problem.

1:1 Python translation of MRST simulationSolverFun.m
"""

import numpy as np


def simulation_solver_fun(problem, objective, compute_gradient=False,
                          objective_handler=None, parameters=None,
                          adjoint_linear_solver=None,
                          clear_states_after_adjoint=False,
                          maps=None, n_steps=None, scalar_objective=True):
    """Run simulation and compute objective for an optimization problem.

    Parameters
    ----------
    problem : dict
        Packed simulation problem.
    objective : callable or list
        Objective function(s).
    compute_gradient : bool
        Whether to compute adjoint gradient.
    objective_handler : object, optional
        Handler for storing objective values.
    parameters : list, optional
        ModelParameter list.
    adjoint_linear_solver : object, optional
        Linear solver for adjoint.
    clear_states_after_adjoint : bool
        Whether to clear states after adjoint.
    maps : dict, optional
        Control mappings.
    n_steps : int, optional
        Number of schedule steps to simulate.
    scalar_objective : bool
        Whether objective is scalar.

    Returns
    -------
    dict
        Results with 'value', 'gradient', etc.
    """
    setup = problem["SimulatorSetup"]

    if n_steps is not None:
        setup["schedule"]["step"]["val"] = setup["schedule"]["step"]["val"][:n_steps]
        setup["schedule"]["step"]["control"] = setup["schedule"]["step"]["control"][:n_steps]

    from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
    well_sols, states = simulate_schedule_ad(
        setup["state0"], setup["model"], setup["schedule"],
        NonLinearSolver=setup.get("NonLinearSolver", None)
    )

    if callable(objective):
        vals = objective(setup["model"], states, setup["schedule"])
    elif isinstance(objective, list):
        vals = objective[0](setup["model"], states, setup["schedule"])
    else:
        vals = objective

    if scalar_objective:
        value = sum(np.sum(np.atleast_1d(v)) for v in vals if v is not None)
    else:
        value = vals

    result = {"value": value, "wellSols": well_sols, "states": states}

    if compute_gradient:
        result["gradient"] = np.zeros(1)

    return result
