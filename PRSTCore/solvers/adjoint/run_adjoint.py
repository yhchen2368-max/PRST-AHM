"""Run adjoint simulation based on forward simulation results.

1:1 Python translation of MRST solvers/adjoint/runAdjoint.m
"""

import numpy as np


def run_adjoint(sim_res, G, S, W, rock, fluid, schedule, controls,
                objective_function, verbose=False, verbose_level=0):
    """Run adjoint simulation backward through schedule steps.

    Parameters
    ----------
    sim_res : list of dict
        Forward simulation results.
    G, S, W, rock, fluid : dict/list
        Standard MRST structures.
    schedule : list of dict
        Schedule.
    controls : dict
        Controls structure.
    objective_function : callable
        Objective function handle.
    verbose : bool
        Verbose output.

    Returns
    -------
    list of dict
        adj_res - adjoint results array.
    """
    from .update_wells import update_wells
    from .solve_adjoint_transport_system import solve_adjoint_transport_system
    from .solve_adjoint_pressure_system import solve_adjoint_pressure_system

    num_steps = len(schedule)
    adj_res = [{"timeInterval": None, "resSol": {}} for _ in range(num_steps + 1)]
    obj = objective_function(G, S, W, rock, fluid, sim_res, schedule, controls)

    if verbose:
        print("\n******* Starting adjoint simulation *******")

    for k in range(num_steps - 1, -1, -1):
        if verbose:
            print(f"Time step {k + 1:3d} of {num_steps:3d},   ", end="")

        W_updated = update_wells(W, schedule[k])

        if verbose:
            print("Transport:", end=" ")
        adj_res = solve_adjoint_transport_system(
            G, S, W_updated, rock, fluid, sim_res, adj_res, obj,
        )

        if verbose:
            print("Pressure:", end=" ")
        adj_res = solve_adjoint_pressure_system(
            G, S, W_updated, rock, fluid, sim_res, adj_res, obj,
        )

        if verbose:
            print()

    return adj_res
