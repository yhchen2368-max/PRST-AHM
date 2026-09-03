"""Compute ensemble simulation objective.

1:1 Python translation of MRST computeEnsembleSimulationObjective.m

This is a placeholder; ensemble optimization requires MPI/parallel support.
"""

import numpy as np


def compute_ensemble_simulation_objective(setup_fn, output_path="", member_ix=None,
                                          schedule=None, objective=None,
                                          compute_gradient=True,
                                          clear_states_after_adjoint=True):
    """Compute ensemble simulation objectives (placeholder)."""
    results = []
    for ix in member_ix:
        problem = setup_fn(ix)
        val = objective(problem["SimulatorSetup"]["model"],
                        [], problem["SimulatorSetup"]["schedule"])
        results.append({"value": np.sum(val) if isinstance(val, list) else val})
    return results
