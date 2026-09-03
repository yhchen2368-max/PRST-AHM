"""Simple BHP optimization example.

1:1 Python translation of MRST solvers/adjoint/examples/simpleBHPOpt.m
"""

import numpy as np


def simple_bhp_opt():
    """Simple BHP optimization using adjoint gradients.

    Returns
    -------
    dict
        Optimization results.
    """
    from PRSTCore.solvers.adjoint import (
        init_schedule, init_controls, run_schedule, run_adjoint,
        compute_gradient, optimize_objective,
    )
    from PRSTCore.solvers.adjoint.objectives import simple_npv

    # Setup simple 1D grid
    nc = 10
    G = {
        "cells": {
            "num": nc,
            "volumes": np.ones(nc) * 1000,
            "faces": np.column_stack([np.arange(1, nc + 2), np.zeros(nc + 1, dtype=int)]),
            "facePos": np.arange(nc + 1),
        },
        "faces": {
            "num": nc + 1,
            "neighbors": np.column_stack([np.arange(nc), np.arange(1, nc + 1)]).tolist(),
        },
    }
    rock = {"poro": np.full(nc, 0.2), "perm": np.full((nc, 1), 100e-15)}
    fluid = {
        "krw": lambda s: s**2, "kro": lambda s: (1 - s)**2,
        "dkrw": lambda s: 2 * s, "dkro": lambda s: -2 * (1 - s),
        "muw": 1e-3, "muo": 5e-3,
    }
    S = {"type": "mixed"}

    # Two wells
    W_init = [
        {"cells": [1], "type": "bhp", "val": 300e5, "sign": 1, "WI": 1e-12,
         "name": "I1", "S": {"BI": np.eye(1), "C": np.eye(1), "D": np.eye(1),
                              "RHS": {"f": np.zeros(1), "h": 0.0}, "sizeB": [1]}},
        {"cells": [nc], "type": "bhp", "val": 200e5, "sign": -1, "WI": 1e-12,
         "name": "P1", "S": {"BI": np.eye(1), "C": np.eye(1), "D": np.eye(1),
                              "RHS": {"f": np.zeros(1), "h": 0.0}, "sizeB": [1]}},
    ]

    schedule = init_schedule(W_init, num_steps=5, total_time=365 * 86400)
    controls = init_controls(schedule, controllable_wells=[1],
                              bhp_min_max=[150e5, 350e5], num_control_steps=5)

    res_sol_init = {
        "s": np.full(nc, 0.2),
        "pressure": np.linspace(300e5, 200e5, nc),
        "flux": np.zeros(G["faces"]["num"]),
        "wellSol": [],
    }

    sim_res, schedule_opt, controls_opt, output = optimize_objective(
        G, S, W_init, rock, fluid, res_sol_init, schedule, controls,
        simple_npv, max_it=5, verbose_level=1,
    )

    return {"sim_res": sim_res, "output": output}
