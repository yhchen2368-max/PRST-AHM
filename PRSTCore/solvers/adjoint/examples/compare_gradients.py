"""Compare adjoint and numerical gradients.

1:1 Python translation of MRST solvers/adjoint/examples/compareGradients.m
"""

import numpy as np


def compare_gradients():
    """Compare adjoint and numerical gradients for validation."""
    from PRSTCore.solvers.adjoint import (
        init_schedule, init_controls, run_schedule, run_adjoint,
        compute_gradient, compute_numerical_gradient,
    )
    from PRSTCore.solvers.adjoint.objectives import simple_npv

    nc = 10
    G = {
        "cells": {"num": nc, "volumes": np.ones(nc) * 1000,
                  "faces": np.column_stack([np.arange(1, nc + 2), np.zeros(nc + 1, dtype=int)]),
                  "facePos": np.arange(nc + 1)},
        "faces": {"num": nc + 1,
                  "neighbors": np.column_stack([np.arange(nc), np.arange(1, nc + 1)]).tolist()},
    }
    rock = {"poro": np.full(nc, 0.2), "perm": np.full((nc, 1), 100e-15)}
    fluid = {"krw": lambda s: s**2, "kro": lambda s: (1 - s)**2,
             "dkrw": lambda s: 2 * s, "dkro": lambda s: -2 * (1 - s),
             "muw": 1e-3, "muo": 5e-3}
    S = {"type": "mixed"}

    W_init = [
        {"cells": [1], "type": "rate", "val": 0.01, "sign": 1, "WI": 1e-12,
         "name": "I1", "S": {"BI": np.eye(1), "C": np.eye(1), "D": np.eye(1),
                              "RHS": {"f": np.zeros(1), "h": 0.0}, "sizeB": [1]}},
        {"cells": [nc], "type": "bhp", "val": 200e5, "sign": -1, "WI": 1e-12,
         "name": "P1", "S": {"BI": np.eye(1), "C": np.eye(1), "D": np.eye(1),
                              "RHS": {"f": np.zeros(1), "h": 0.0}, "sizeB": [1]}},
    ]

    schedule = init_schedule(W_init, num_steps=3, total_time=100 * 86400)
    controls = init_controls(schedule, controllable_wells=[0],
                              rate_min_max=[0.001, 0.1], num_control_steps=3)
    res_sol_init = {"s": np.full(nc, 0.2), "pressure": np.linspace(300e5, 200e5, nc),
                    "flux": np.zeros(G["faces"]["num"]), "wellSol": []}

    sim_res = run_schedule(res_sol_init, G, S, W_init, rock, fluid, schedule)
    adj_res = run_adjoint(sim_res, G, S, W_init, rock, fluid, schedule, controls,
                           simple_npv)
    adj_grad = np.column_stack(compute_gradient(W_init, adj_res, schedule, controls))
    num_grad = compute_numerical_gradient(sim_res, G, S, W_init, rock, fluid, schedule,
                                           controls, simple_npv)

    print("Adjoint gradient:\n", adj_grad)
    print("Numerical gradient:\n", num_grad)
    print("Relative difference:", np.linalg.norm(adj_grad - num_grad) /
          max(np.linalg.norm(adj_grad), 1e-12))

    return {"adjoint": adj_grad, "numerical": num_grad}
