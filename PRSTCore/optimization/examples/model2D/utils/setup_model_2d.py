"""Optimize a 2D model with well controls.

1:1 Python translation of MRST examples/model2D/optimizeModel2D.m (placeholder).
"""

import numpy as np


def setup_model_2d():
    """Create a simple 2D reservoir model for optimization testing.

    Returns
    -------
    dict
        Problem dictionary with model, schedule, state0.
    """
    nx, ny = 20, 10
    n_cells = nx * ny

    G = {
        "cells": {
            "num": n_cells,
            "centroids": np.column_stack([
                np.tile(np.linspace(0, 2000, nx), ny),
                np.repeat(np.linspace(0, 1000, ny), nx),
                np.zeros(n_cells),
            ]),
        }
    }

    poro = 0.2 * np.ones(n_cells)
    perm = 100 * np.ones(n_cells)

    rock = {"poro": poro, "perm": perm.reshape(-1, 1)}

    # One injector, one producer
    W = [
        {"cells": [0], "type": "rate", "val": 0.1, "sign": 1, "status": True,
         "WI": 1.0, "name": "I1", "dir": "z", "r": 0.1},
        {"cells": [n_cells - 1], "type": "bhp", "val": 200e5, "sign": -1,
         "status": True, "WI": 1.0, "name": "P1", "dir": "z", "r": 0.1,
         "lims": {"bhp": 200e5}},
    ]

    n_steps = 10
    dt = np.full(n_steps, 365 * 86400 / 10)  # ~36.5 days per step
    schedule = {
        "step": {"val": dt, "control": np.zeros(n_steps, dtype=int)},
        "control": [{"W": W}],
    }

    state0 = {"pressure": 250e5 * np.ones(n_cells),
              "s": np.column_stack([0.2 * np.ones(n_cells), 0.8 * np.ones(n_cells)])}

    model = {"G": G, "rock": rock, "operators": {"T": np.ones(n_cells * 3), "pv": poro}}

    return {"model": model, "schedule": schedule, "state0": state0}
