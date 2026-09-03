"""Smoke tests for the MRST-style PRSTCore upscaling path."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PRSTCore.ad_core.upscale import upscale_model_tpfa, upscale_schedule
from PRSTCore.coarsegrid import compress_partition, generate_coarse_grid, process_partition
from PRSTCore.coarsegrid.utils.coarsen_geometry import coarsen_geometry
from PRSTCore.optimization.utils.parameters import (
    add_parameter,
    get_scaled_parameter_vector,
    update_setup_from_scaled_parameters,
)


def make_2x2_grid():
    return {
        "cells": {
            "num": 4,
            "centroids": np.array(
                [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0], [1.5, 1.5, 0.0]]
            ),
            "volumes": np.ones(4),
            "faces": np.array(
                [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0],
                 [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]]
            ),
            "facePos": np.array([0, 3, 6, 9, 12]),
        },
        "faces": {
            "num": 12,
            "neighbors": np.array(
                [[1, 0], [1, 2], [1, 3],
                 [2, 0], [2, 4], [2, 0],
                 [3, 0], [3, 4], [3, 0],
                 [4, 0], [4, 0], [0, 4]]
            ),
            "areas": np.ones(12),
            "normals": np.tile(np.array([[1.0, 0.0, 0.0]]), (12, 1)),
            "centroids": np.zeros((12, 3)),
        },
        "griddim": 2,
    }


def main():
    G = make_2x2_grid()
    rock = {"poro": np.array([0.2, 0.2, 0.3, 0.3]), "perm": np.ones((4, 1)) * 100.0}
    operators = {
        "N": np.array([[1, 2], [1, 3], [2, 4], [3, 4]]),
        "T": np.array([1.0, 2.0, 3.0, 4.0]),
        "pv": np.array([0.2, 0.2, 0.3, 0.3]),
    }
    model = {"G": G, "rock": rock, "operators": operators}

    split_grid = {"cells": {"num": 4}, "faces": {"num": 2, "neighbors": np.array([[1, 2], [3, 4]])}}
    q = process_partition(split_grid, np.array([1, 1, 1, 1]))
    assert q.tolist() == [1, 1, 2, 2]

    p = compress_partition(process_partition(G, np.array([1, 2, 1, 2])))
    CG = coarsen_geometry(generate_coarse_grid(G, p))
    assert CG["cells"]["num"] == 2
    assert CG["faces"]["neighbors"].tolist() == [[1, 0], [1, 2], [2, 0]]

    coarse = upscale_model_tpfa(model, p, trans_from_rock=False)
    assert coarse["operators"]["N"].tolist() == [[1, 2]]
    assert np.allclose(coarse["operators"]["T"], [5.0])
    assert np.allclose(coarse["operators"]["pv"], [0.5, 0.5])

    schedule = {
        "step": {"val": [1.0], "control": [1]},
        "control": [{"W": [{"name": "W1", "cells": [1, 3], "WI": [10.0, 20.0],
                             "dir": ["z", "z"], "r": [0.1, 0.1], "refDepth": 0.0}]}],
    }
    coarse_schedule = upscale_schedule(coarse, schedule, well_upscale_method="sum")
    well = coarse_schedule["control"][0]["W"][0]
    assert well["cells"] == [1]
    assert np.allclose(well["WI"], [30.0])

    coarse["conntrans"] = np.ones(1)
    setup = {"model": coarse, "schedule": coarse_schedule, "state0": {}}
    params = []
    for name, scaling in (("porevolume", "linear"), ("transmissibility", "log"), ("conntrans", "log")):
        params = add_parameter(params, setup, name=name, scaling=scaling, relative_limits=[0.5, 2.0])
    u = get_scaled_parameter_vector(setup, params)
    updated = update_setup_from_scaled_parameters(setup, params, u)
    assert np.allclose(updated["model"]["operators"]["T"], coarse["operators"]["T"])
    assert np.allclose(updated["schedule"]["control"][0]["W"][0]["WI"], [30.0])

    print("PRSTCore upscale core smoke test passed")


if __name__ == "__main__":
    main()
