"""Shared helpers for runnable model-tuning examples.

These utilities keep the translated scripts close to MRST flow while using
lightweight synthetic setups that run with the current Python implementation.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# ---- bootstrap: ensure the repo root is on sys.path ----
_SHARED_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SHARED_DIR.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PRSTCore.ad_core.simulators.sim_runner import (
    get_packed_simulator_output,
    pack_simulation_problem,
    simulate_packed_problem,
)
from PRSTCore.ad_core.upscale import upscale_model_tpfa, upscale_schedule, upscale_state
from PRSTCore.ad_core.utils import simple_schedule
from PRSTCore.coarsegrid import compress_partition, partition_ui
from PRSTCore.network_models.utils import make_random_training
from PRSTCore.optimization import evaluate_match
from PRSTCore.optimization.objectives import match_observed_ow
from PRSTCore.optimization.utils.parameters import (
    add_parameter,
    get_scaled_parameter_vector,
    update_setup_from_scaled_parameters,
)


@dataclass
class CaseData:
    model: dict
    state0: dict
    schedule: dict


def make_synthetic_case(nx: int = 6, ny: int = 5, nsteps: int = 12, seed: int = 0) -> CaseData:
    """Build a small oil-water synthetic case used by model tuning examples."""
    n_cells = nx * ny
    rng = np.random.default_rng(seed)
    centroids = np.column_stack(
        [
            np.tile(np.linspace(0.0, float(nx - 1), nx), ny),
            np.repeat(np.linspace(0.0, float(ny - 1), ny), nx),
            np.zeros(n_cells),
        ]
    )
    poro = 0.18 + 0.06 * rng.random(n_cells)
    perm = 80.0 + 350.0 * rng.random(n_cells)

    wells = []
    corners = [0, nx - 1, n_cells - nx, n_cells - 1]
    for i, c in enumerate(corners):
        if i < 2:
            wells.append(
                {
                    "cells": [int(c) + 1],
                    "type": "rate",
                    "val": 0.12,
                    "sign": 1,
                    "status": True,
                    "name": f"I{i+1}",
                }
            )
        else:
            wells.append(
                {
                    "cells": [int(c) + 1],
                    "type": "bhp",
                    "val": 190.0e5,
                    "sign": -1,
                    "status": True,
                    "name": f"P{i-1}",
                    "lims": {"bhp": 190.0e5},
                }
            )

    controls = [{"W": copy.deepcopy(wells)} for _ in range(nsteps)]
    schedule = simple_schedule(np.full(nsteps, 30.0 * 86400.0), controls)
    state0 = {
        "pressure": 220.0e5 * np.ones(n_cells),
        "s": np.column_stack([0.25 * np.ones(n_cells), 0.75 * np.ones(n_cells)]),
    }
    model = {
        "G": {"cells": {"num": n_cells, "centroids": centroids}, "faces": {"num": n_cells * 3}},
        "rock": {"poro": poro, "perm": perm.reshape(-1, 1)},
        "operators": {"T": np.ones(n_cells * 3), "pv": poro.copy()},
    }
    return CaseData(model=model, state0=state0, schedule=schedule)


def make_training_schedule(case: CaseData, r_scale: float = 0.25, bhp_scale: float = 0.05) -> dict:
    setup = {
        "state0": case.state0,
        "model": case.model,
        "schedule": case.schedule,
        "name": "synthetic-training",
    }
    return make_random_training(setup, r_scale=r_scale, bhp_scale=bhp_scale, shutin=False)["schedule"]


def run_case(name: str, case: CaseData, schedule: dict):
    problem = pack_simulation_problem(case.state0, case.model, schedule, name)
    simulate_packed_problem(problem)
    return get_packed_simulator_output(problem)


def make_coarse_setup(case: CaseData, schedule: dict, part: tuple[int, int, int] = (3, 3, 1)) -> dict:
    q = partition_ui(case.model["G"], list(part))
    q = compress_partition(q)
    c_model = upscale_model_tpfa(case.model, q, trans_from_rock=False)
    c_state0 = upscale_state(c_model, case.model, case.state0)
    c_schedule = upscale_schedule(c_model, schedule)
    return {"model": c_model, "state0": c_state0, "schedule": c_schedule}


def add_tuning_parameters(setup: dict, include_relperm: bool = False):
    model = setup["model"]
    model["porevolume"] = np.asarray(model["operators"]["pv"], dtype=float)
    model["conntrans"] = np.ones_like(model["porevolume"])
    model["transmissibility"] = np.asarray(model["operators"]["T"], dtype=float)

    config = [
        ("porevolume", "linear", [0.2, 4.0]),
        ("conntrans", "log", [1e-3, 1e2]),
        ("transmissibility", "log", [1e-3, 1e2]),
    ]
    if include_relperm:
        n = model["G"]["cells"]["num"]
        for key in ("swl", "swcr", "swu", "sowcr", "krw", "kro"):
            model[key] = np.full(n, 0.5)
        config.extend(
            [
                ("swl", "linear", [0.0, 0.3]),
                ("swcr", "linear", [0.0, 0.4]),
                ("swu", "linear", [0.7, 1.0]),
                ("sowcr", "linear", [0.0, 0.4]),
                ("krw", "linear", [0.5, 1.5]),
                ("kro", "linear", [0.5, 1.5]),
            ]
        )

    params = []
    for name, scaling, rel_lim in config:
        params = add_parameter(params, setup, name=name, scaling=scaling, relative_limits=rel_lim)
    return params


def make_mismatch(weighting: dict | None = None) -> Callable:
    weights = {
        "WaterRateWeight": 86400.0 / 10000.0,
        "OilRateWeight": 86400.0 / 10000.0,
        "BHPWeight": 1.0 / (500.0 * 1e5),
    }
    if weighting is not None:
        weights.update(weighting)

    def mismatch(model, states, schedule, states_ref, compute_partials, tstep, state):
        return match_observed_ow(
            model,
            states,
            schedule,
            states_ref,
            compute_partials=compute_partials,
            tstep=tstep,
            state=state,
            weighting=weights,
            from_states=False,
        )

    return mismatch


def evaluate_objective(u: np.ndarray, obj_fn: Callable, setup: dict, params: list, states_ref: list):
    return evaluate_match(u, obj_fn, setup, params, states_ref, Gradient="AdjointAD")


def setup_with_parameters(case: CaseData, schedule: dict, include_relperm: bool = False):
    coarse_setup = make_coarse_setup(case, schedule)
    params = add_tuning_parameters(coarse_setup, include_relperm=include_relperm)
    p0 = get_scaled_parameter_vector(coarse_setup, params)
    return coarse_setup, params, p0


def apply_parameters(setup: dict, params: list, pvec: np.ndarray):
    return update_setup_from_scaled_parameters(setup, params, pvec)
