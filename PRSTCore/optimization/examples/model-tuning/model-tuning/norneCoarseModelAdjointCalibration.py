"""Runnable translation of MRST norneCoarseModelAdjointCalibration.m."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
_REPO_ROOT = _THIS.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PRSTCore.ad_core.plotting import plot_well_sols
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.optimization import evaluate_match, unit_box_bfgs

from _shared import (
    apply_parameters,
    make_mismatch,
    make_synthetic_case,
    make_training_schedule,
    run_case,
    setup_with_parameters,
)


def run_norne_coarse_model_adjoint_calibration(max_it: int = 30):
    """Calibrate a coarse model from synthetic Norne-like data."""
    print("[norneCoarseModelAdjointCalibration] setup reference")
    case = make_synthetic_case(nx=12, ny=8, nsteps=24, seed=21)
    ws_ref, states_ref = run_case("norne_ref", case, case.schedule)

    train_steps = case.schedule["step"]["val"].size // 2
    train_schedule = {
        "step": {
            "val": case.schedule["step"]["val"][:train_steps],
            "control": case.schedule["step"]["control"][:train_steps],
        },
        "control": case.schedule["control"][:train_steps],
    }

    setup, params, p0 = setup_with_parameters(case, train_schedule, include_relperm=True)
    mismatch_fn = make_mismatch()

    print("[norneCoarseModelAdjointCalibration] optimize")
    v, p_opt, history = unit_box_bfgs(
        p0,
        lambda p: evaluate_match(p, mismatch_fn, setup, params, states_ref[:train_steps], Gradient="AdjointAD"),
        maximize=False,
        obj_change_tol=1e-4,
        max_it=max_it,
        lbfgs_strategy="dynamic",
        lbfgs_num=5,
    )

    setup_opt = apply_parameters(setup, params, p_opt)
    setup_opt["schedule"] = case.schedule
    ws_opt, _ = simulate_schedule_ad(setup_opt["state0"], setup_opt["model"], setup_opt["schedule"])

    ws_init, _ = simulate_schedule_ad(setup["state0"], setup["model"], setup["schedule"])

    plot_well_sols(
        [ws_ref[:train_steps], ws_init, ws_opt[:train_steps]],
        [
            np.cumsum(train_schedule["step"]["val"]),
            np.cumsum(setup["schedule"]["step"]["val"]),
            np.cumsum(train_schedule["step"]["val"]),
        ],
        dataset_names=["reference", "initial", "optimized"],
        field="qOs",
        selected_wells=[0, 1],
    )

    print(f"[norneCoarseModelAdjointCalibration] done: objective={v:.6e}, iters={len(history)}")
    return {"objective": float(v), "history": history, "p_opt": p_opt}


if __name__ == "__main__":
    run_norne_coarse_model_adjoint_calibration()
