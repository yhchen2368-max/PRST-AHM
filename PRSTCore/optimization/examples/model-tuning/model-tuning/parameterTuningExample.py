"""Runnable translation of MRST parameterTuningExample.m."""

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
    make_synthetic_case,
    make_training_schedule,
    run_case,
    setup_with_parameters,
    make_mismatch,
    apply_parameters,
)


def run_parameter_tuning_example(max_it: int = 20):
    """Tune a coarse model against synthetic fine-scale reference data."""
    print("[parameterTuningExample] setup fine-scale case")
    case = make_synthetic_case(nx=8, ny=6, nsteps=14, seed=11)

    train_schedule = make_training_schedule(case, r_scale=0.35, bhp_scale=0.06)
    print("[parameterTuningExample] simulate reference fine model")
    ws_ref, states_ref = run_case("parameter_tuning_ref", case, train_schedule)

    print("[parameterTuningExample] setup coarse model + parameters")
    setup, params, p0 = setup_with_parameters(case, train_schedule, include_relperm=True)
    mismatch_fn = make_mismatch(
        {
            "WaterRateWeight": (5.0 / 86400.0) ** -1,
            "OilRateWeight": (5.0 / 86400.0) ** -1,
            "BHPWeight": (5.0e5) ** -1,
        }
    )

    def objh(pvec):
        return evaluate_match(pvec, mismatch_fn, setup, params, states_ref, Gradient="AdjointAD")

    print("[parameterTuningExample] optimize (BFGS)")
    v_opt, p_opt, history = unit_box_bfgs(
        p0,
        objh,
        maximize=False,
        obj_change_tol=1e-7,
        grad_tol=1e-4,
        max_it=max_it,
        lbfgs_strategy="dynamic",
        lbfgs_num=10,
    )

    tuned = apply_parameters(setup, params, p_opt)
    ws_tuned, states_tuned = simulate_schedule_ad(tuned["state0"], tuned["model"], tuned["schedule"])

    ws_coarse0, _ = simulate_schedule_ad(setup["state0"], setup["model"], setup["schedule"])

    plot_well_sols(
        [ws_ref, ws_coarse0, ws_tuned],
        [
            np.cumsum(train_schedule["step"]["val"]),
            np.cumsum(setup["schedule"]["step"]["val"]),
            np.cumsum(tuned["schedule"]["step"]["val"]),
        ],
        dataset_names=["reference", "initial coarse", "tuned coarse"],
        field="qOs",
        selected_wells=[0, 1],
    )

    print(f"[parameterTuningExample] done: objective={v_opt:.6e}, iters={len(history)}")
    return {
        "objective": float(v_opt),
        "history": history,
        "p_opt": p_opt,
        "states_ref": states_ref,
        "states_tuned": states_tuned,
    }


if __name__ == "__main__":
    run_parameter_tuning_example()
