"""Runnable translation of MRST eggCoarseModelAdjointCalibration.m."""

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
from PRSTCore.optimization import evaluate_match, unit_box_bfgs, unit_box_lm

from _shared import apply_parameters, make_mismatch, make_synthetic_case, make_training_schedule, run_case, setup_with_parameters


def run_egg_coarse_model_adjoint_calibration(max_it: int = 20):
    """Calibrate coarse synthetic EGG-like model with BFGS and LM."""
    print("[eggCoarseModelAdjointCalibration] setup and simulate reference")
    case = make_synthetic_case(nx=14, ny=10, nsteps=18, seed=7)
    train_schedule = make_training_schedule(case, r_scale=0.4, bhp_scale=0.03)
    ws_ref, states_ref = run_case("egg_ref", case, train_schedule)

    setup, params, p0 = setup_with_parameters(case, train_schedule, include_relperm=True)
    mismatch_sum = make_mismatch({"WaterRateWeight": 1.0 / (150.0 / 86400.0), "OilRateWeight": 1.0 / (80.0 / 86400.0)})

    print("[eggCoarseModelAdjointCalibration] optimize with BFGS")
    v_bfgs, p_bfgs, h_bfgs = unit_box_bfgs(
        p0,
        lambda p: evaluate_match(p, mismatch_sum, setup, params, states_ref, Gradient="AdjointAD"),
        maximize=False,
        obj_change_tol=1e-5,
        max_it=max_it,
    )

    print("[eggCoarseModelAdjointCalibration] optimize with LM")
    def residual_func(pvec):
        vals = evaluate_match(pvec, mismatch_sum, setup, params, states_ref, Gradient="none")[0]
        r = np.atleast_1d(vals)
        J = np.eye(pvec.size)[: r.size, :]
        return r, J, {}

    p_lm = unit_box_lm(p0, residual_func, max_iter=min(max_it, 12))

    tuned_bfgs = apply_parameters(setup, params, p_bfgs)
    tuned_lm = apply_parameters(setup, params, p_lm)

    ws_coarse0, _ = simulate_schedule_ad(setup["state0"], setup["model"], setup["schedule"])
    ws_bfgs, _ = simulate_schedule_ad(tuned_bfgs["state0"], tuned_bfgs["model"], tuned_bfgs["schedule"])
    ws_lm, _ = simulate_schedule_ad(tuned_lm["state0"], tuned_lm["model"], tuned_lm["schedule"])

    t = np.cumsum(train_schedule["step"]["val"])
    plot_well_sols(
        [ws_ref, ws_coarse0, ws_bfgs, ws_lm],
        [t, t, t, t],
        dataset_names=["reference", "initial", "tuned (BFGS)", "tuned (LM)"],
        field="qWs",
        selected_wells=[0, 1],
    )

    print(f"[eggCoarseModelAdjointCalibration] done: BFGS={v_bfgs:.6e}, iters={len(h_bfgs)}")
    return {"bfgs_objective": float(v_bfgs), "p_bfgs": p_bfgs, "p_lm": p_lm}


if __name__ == "__main__":
    run_egg_coarse_model_adjoint_calibration()
