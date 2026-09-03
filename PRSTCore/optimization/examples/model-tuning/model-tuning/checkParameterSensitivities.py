"""Runnable translation of MRST checkParameterSensitivities.m."""

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

from PRSTCore.optimization import evaluate_match, unit_box_bfgs

from _shared import make_mismatch, make_synthetic_case, setup_with_parameters, run_case


def run_check_parameter_sensitivities(epsilons=(1e-6, 1e-7, 1e-8, 1e-9)):
    """Compare adjoint/finite-difference directional derivatives."""
    print("[checkParameterSensitivities] setup synthetic model")
    case = make_synthetic_case(nx=10, ny=10, nsteps=8, seed=3)
    _, states_ref = run_case("sens_ref", case, case.schedule)

    setup, params, p0 = setup_with_parameters(case, case.schedule, include_relperm=True)
    mismatch_fn = make_mismatch(
        {
            "WaterRateWeight": (300.0 / 86400.0) ** -1,
            "OilRateWeight": (300.0 / 86400.0) ** -1,
            "BHPWeight": (500.0e5) ** -1,
        }
    )

    v0, g0, *_ = evaluate_match(p0, mismatch_fn, setup, params, states_ref, Gradient="AdjointAD")
    print(f"[checkParameterSensitivities] baseline objective={v0:.6e}")

    rng = np.random.default_rng(0)
    du = rng.random(p0.size)
    du = du / np.linalg.norm(du)

    print("Directional gradient comparison:")
    print("epsilon      finite-diff           adjoint-proj        rel.err")
    for eps in epsilons:
        vp, *_ = evaluate_match(p0 + eps * du, mismatch_fn, setup, params, states_ref, Gradient="none")
        g_fd = (vp - v0) / eps
        g_ad = float(g0 @ du)
        rel = abs(g_fd - g_ad) / max(abs(g_ad), 1e-12)
        print(f"{eps:8.1e}  {g_fd: .6e}  {g_ad: .6e}  {rel: .3e}")

    # Keep one optimization pass to mirror the MRST workflow.
    v_opt, p_opt, hist = unit_box_bfgs(
        p0,
        lambda p: evaluate_match(p, mismatch_fn, setup, params, states_ref, Gradient="AdjointAD"),
        maximize=False,
        max_it=6,
        grad_tol=1e-4,
        obj_change_tol=1e-7,
    )
    print(f"[checkParameterSensitivities] quick optimize objective={v_opt:.6e}, iters={len(hist)}")

    return {"objective0": float(v0), "objective_opt": float(v_opt), "p_opt": p_opt}


if __name__ == "__main__":
    run_check_parameter_sensitivities()
