"""Runnable translation of MRST valveOptimizationSPE10.m."""

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

from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.ad_core.utils import simple_schedule
from PRSTCore.optimization import evaluate_match, unit_box_bfgs
from PRSTCore.optimization.objectives import npv_ow
from PRSTCore.optimization.utils.parameters import add_parameter

from _shared import make_synthetic_case


def run_valve_optimization_spe10(max_it: int = 15):
    """Optimize synthetic valve multipliers (conntrans) against NPV objective."""
    print("[valveOptimizationSPE10] setup synthetic SPE10-like case")
    case = make_synthetic_case(nx=22, ny=10, nsteps=16, seed=12)

    model = case.model
    state0 = case.state0
    schedule = case.schedule
    model["conntrans"] = np.ones(model["G"]["cells"]["num"])

    # Parameterize connection transmissibility multipliers as valve controls.
    setup = {"model": model, "state0": state0, "schedule": schedule}
    params = add_parameter([], setup, name="conntrans", scaling="log", box_lims=[1e-4, 1.0])
    p0 = np.full(params[0].n_param, 0.9)

    def npv_objective(mod, states, sch, _obs, compute_partials, tstep, state):
        vals = npv_ow(
            mod,
            states,
            sch,
            oil_price=50.0,
            water_production_cost=5.0,
            water_injection_cost=3.0,
            discount_factor=0.1,
            compute_partials=compute_partials,
            tstep=tstep,
            state=state,
            from_states=False,
        )
        return vals

    # evaluate_match minimizes (-sum(vals)) internally; maximize NPV => minimize that value.
    print("[valveOptimizationSPE10] optimize conntrans multipliers")
    v_opt, p_opt, history = unit_box_bfgs(
        p0,
        lambda p: evaluate_match(p, npv_objective, setup, params, [], Gradient="AdjointAD"),
        maximize=False,
        obj_change_tol=1e-5,
        grad_tol=1e-4,
        max_it=max_it,
        lbfgs_strategy="dynamic",
        lbfgs_num=5,
    )

    setup_opt = dict(setup)
    setup_opt["model"] = dict(model)
    setup_opt["model"]["conntrans"] = params[0].unscale(p_opt)
    ws_opt, _ = simulate_schedule_ad(setup_opt["state0"], setup_opt["model"], setup_opt["schedule"])
    ws_base, _ = simulate_schedule_ad(state0, model, schedule)

    print(f"[valveOptimizationSPE10] done: objective={v_opt:.6e}, iters={len(history)}")
    return {
        "objective": float(v_opt),
        "history": history,
        "valve_vector": p_opt,
        "well_sols_base": ws_base,
        "well_sols_opt": ws_opt,
    }


if __name__ == "__main__":
    run_valve_optimization_spe10()
