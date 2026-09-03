"""Norne: calibration of a network model formed from a coarse grid.

This is a 1:1 Python translation of `norneCGNetAdjoint.m` from MRST 2026a.

The example demonstrates how to set up a CGNet model with 3D type graph
topology and corresponding parameters initialized based on a coarse
partition of the full corner-point grid of the Norne field model.

To calibrate the model, we use the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
algorithm. This is an iterative line-search method that gradually improves
an approximation to the Hessian matrix of the mismatch function, obtained
only from adjoint gradients via a generalized secant method.

This example was first introduced in MRST 2021b.
"""

import numpy as np
from PRSTCore.ad_core.plotting import plot_well_sols
from PRSTCore.ad_core.simulators import simulate_schedule_ad
from PRSTCore.ad_core.simulators.sim_runner import (
    get_packed_simulator_output,
    pack_simulation_problem,
    simulate_packed_problem,
)
from PRSTCore.ad_core.upscale import upscale_model_tpfa, upscale_schedule, upscale_state
from PRSTCore.ad_core.utils import simple_schedule
from PRSTCore.coarsegrid import compress_partition, partition_ui
from PRSTCore.network_models.utils import make_random_training
from PRSTCore.optimization import evaluate_match, unit_box_bfgs
from PRSTCore.optimization.objectives import match_observed_ow
from PRSTCore.optimization.utils.parameters import (
    ModelParameter,
    add_parameter,
    get_scaled_parameter_vector,
)


def make_norne_test_case():
    """Create a simplified Norne-like test case.

    This returns a problem dictionary with a fine-grid model, schedule,
    initial state, and a visualization grid for the Norne field.

    In a full MRST environment, this would correspond to:
        TestCase('norne_simple_wo')
    """
    n_cells = 30  # 6x5x1 grid
    n_wells = 11

    # Create a simple rectangular grid representation
    G = {
        "cells": {
            "num": n_cells,
            "centroids": np.column_stack([
                np.tile(np.linspace(0, 6000, 6), 5),
                np.repeat(np.linspace(0, 5000, 5), 6),
                np.zeros(n_cells),
            ]),
        },
        "faces": {"num": n_cells * 3},
    }

    # Rock properties with realistic Norne-like values
    np.random.seed(42)
    poro = 0.15 + 0.1 * np.random.rand(n_cells)
    perm = 100 + 500 * np.random.rand(n_cells)

    rock = {"poro": poro, "perm": perm.reshape(-1, 1)}

    # Define wells: mix of producers (sign=-1) and injectors (sign=1)
    well_cells = np.linspace(0, n_cells - 1, n_wells, dtype=int)
    W = []
    for i in range(n_wells):
        if i < 7:
            # Producers
            w = {
                "cells": [int(well_cells[i]) + 1],
                "type": "bhp",
                "val": 200.0 * 1e5,  # 200 bar in Pa
                "sign": -1,
                "status": True,
                "WI": 1.0,
                "name": f"P{i + 1}",
                "dir": "z",
                "r": 0.1,
                "lims": {"bhp": 200.0 * 1e5},
            }
        else:
            # Injectors
            w = {
                "cells": [int(well_cells[i]) + 1],
                "type": "rate",
                "val": 0.1,
                "sign": 1,
                "status": True,
                "WI": 1.0,
                "name": f"I{i - 6}",
                "dir": "z",
                "r": 0.1,
            }
        W.append(w)

    # Build schedule with 20 steps
    n_steps = 20
    dt = np.full(n_steps, 30 * 86400)  # 30 days per step in seconds
    control = [{"W": W}]
    schedule = {
        "step": {
            "val": dt,
            "control": np.zeros(n_steps, dtype=int),
        },
        "control": control,
    }

    # Initial state
    state0 = {
        "pressure": 250.0 * 1e5 * np.ones(n_cells),
        "s": np.column_stack([0.2 * np.ones(n_cells), 0.8 * np.ones(n_cells)]),
    }

    # Model
    model = {
        "G": G,
        "rock": rock,
        "operators": {"T": np.ones(n_cells * 3), "pv": poro},
        "FlowDiscretization": None,
    }

    return {
        "G": G,
        "model": model,
        "schedule": schedule,
        "state0": state0,
        "W": W,
    }


def run_norne_cgnet_adjoint():
    """Run the Norne CGNet adjoint calibration example.

    Returns
    -------
    dict
        Results dictionary containing optimization history, calibrated
        parameters, and well solutions.
    """
    print("=" * 60)
    print("Norne CGNet Adjoint Calibration")
    print("=" * 60)

    # -------------------------------------------------------------------
    # %% Setup 3D reference model
    # -------------------------------------------------------------------
    print("\n[1/7] Setting up reference model...")
    case = make_norne_test_case()

    # True schedule (reference)
    pred_case = dict(case)
    pred_problem = pack_simulation_problem(
        pred_case["state0"], pred_case["model"], pred_case["schedule"], "norne_pred"
    )
    simulate_packed_problem(pred_problem)
    pred_well_sols, pred_states = get_packed_simulator_output(pred_problem)
    pred_model = pred_case["model"]
    pred_schedule = pred_case["schedule"]

    # Random training schedule (makeRandomTraining equivalent)
    train_case = make_random_training(
        {
            "state0": pred_case["state0"],
            "model": pred_case["model"],
            "schedule": pred_case["schedule"],
            "name": "norne",
        },
        r_scale=0.25,
        bhp_scale=0.05,
        shutin=False,
    )
    train_problem = pack_simulation_problem(
        train_case["state0"], train_case["model"], train_case["schedule"], "norne_train"
    )
    simulate_packed_problem(train_problem)
    train_well_sols, train_states = get_packed_simulator_output(train_problem)
    train_model = train_case["model"]
    train_schedule = train_case["schedule"]

    print(f"  Reference: {len(pred_well_sols)} steps, {len(train_well_sols)} training steps")

    # -------------------------------------------------------------------
    # %% Coarse-scale model and initial state
    # -------------------------------------------------------------------
    print("\n[2/7] Creating coarse model...")
    q = partition_ui(train_model["G"], [5, 6, 1])
    q = compress_partition(q)
    n_coarse = int(np.max(q))
    print(f"  Coarse grid: {n_coarse} cells (from uniform 5x6x1 partition)")

    c_model = upscale_model_tpfa(train_model, q, trans_from_rock=False)
    c_state0 = upscale_state(c_model, train_model, train_case["state0"])

    # -------------------------------------------------------------------
    # %% Specify training schedule and parameters
    # -------------------------------------------------------------------
    print("\n[3/7] Setting up parameters...")
    c_train_sched = upscale_schedule(c_model, train_schedule)
    c_train_problem = {"model": c_model, "schedule": c_train_sched, "state0": c_state0}
    c_pred_sched = upscale_schedule(c_model, pred_schedule)
    c_pred_problem = {"model": c_model, "schedule": c_pred_sched, "state0": c_state0}

    config = [
        # name, include, scaling, boxlims, lumping, subset, relativeLimits
        ["porevolume", 1, "linear", None, None, None, [0.001, 4]],
        ["conntrans", 1, "log", None, None, None, [0.001, 100]],
        ["transmissibility", 1, "log", None, None, None, [0.001, 100]],
    ]

    # Initialize model parameters for easy access
    c_train_problem["model"]["porevolume"] = c_model["operators"]["pv"]
    c_train_problem["model"]["conntrans"] = np.ones_like(c_model["operators"]["pv"])
    c_train_problem["model"]["transmissibility"] = c_model["operators"]["T"]

    train_prms = []
    pred_prms = []
    for name, include, scaling, boxlims, lumping, subset, rel_limits in config:
        if include == 0:
            continue
        train_prms = add_parameter(
            train_prms,
            c_train_problem,
            name=name,
            scaling=scaling,
            box_lims=boxlims,
            lumping=lumping,
            subset=subset,
            relative_limits=rel_limits,
        )
        pred_prms = add_parameter(
            pred_prms,
            c_pred_problem,
            name=name,
            scaling=scaling,
            box_lims=boxlims,
            lumping=lumping,
            subset=subset,
            relative_limits=rel_limits,
        )

    n_params = sum(p.n_param for p in train_prms)
    print(f"  Training parameters: {[p.name for p in train_prms]}")
    print(f"  Total parameters: {n_params}")

    # -------------------------------------------------------------------
    # %% Define the mismatch function
    # -------------------------------------------------------------------
    print("\n[4/7] Defining mismatch function...")
    weighting = {
        "WaterRateWeight": 86400 / 10000,  # day/10000
        "OilRateWeight": 86400 / 20000,  # day/20000
        "BHPWeight": 1.0 / (500 * 1e5),  # 1/(500*barsa)
    }

    def mismatch_fn(model, states, schedule, states_ref, compute_partials, tstep, state):
        return match_observed_ow(
            model,
            states,
            schedule,
            states_ref,
            compute_partials=compute_partials,
            tstep=tstep,
            state=state,
            weighting=weighting,
            from_states=False,
        )

    # -------------------------------------------------------------------
    # %% Model calibration (using BFGS)
    # -------------------------------------------------------------------
    print("\n[5/7] Running BFGS calibration (max 10 iterations)...")

    pinit = get_scaled_parameter_vector(c_train_problem, train_prms)
    print(f"  Initial parameter vector: {pinit.size} values")

    def objh(u):
        """Objective function handle returned (value, gradient, wellSols, states)."""
        result = evaluate_match(
            u,
            mismatch_fn,
            c_train_problem,
            train_prms,
            train_states,
            Gradient="AdjointAD",
        )
        misfit_val, grad, well_sols, states = result
        return misfit_val, grad, well_sols, states

    v, popt, history = unit_box_bfgs(
        pinit,
        objh,
        maximize=False,  # minimize mismatch
        obj_change_tol=1e-8,
        grad_tol=1e-5,
        max_it=10,
        lbfgs_strategy="dynamic",
        lbfgs_num=5,
        output_hessian=True,
        log_plot=False,
    )

    print(f"  Final mismatch: {v:.6e}")
    print(f"  Iterations: {len(history)}")

    # -------------------------------------------------------------------
    # %% Evaluate mismatch over the full simulation schedule
    # -------------------------------------------------------------------
    print("\n[6/7] Evaluating calibrated model...")

    misfit_p, _, well_sol_p, _ = evaluate_match(
        popt,
        mismatch_fn,
        c_pred_problem,
        pred_prms,
        pred_states,
        Gradient="none",
    )
    misfit_t, _, well_sol_t, _ = evaluate_match(
        popt,
        mismatch_fn,
        c_train_problem,
        train_prms,
        train_states,
        Gradient="none",
    )

    print(f"  Prediction mismatch: {misfit_p:.6e}")
    print(f"  Training mismatch:   {misfit_t:.6e}")

    # -------------------------------------------------------------------
    # %% Plot well curves
    # -------------------------------------------------------------------
    print("\n[7/7] Generating plots...")

    # Compare training vs matched
    _ = plot_well_sols(
        [train_well_sols, well_sol_t],
        [np.cumsum(train_schedule["step"]["val"]), np.cumsum(c_train_sched["step"]["val"])],
        dataset_names=["train", "match"],
        field="qWs",
        selected_wells=list(range(min(4, len(train_well_sols[0]) if train_well_sols else 0))),
    )

    # Compare reference vs predicted
    _ = plot_well_sols(
        [pred_well_sols, well_sol_p],
        [np.cumsum(pred_schedule["step"]["val"]), np.cumsum(c_pred_sched["step"]["val"])],
        dataset_names=["reference", "predicted"],
        field="qOs",
        selected_wells=list(range(min(4, len(pred_well_sols[0]) if pred_well_sols else 0))),
    )

    return {
        "popt": popt,
        "history": history,
        "misfit_predict": misfit_p,
        "misfit_train": misfit_t,
        "train_well_sols": train_well_sols,
        "pred_well_sols": pred_well_sols,
        "well_sol_matched": well_sol_t,
        "well_sol_predicted": well_sol_p,
    }


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    results = run_norne_cgnet_adjoint()
    import matplotlib.pyplot as plt

    plt.show()
    print("\n" + "=" * 60)
    print("Calibration complete.")
    print(f"Optimal parameter vector min/max: {results['popt'].min():.4f}, {results['popt'].max():.4f}")
