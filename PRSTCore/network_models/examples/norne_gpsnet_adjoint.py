"""Norne: calibration of a GPSNet model with adjoint gradients.

1:1 Python translation of MRST
modules/network-models/examples/norneGPSNetAdjoint.m

Demonstrates setting up a GPSNet type reduced network model for a
geostatistical realization of the Norne field model, and calibrating
it using the BFGS algorithm with adjoint gradients.
"""

import numpy as np
from PRSTCore.ad_core.plotting import plot_well_sols
from PRSTCore.ad_core.simulators.sim_runner import (
    get_packed_simulator_output,
    pack_simulation_problem,
    simulate_packed_problem,
)
from PRSTCore.ad_core.utils import simple_schedule
from PRSTCore.network_models.utils import make_random_training
from PRSTCore.network_models import Network, GPSNet
from PRSTCore.network_models.utils import gps_net_simulation_setup
from PRSTCore.optimization import evaluate_match, unit_box_bfgs
from PRSTCore.optimization.objectives import match_observed_ow
from PRSTCore.optimization.utils.parameters import add_parameter


def make_norne_case_gpsnet():
    """Create a simplified Norne test case for GPSNet calibration."""
    n_cells = 30
    n_wells = 11
    np.random.seed(42)

    G = {
        "cells": {
            "num": n_cells,
            "centroids": np.column_stack([
                np.tile(np.linspace(0, 6000, 6), 5),
                np.repeat(np.linspace(0, 5000, 5), 6),
                np.zeros(n_cells),
            ]),
        }
    }
    poro = 0.15 + 0.1 * np.random.rand(n_cells)
    perm = 100 + 500 * np.random.rand(n_cells)
    rock = {"poro": poro, "perm": perm.reshape(-1, 1)}

    well_cells = np.linspace(0, n_cells - 1, n_wells, dtype=int)
    W = []
    for i in range(n_wells):
        if i < 6:
            w = {"cells": [int(well_cells[i]) + 1], "type": "rate", "val": 0.1,
                 "sign": 1, "status": True, "WI": 1.0,
                 "name": f"I{i+1}", "r": 0.1, "compi": [1, 0]}
        else:
            w = {"cells": [int(well_cells[i]) + 1], "type": "bhp",
                 "val": 200e5, "sign": -1, "status": True, "WI": 1.0,
                 "name": f"P{i-5}", "r": 0.1, "lims": {"bhp": 200e5}}
        W.append(w)

    n_steps = 10
    dt = np.full(n_steps, 30 * 86400)
    schedule = {
        "step": {"val": dt, "control": np.zeros(n_steps, dtype=int)},
        "control": [{"W": W}],
    }
    state0 = {"pressure": 250e5 * np.ones(n_cells),
              "s": np.column_stack([0.2 * np.ones(n_cells), 0.8 * np.ones(n_cells)])}
    model = {"G": G, "rock": rock, "fluid": {},
             "operators": {"T": np.ones(n_cells * 3), "pv": poro}}

    return {"model": model, "schedule": schedule, "state0": state0, "W": W,
            "G": G, "name": "norne_simple_wo"}


def run_norne_gpsnet_adjoint():
    """Run Norne GPSNet adjoint calibration example."""
    print("=" * 60)
    print("Norne GPSNet Adjoint Calibration")
    print("=" * 60)

    # Setup reference model
    print("\n[1/6] Setting up reference model...")
    case = make_norne_case_gpsnet()
    pred_case = dict(case)
    pred_problem = pack_simulation_problem(
        pred_case["state0"], pred_case["model"],
        pred_case["schedule"], "norne_pred_gpsnet",
    )
    simulate_packed_problem(pred_problem)
    pred_well_sols, pred_states = get_packed_simulator_output(pred_problem)
    pred_model = pred_case["model"]
    pred_schedule = pred_case["schedule"]
    Wpred = pred_schedule["control"][0]["W"]

    # Random training schedule
    train_case = make_random_training(
        {"state0": case["state0"], "model": case["model"],
         "schedule": case["schedule"], "name": "norne"},
        r_scale=0.25, bhp_scale=0.05, shutin=False,
    )
    train_problem = pack_simulation_problem(
        train_case["state0"], train_case["model"],
        train_case["schedule"], "norne_train_gpsnet",
    )
    simulate_packed_problem(train_problem)
    train_well_sols, train_states = get_packed_simulator_output(train_problem)
    train_model = train_case["model"]
    train_schedule = train_case["schedule"]

    print(f"  Reference: {len(pred_well_sols)} steps, {len(train_well_sols)} training")

    # Create network
    print("\n[2/6] Creating injector-to-producer network...")
    Wnw = [dict(w) for w in Wpred]
    for w in Wnw:
        cells = w["cells"]
        w["cells"] = [cells[len(cells) // 2]]

    injectors = list(range(6))
    producers = list(range(6, 11))
    ntwrk = Network(Wnw, pred_model["G"], type="injectors_to_producers",
                    injectors=injectors, producers=producers)
    print(f"  Nodes: {ntwrk.num_nodes}, Edges: {ntwrk.num_edges}")

    # Create GPSNet
    print("\n[3/6] Building GPSNet model...")
    gps_net = GPSNet(pred_model, ntwrk, Wpred, nc=10)
    print(f"  Model cells: {gps_net.model['G']['cells']['num']}")

    # Setup training and prediction
    print("\n[4/6] Setting up parameters...")
    train_setup = gps_net_simulation_setup(gps_net, train_schedule)
    pred_setup = gps_net_simulation_setup(gps_net, pred_schedule)

    cell_edge_no, cell_ix = gps_net.get_mapping("cells")
    face_edge_no, face_ix = gps_net.get_mapping("faces")

    train_setup["model"]["porevolume"] = gps_net.model["operators"]["pv"]
    train_setup["model"]["conntrans"] = np.ones(len(gps_net.W))
    train_setup["model"]["transmissibility"] = gps_net.model["operators"]["T"]
    pred_setup["model"]["porevolume"] = gps_net.model["operators"]["pv"]
    pred_setup["model"]["conntrans"] = np.ones(len(gps_net.W))
    pred_setup["model"]["transmissibility"] = gps_net.model["operators"]["T"]

    config = [
        ["porevolume", 1, "linear", None, cell_edge_no, cell_ix, [0.001, 4]],
        ["conntrans", 1, "log", None, None, None, [0.001, 100]],
        ["transmissibility", 1, "log", None, face_edge_no, face_ix, [0.001, 100]],
    ]

    train_prms = []
    pred_prms = []
    for name, inc, sc, bl, lp, ss, rl in config:
        if inc == 0:
            continue
        train_prms = add_parameter(train_prms, train_setup, name=name,
                                   scaling=sc, box_lims=bl, lumping=lp,
                                   subset=ss, relative_limits=rl)
        pred_prms = add_parameter(pred_prms, pred_setup, name=name,
                                  scaling=sc, box_lims=bl, lumping=lp,
                                  subset=ss, relative_limits=rl)

    n_params = sum(p.n_param for p in train_prms)
    print(f"  Training parameters: {[p.name for p in train_prms]}")
    print(f"  Total: {n_params}")

    # Mismatch function
    print("\n[5/6] Calibrating with BFGS (max 5 iterations)...")
    weighting = {
        "WaterRateWeight": 86400 / 10000,
        "OilRateWeight": 86400 / 20000,
        "BHPWeight": 1.0 / (500e5),
    }

    def mismatch_fn(model, states, schedule, states_ref, tt, tstep, state):
        return match_observed_ow(model, states, schedule, states_ref,
                                 compute_partials=tt, tstep=tstep,
                                 weighting=weighting, state=state,
                                 from_states=False)

    pinit = gps_net.get_scaled_parameter_vector(train_setup, train_prms,
                                                 connscale=0.5)

    objh = lambda p: evaluate_match(p, mismatch_fn, train_setup, train_prms,
                                     train_states)

    v, popt, history = unit_box_bfgs(pinit, objh, maximize=False,
                                      obj_change_tol=1e-8, grad_tol=1e-5,
                                      max_it=5, lbfgs_strategy="dynamic",
                                      lbfgs_num=5, output_hessian=True)

    print(f"  Final mismatch: {v:.6e}, Iterations: {len(history)}")

    # Evaluate
    print("\n[6/6] Evaluating calibrated model...")
    misfit_p, _, well_sol_p, _ = evaluate_match(
        popt, mismatch_fn, pred_setup, pred_prms, pred_states, Gradient="none")
    misfit_t, _, well_sol_t, _ = evaluate_match(
        popt, mismatch_fn, train_setup, train_prms, train_states, Gradient="none")
    print(f"  Prediction mismatch: {misfit_p:.6e}")
    print(f"  Training mismatch:   {misfit_t:.6e}")

    return {
        "popt": popt, "history": history,
        "misfit_predict": misfit_p, "misfit_train": misfit_t,
        "gps_net": gps_net, "network": ntwrk,
    }


if __name__ == "__main__":
    results = run_norne_gpsnet_adjoint()
    print("\n" + "=" * 60)
    print("Calibration complete.")
    print(f"Optimal params min/max: {results['popt'].min():.4f}, {results['popt'].max():.4f}")
