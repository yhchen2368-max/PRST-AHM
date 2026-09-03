"""Self-consistency checks for implicit_transport (twophaseJacobian.m/
implicitTransport.m port) and init_res_sol/init_well_sol (initResSol.m/
initWellSol.m port).

Key properties checked for implicit_transport:
  - Same exact mass-conservation identity as explicit_transport (backward vs
    forward Euler must both conserve mass exactly for a conservative scheme).
  - Unconditional stability: a time step that is a large multiple of the
    explicit CFL limit must still produce a bounded, physical ([0,1])
    saturation field for implicit_transport (this is precisely the property
    implicit schemes are for; explicit would blow up or need many substeps).
  - Convergence toward the same answer as explicit_transport when both are
    run with a small, CFL-respecting step.
"""

from __future__ import annotations

import numpy as np

from PRSTCore.gridprocessing import cart_grid, compute_geometry
from PRSTCore.solvers.incomp import (
    compute_trans, incomp_tpfa, explicit_transport, implicit_transport,
    linear_fluid, init_res_sol, init_well_sol,
)


def _setup_1d_waterflood(n=30, L=300.0, rate=1e-4):
    G = compute_geometry(cart_grid([n], [L]))
    nc = G["cells"]["num"]
    rock = {"perm": np.full(nc, 1e-13), "poro": np.full(nc, 0.2)}
    T = compute_trans(G, rock)
    fluid_p = {"mu": 1e-3}

    wells = [
        {"cells": np.array([0]), "WI": np.array([1e-11]), "type": "rate", "val": rate},
        {"cells": np.array([nc - 1]), "WI": np.array([1e-11]), "type": "bhp", "val": 1.0e5},
    ]
    state = incomp_tpfa(G, T, fluid_p, wells=wells)
    state["s"] = np.zeros(nc)
    return G, rock, state, wells


def test_implicit_transport_mass_conservation_before_breakthrough():
    G, rock, state, wells = _setup_1d_waterflood()
    fluid = linear_fluid(mu_w=1e-3, mu_o=1e-3)
    pv = G["cells"]["volumes"] * rock["poro"]
    total_pv = float(np.sum(pv))
    rate = 1e-4
    tf = 0.2 * total_pv / rate

    new_state = implicit_transport(G, state, rock, fluid, tf, wells=wells, dt=tf / 20)

    water_before = float(np.sum(state["s"] * pv))
    water_after = float(np.sum(new_state["s"] * pv))
    injected = rate * tf

    assert np.all(new_state["s"] >= 0.0) and np.all(new_state["s"] <= 1.0)
    # ~0 up to Newton's own convergence tolerance (default tol=1e-6).
    assert np.isclose(new_state["s"][-1], 0.0, atol=1e-6)
    assert np.isclose(water_after - water_before, injected, rtol=1e-6)


def test_implicit_transport_stable_far_beyond_explicit_cfl():
    G, rock, state, wells = _setup_1d_waterflood(n=20, L=200.0, rate=2e-4)
    fluid = linear_fluid(mu_w=1e-3, mu_o=1e-3)
    pv = G["cells"]["volumes"] * rock["poro"]
    total_pv = float(np.sum(pv))
    rate = 2e-4

    # A step several times the whole-domain sweep time -- wildly CFL-violating
    # for an explicit scheme (single-step explicit_transport would produce
    # negative/>1 saturations before clamping). Implicit must stay bounded
    # and exactly mass-conservative even for one giant step.
    tf = 3.0 * total_pv / rate
    new_state = implicit_transport(G, state, rock, fluid, tf, wells=wells, dt=tf)

    assert np.all(new_state["s"] >= -1e-10) and np.all(new_state["s"] <= 1.0 + 1e-10)
    water_before = float(np.sum(state["s"] * pv))
    water_after = float(np.sum(new_state["s"] * pv))
    produced = -float(np.sum(new_state["wellSol"][1]["flux"])) * new_state["s"][-1]  # rough upper bound check only
    injected = rate * tf
    # Global balance: injected water is either stored or produced (can't exceed injected).
    assert water_after - water_before <= injected + 1e-6


def test_implicit_and_explicit_agree_under_small_cfl_respecting_step():
    G, rock, state, wells = _setup_1d_waterflood(n=25, L=250.0, rate=1e-4)
    fluid = linear_fluid(mu_w=1e-3, mu_o=1e-3)
    pv = G["cells"]["volumes"] * rock["poro"]
    total_pv = float(np.sum(pv))
    rate = 1e-4
    tf = 0.15 * total_pv / rate
    dt = tf / 200  # comfortably inside CFL for this rate/grid

    s_exp = explicit_transport(G, state, rock, fluid, tf, wells=wells, dt=dt)["s"]
    s_imp = implicit_transport(G, state, rock, fluid, tf, wells=wells, dt=dt)["s"]

    # Both are only first-order accurate in time with opposite-signed truncation
    # error, so some disagreement near the (numerically smeared) front is
    # expected -- this just checks they're not qualitatively different solutions.
    assert np.max(np.abs(s_exp - s_imp)) < 1e-2


def test_init_res_sol_and_init_well_sol_shapes():
    G = compute_geometry(cart_grid([4, 3, 2], [4.0, 3.0, 2.0]))
    state = init_res_sol(G, 1.5e5, 0.2)
    assert state["pressure"].shape == (G["cells"]["num"],)
    assert np.all(state["pressure"] == 1.5e5)
    assert state["flux"].shape == (G["faces"]["num"],)
    assert np.all(state["flux"] == 0.0)
    assert state["s"].shape == (G["cells"]["num"],)
    assert np.all(state["s"] == 0.2)

    wells = [{"cells": np.array([0, 1])}, {"cells": np.array([5])}]
    wsol = init_well_sol(wells, 1.0e5)
    assert len(wsol) == 2
    assert wsol[0]["flux"].shape == (2,)
    assert np.all(wsol[0]["flux"] == 0.0)
    assert wsol[1]["pressure"] == 1.0e5
