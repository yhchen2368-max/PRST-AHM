"""Self-consistency checks for PRSTCore.solvers.incomp.transport (explicit_transport),
ported from MRST's twophaseUpwFE.m / initTransport.m.

The strongest, discretization-independent property a conservative upwind
finite-volume scheme must satisfy is exact mass conservation: the total
water volume in the domain can only change by exactly the net injected
volume (before any injected water reaches a producer -- after breakthrough,
some of what balances the books leaves through the producer instead, which
is checked separately by tracking produced water via each cell's outflow
weighted by its upstream fractional flow).
"""

from __future__ import annotations

import numpy as np

from PRSTCore.gridprocessing import cart_grid, compute_geometry
from PRSTCore.solvers.incomp import compute_trans, incomp_tpfa, explicit_transport, linear_fluid, corey_fluid


def _setup_1d_waterflood(n=30, L=300.0, rate=1e-4):
    G = compute_geometry(cart_grid([n], [L]))
    nc = G["cells"]["num"]
    rock = {"perm": np.full(nc, 1e-13), "poro": np.full(nc, 0.2)}
    T = compute_trans(G, rock)
    fluid_p = {"mu": 1e-3}  # pressure solve uses unit-mobility-equivalent fluid

    wells = [
        {"cells": np.array([0]), "WI": np.array([1e-11]), "type": "rate", "val": rate},
        {"cells": np.array([nc - 1]), "WI": np.array([1e-11]), "type": "bhp", "val": 1.0e5},
    ]
    state = incomp_tpfa(G, T, fluid_p, wells=wells)
    state["s"] = np.zeros(nc)
    return G, rock, state, wells


def test_mass_conservation_before_breakthrough():
    G, rock, state, wells = _setup_1d_waterflood()
    fluid = linear_fluid(mu_w=1e-3, mu_o=1e-3)
    pv = G["cells"]["volumes"] * rock["poro"]

    # Pick tf well short of the time needed for the front to cross the domain
    # (total pv / rate is the time to sweep the whole domain once).
    total_pv = float(np.sum(pv))
    rate = 1e-4
    tf = 0.2 * total_pv / rate

    new_state = explicit_transport(G, state, rock, fluid, tf, wells=wells, dt=tf / 50)

    water_before = float(np.sum(state["s"] * pv))
    water_after = float(np.sum(new_state["s"] * pv))
    injected = rate * tf

    assert np.all(new_state["s"] >= 0.0) and np.all(new_state["s"] <= 1.0)
    # No breakthrough yet: producer cell saturation should still be ~0.
    assert np.isclose(new_state["s"][-1], 0.0, atol=1e-10)
    assert np.isclose(water_after - water_before, injected, rtol=1e-8)


def test_mass_conservation_with_production_after_breakthrough():
    G, rock, state, wells = _setup_1d_waterflood(n=15, L=150.0, rate=2e-4)
    fluid = linear_fluid(mu_w=1e-3, mu_o=1e-3)
    pv = G["cells"]["volumes"] * rock["poro"]
    total_pv = float(np.sum(pv))
    rate = 2e-4

    # Run long enough to sweep past breakthrough.
    tf = 1.5 * total_pv / rate
    dt = tf / 400

    s = state.copy()
    produced_water = 0.0
    t = 0.0
    while t < tf:
        h = min(dt, tf - t)
        step_state = explicit_transport(G, s, rock, fluid, h, wells=wells, dt=h)
        # Water produced this step: producer's outflow rate * upstream fractional flow.
        mu_w, mu_o = fluid.mu
        krw, kro = fluid.relperm(s["s"])
        f_w = (krw / mu_w) / (krw / mu_w + kro / mu_o)
        prod_flux = -float(np.sum(step_state["wellSol"][1]["flux"]))  # positive = produced volume
        produced_water += prod_flux * f_w[-1] * h
        s = step_state
        t += h

    water_before = 0.0
    water_after = float(np.sum(s["s"] * pv))
    injected = rate * tf

    assert np.any(s["s"][-1] > 0.0), "expected breakthrough within tf"
    # Global mass balance: injected = stored in domain + produced.
    assert np.isclose(injected, (water_after - water_before) + produced_water, rtol=5e-2)


def test_corey_fluid_relperm_shape():
    fluid = corey_fluid(mu_w=1e-3, mu_o=2e-3, nw=2, no=2, swc=0.1, sor=0.15)
    krw0, kro0 = fluid.relperm(np.array([0.1]))
    krw1, kro1 = fluid.relperm(np.array([0.85]))
    assert np.isclose(krw0[0], 0.0, atol=1e-12)
    assert np.isclose(kro1[0], 0.0, atol=1e-12)
    assert np.isclose(krw1[0], 1.0, atol=1e-8)
    assert np.isclose(kro0[0], 1.0, atol=1e-8)
