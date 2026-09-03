"""Unit tests for SimpleWell/FacilityModel (WellModel.m/FacilityModel.m ports),
independent of the SPE9/Egg end-to-end regression that already bit-for-bit
validated this code as an in-place extraction from GenericBlackOilModel.
"""

from __future__ import annotations

import numpy as np

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.models.facility_model import FacilityModel


def _identity_component_mass_fn(rhoS_phases):
    def fn(c, qph):
        return [q * rho for q, rho in zip(qph, rhoS_phases)]
    return fn


def test_single_perforation_producer_rate_matches_peaceman_formula():
    nc, nw = 3, 1
    nvar = 2 * nc + 3 * nw
    p = SparseADI.variable(np.array([200.0, 150.0, 100.0]), nvar, 0)
    sw = SparseADI.variable(np.array([0.3, 0.3, 0.3]), nvar, nc)  # unused placeholder slot
    bhp = SparseADI.variable(np.array([50.0]), nvar, 2 * nc + 2 * nw)

    WI, mu_w, mu_o = 1e-11, 1e-3, 2e-3
    lamW = SparseADI.constant(np.full(nc, 0.5 / mu_w), nvar)
    lamO = SparseADI.constant(np.full(nc, 0.5 / mu_o), nvar)
    rhoWS, rhoOS = 1000.0, 850.0

    well = {"cells": [0], "WI": [WI], "type": "bhp", "val": 50.0, "sign": -1}
    fm = FacilityModel(well_cells_fn=lambda w: w["cells"])
    state = {}

    (srcW, srcO), (surfW, surfO), perf_phase, perf_comp = fm.compute_well_contributions(
        wells=[well], state=state, p=p, bhp=bhp, lam_phases=[lamW, lamO], rhoS_phases=[rhoWS, rhoOS],
        component_mass_fn=_identity_component_mass_fn([rhoWS, rhoOS]),
        nc=nc, nw=nw, nvar=nvar, n_component_phases=2,
    )

    # Producer: drawdown = p[0] - bhp = 150; perforation volumetric rate =
    # -WI*mobility*drawdown (Peaceman; MRST's sign convention makes a
    # producer's completion rate negative), no crossflow/injection mixing
    # since this is a single all-producing perforation.
    drawdown = 200.0 - 50.0
    expected_qW = -WI * (0.5 / mu_w) * drawdown
    expected_qO = -WI * (0.5 / mu_o) * drawdown
    assert np.isclose(perf_phase[0][0, 0], expected_qW, rtol=1e-10)
    assert np.isclose(perf_phase[0][0, 1], expected_qO, rtol=1e-10)
    assert np.isclose(surfW.val[0], expected_qW, rtol=1e-10)  # component_mass/rhoS == volumetric rate here
    assert np.isclose(srcW.val[0], expected_qW * rhoWS, rtol=1e-10)  # reservoir-cell source, mass units


def test_shut_perforation_contributes_nothing():
    nc, nw = 2, 1
    nvar = 2 * nc + 3 * nw
    p = SparseADI.variable(np.array([200.0, 200.0]), nvar, 0)
    bhp = SparseADI.variable(np.array([50.0]), nvar, 2 * nc + 2 * nw)
    lamW = SparseADI.constant(np.full(nc, 1.0), nvar)
    lamO = SparseADI.constant(np.full(nc, 1.0), nvar)

    well = {"cells": [0, 1], "WI": [1e-11, 1e-11], "cstatus": [True, False], "type": "bhp", "val": 50.0, "sign": -1}
    fm = FacilityModel(well_cells_fn=lambda w: w["cells"])
    (srcW, srcO), (surfW, surfO), perf_phase, perf_comp = fm.compute_well_contributions(
        wells=[well], state={}, p=p, bhp=bhp, lam_phases=[lamW, lamO], rhoS_phases=[1000.0, 850.0],
        component_mass_fn=_identity_component_mass_fn([1000.0, 850.0]),
        nc=nc, nw=nw, nvar=nvar, n_component_phases=2,
    )
    assert perf_phase[0].shape[0] == 1  # only the open perforation contributes
    assert np.isclose(srcW.val[1], 0.0)  # cell 1 (shut perf) gets no source


def test_multi_well_surface_rates_indexed_independently():
    nc, nw = 2, 2
    nvar = 2 * nc + 3 * nw
    p = SparseADI.variable(np.array([200.0, 200.0]), nvar, 0)
    bhp = SparseADI.variable(np.array([50.0, 30.0]), nvar, 2 * nc + 2 * nw)
    lamW = SparseADI.constant(np.full(nc, 1.0), nvar)
    lamO = SparseADI.constant(np.full(nc, 1.0), nvar)

    wells = [
        {"cells": [0], "WI": [1e-11], "type": "bhp", "val": 50.0, "sign": -1},
        {"cells": [1], "WI": [2e-11], "type": "bhp", "val": 30.0, "sign": -1},
    ]
    fm = FacilityModel(well_cells_fn=lambda w: w["cells"])
    (srcW, srcO), (surfW, surfO), *_ = fm.compute_well_contributions(
        wells=wells, state={}, p=p, bhp=bhp, lam_phases=[lamW, lamO], rhoS_phases=[1000.0, 850.0],
        component_mass_fn=_identity_component_mass_fn([1000.0, 850.0]),
        nc=nc, nw=nw, nvar=nvar, n_component_phases=2,
    )
    # Both are producers (negative rate, MRST convention); well 1 has a
    # bigger WI and drawdown -> strictly larger-magnitude (more negative) rate.
    assert surfW.val[1] < surfW.val[0] < 0.0


def test_control_equations_bhp_and_rate():
    nw = 2
    nvar = 3 * nw
    qws = SparseADI.variable(np.array([5.0, 7.0]), nvar, 0)
    qos = SparseADI.variable(np.array([1.0, 2.0]), nvar, nw)
    bhp = SparseADI.variable(np.array([100.0, 200.0]), nvar, 2 * nw)
    wells = [
        {"type": "bhp", "val": 120.0},
        {"type": "rate", "val": 10.0},
    ]
    closure = FacilityModel.compute_control_equations(
        wells, qs_phases={"w": qws, "o": qos}, bhp=bhp, phase_order=["w", "o"]
    )
    assert np.isclose(closure[0].val[0], (100.0 - 120.0) / (86400.0 * 1.0e5))
    assert np.isclose(closure[1].val[0], 7.0 + 2.0 - 10.0)
