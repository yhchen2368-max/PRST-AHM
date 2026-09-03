"""Tests for PhysicalModel/ReservoirModel primitives, ported from MRST's
PhysicalModel.m (updateStateFromIncrement/capProperty) and ReservoirModel.m
(dpMax/dsMax limiting, CNV/MB convergence).

The "matches_legacy_*" tests are the important ones for the
GenericBlackOilModel refactor: they reproduce the *exact* inline formulas
that ``_update_state_mrst_generic``/``_update_state_mrst_generic_ow`` and
``_check_convergence_mrst_generic``/``_check_convergence_mrst_generic_ow``
used before being switched over to call the new shared primitives, and
check bit-for-bit agreement on realistic (always-positive pressure) data --
proving the extraction didn't change SPE1/SPE9/Norne/Egg numerics, which
have no other automated regression coverage.
"""

from __future__ import annotations

import numpy as np

from PRSTCore.ad_core.models.physical_model import PhysicalModel
from PRSTCore.ad_core.models.reservoir_model import ReservoirModel


# ---------------------------------------------------------------------
# Basic behavior of the primitives themselves
# ---------------------------------------------------------------------

def test_limit_increment_no_limits_is_a_plain_update():
    v0 = np.array([1.0, 2.0, 3.0])
    dv = np.array([0.5, -1.0, 10.0])
    out = PhysicalModel.limit_increment(v0, dv)
    assert np.allclose(out, v0 + dv)


def test_limit_increment_relative_limit_binds():
    v0 = np.array([100.0])
    dv = np.array([50.0])  # 50% relative change
    out = PhysicalModel.limit_increment(v0, dv, rel_max=0.1)  # cap at 10%
    assert np.isclose(out[0], 110.0)


def test_limit_increment_absolute_limit_binds():
    v0 = np.array([100.0])
    dv = np.array([50.0])
    out = PhysicalModel.limit_increment(v0, dv, abs_max=5.0)
    assert np.isclose(out[0], 105.0)


def test_limit_increment_takes_tighter_of_relative_and_absolute():
    v0 = np.array([100.0])
    dv = np.array([50.0])
    out = PhysicalModel.limit_increment(v0, dv, rel_max=0.2, abs_max=5.0)  # rel allows +20, abs allows +5
    assert np.isclose(out[0], 105.0)


def test_limit_increment_zero_increment_is_a_noop_even_at_zero_value():
    v0 = np.array([0.0])
    dv = np.array([0.0])
    out = PhysicalModel.limit_increment(v0, dv, rel_max=0.1, abs_max=1.0)
    assert np.isclose(out[0], 0.0)


def test_cap_property_clamps():
    x = np.array([-5.0, 0.5, 5.0])
    out = PhysicalModel.cap_property(x, 0.0, 1.0)
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_cnv_mb_from_residual_matches_hand_computation():
    residual = np.array([1.0, -2.0, 0.5])
    b = np.array([1.1, 1.2, 1.0])
    pv = np.array([10.0, 20.0, 30.0])
    rho = 1000.0
    dt = 86400.0
    cnv, mb = ReservoirModel.cnv_mb_from_residual(residual, b, rho, pv, dt)

    eq = residual / rho
    b_avg = np.mean(1.0 / b)
    expected_cnv = b_avg * dt * np.max(np.abs(eq) / pv)
    expected_mb = abs(b_avg * dt * np.sum(eq)) / np.sum(pv)
    assert np.isclose(cnv, expected_cnv)
    assert np.isclose(mb, expected_mb)


# ---------------------------------------------------------------------
# Legacy-formula equivalence, on realistic data (positive pressures ~1e7 Pa,
# saturations in [0,1]) -- the domain the actual SPE1/SPE9/Norne/Egg runs live in.
# ---------------------------------------------------------------------

def _legacy_pressure_limit_ow(p0, dp, dpMaxRel, dpMaxAbs, pmin, pmax):
    """Verbatim copy of the old inline formula from
    ``_update_state_mrst_generic_ow`` (pre-refactor)."""
    nc = p0.size
    pscale = np.ones(nc)
    if np.isfinite(dpMaxRel):
        pscale = np.minimum(pscale, np.divide(dpMaxRel, np.abs(dp / p0), out=np.full(nc, np.inf), where=np.abs(p0) > 0.0))
    if np.isfinite(dpMaxAbs):
        pscale = np.minimum(pscale, np.divide(dpMaxAbs, np.abs(dp), out=np.full(nc, np.inf), where=np.abs(dp) > 0.0))
    return np.clip(p0 + dp * pscale, pmin, pmax)


def _legacy_pressure_limit_3ph(p0, dp, dpMaxRel, dpMaxAbs, pmin, pmax):
    """Verbatim copy of the old inline formula from
    ``_update_state_mrst_generic`` (pre-refactor)."""
    nc = p0.size
    scale = np.ones(nc, dtype=float)
    if np.isfinite(dpMaxRel):
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.minimum(scale, np.minimum(dpMaxRel / np.abs(dp / p0), 1.0))
    if np.isfinite(dpMaxAbs):
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.minimum(scale, np.minimum(dpMaxAbs / np.abs(dp), 1.0))
    scale[~np.isfinite(scale)] = 1.0
    return np.clip(p0 + dp * scale, pmin, pmax)


def _legacy_saturation_scale_ow(dsw, dsMaxAbs):
    nc = dsw.size
    if np.isfinite(dsMaxAbs):
        mag = np.abs(dsw)
        return np.minimum(np.divide(dsMaxAbs, mag, out=np.full(nc, np.inf), where=mag > 0), 1.0)
    return np.ones(nc)


def _legacy_saturation_scale_3ph(dsw, dsg, dso, dsMaxAbs):
    nc = dsw.size
    if np.isfinite(dsMaxAbs):
        mag = np.maximum(np.maximum(np.abs(dsw), np.abs(dsg)), np.abs(dso))
        with np.errstate(divide='ignore', invalid='ignore'):
            sscale = np.minimum(dsMaxAbs / mag, 1.0)
        sscale[~np.isfinite(sscale)] = 1.0
        return sscale
    return np.ones(nc)


def test_limit_pressure_increment_matches_legacy_ow_formula_on_realistic_data():
    rng = np.random.default_rng(0)
    nc = 200
    p0 = rng.uniform(1.0e7, 4.0e7, nc)  # realistic reservoir pressures, always > 0
    dp = rng.uniform(-5.0e6, 5.0e6, nc)

    model = ReservoirModel(dpMaxRel=0.2, dpMaxAbs=2.0e6, minimumPressure=1.0e5, maximumPressure=1.0e8)
    new = model.limit_pressure_increment(p0, dp)
    legacy = _legacy_pressure_limit_ow(p0, dp, model.dpMaxRel, model.dpMaxAbs, model.minimumPressure, model.maximumPressure)
    assert np.allclose(new, legacy, rtol=1e-13, atol=1e-8)


def test_limit_pressure_increment_matches_legacy_3ph_formula_on_realistic_data():
    rng = np.random.default_rng(1)
    nc = 200
    p0 = rng.uniform(1.0e7, 4.0e7, nc)
    dp = rng.uniform(-5.0e6, 5.0e6, nc)

    model = ReservoirModel(dpMaxRel=0.15, dpMaxAbs=3.0e6, minimumPressure=1.0e5, maximumPressure=1.0e8)
    new = model.limit_pressure_increment(p0, dp)
    legacy = _legacy_pressure_limit_3ph(p0, dp, model.dpMaxRel, model.dpMaxAbs, model.minimumPressure, model.maximumPressure)
    assert np.allclose(new, legacy, rtol=1e-13, atol=1e-8)


def test_limit_pressure_increment_matches_legacy_formulas_with_only_one_limit_active():
    rng = np.random.default_rng(2)
    nc = 100
    p0 = rng.uniform(1.0e7, 4.0e7, nc)
    dp = rng.uniform(-5.0e6, 5.0e6, nc)

    # Only relative limit finite.
    model = ReservoirModel(dpMaxRel=0.1, dpMaxAbs=np.inf, minimumPressure=-np.inf, maximumPressure=np.inf)
    new = model.limit_pressure_increment(p0, dp)
    assert np.allclose(new, _legacy_pressure_limit_ow(p0, dp, 0.1, np.inf, -np.inf, np.inf), rtol=1e-13, atol=1e-8)
    assert np.allclose(new, _legacy_pressure_limit_3ph(p0, dp, 0.1, np.inf, -np.inf, np.inf), rtol=1e-13, atol=1e-8)

    # Only absolute limit finite.
    model2 = ReservoirModel(dpMaxRel=np.inf, dpMaxAbs=1.0e6, minimumPressure=-np.inf, maximumPressure=np.inf)
    new2 = model2.limit_pressure_increment(p0, dp)
    assert np.allclose(new2, _legacy_pressure_limit_ow(p0, dp, np.inf, 1.0e6, -np.inf, np.inf), rtol=1e-13, atol=1e-8)
    assert np.allclose(new2, _legacy_pressure_limit_3ph(p0, dp, np.inf, 1.0e6, -np.inf, np.inf), rtol=1e-13, atol=1e-8)


def test_limit_saturation_increment_matches_legacy_ow_formula():
    rng = np.random.default_rng(3)
    nc = 200
    sw0 = rng.uniform(0.1, 0.9, nc)
    dsw = rng.uniform(-0.4, 0.4, nc)

    model = ReservoirModel(dsMaxAbs=0.2)
    new = model.limit_saturation_increment(sw0, dsw)
    legacy_scale = _legacy_saturation_scale_ow(dsw, model.dsMaxAbs)
    legacy = sw0 + dsw * legacy_scale
    assert np.allclose(new, legacy, rtol=1e-13, atol=1e-10)


def test_limit_saturation_increment_matches_legacy_3ph_formula_per_component():
    # The 3-phase legacy code computes ONE shared scale from the worst of
    # (dsw, dsg, dso), then applies it to sw and sg individually. Reproduce
    # that composition using the new primitive's underlying scale via two
    # separate limit_increment calls sharing a precomputed scale.
    rng = np.random.default_rng(4)
    nc = 200
    sw0 = rng.uniform(0.1, 0.6, nc)
    sg0 = rng.uniform(0.0, 0.3, nc)
    dsw = rng.uniform(-0.3, 0.3, nc)
    dsg = rng.uniform(-0.3, 0.3, nc)
    dso = -(dsw + dsg)
    dsMaxAbs = 0.15

    legacy_scale = _legacy_saturation_scale_3ph(dsw, dsg, dso, dsMaxAbs)
    legacy_sw = sw0 + dsw * legacy_scale
    legacy_sg = sg0 + dsg * legacy_scale

    # New composition: the worst-of-three magnitude, capped by dsMaxAbs via
    # limit_increment applied to that worst-magnitude proxy, then reused as a
    # shared scale for sw/sg -- this is exactly what
    # ReservoirModel.updateSaturations does for a *joint* saturation limit.
    worst = np.maximum(np.maximum(np.abs(dsw), np.abs(dsg)), np.abs(dso))
    shared_scale = np.clip(PhysicalModel.limit_increment(np.zeros(nc), worst, abs_max=dsMaxAbs) / np.where(worst > 0, worst, 1.0), 0.0, 1.0)
    new_sw = sw0 + dsw * shared_scale
    new_sg = sg0 + dsg * shared_scale

    assert np.allclose(new_sw, legacy_sw, rtol=1e-12, atol=1e-10)
    assert np.allclose(new_sg, legacy_sg, rtol=1e-12, atol=1e-10)


def test_shared_saturation_scale_matches_legacy_3ph_formula():
    rng = np.random.default_rng(6)
    nc = 200
    dsw = rng.uniform(-0.3, 0.3, nc)
    dsg = rng.uniform(-0.3, 0.3, nc)
    dso = -(dsw + dsg)
    dsMaxAbs = 0.15

    model = ReservoirModel(dsMaxAbs=dsMaxAbs)
    scale = model.shared_saturation_scale(dsw, dsg, dso)
    legacy_scale = _legacy_saturation_scale_3ph(dsw, dsg, dso, dsMaxAbs)
    assert np.allclose(scale, legacy_scale, rtol=1e-12, atol=1e-12)


def test_cnv_mb_matches_legacy_per_phase_loop():
    rng = np.random.default_rng(5)
    nc = 300
    residual = rng.uniform(-1.0, 1.0, 3 * nc)
    bw = rng.uniform(0.98, 1.02, nc)
    bo = rng.uniform(1.1, 1.3, nc)
    bg = rng.uniform(0.004, 0.006, nc)
    rho_w, rho_o, rho_g = 1000.0, 850.0, 1.2
    pv = rng.uniform(50.0, 150.0, nc)
    dt = 86400.0 * 3

    b = (bw, bo, bg)
    rho_s = (rho_w, rho_o, rho_g)
    pv_total = float(np.sum(pv))
    legacy_cnv, legacy_mb = [], []
    for iph in range(3):
        eq = residual[iph * nc:(iph + 1) * nc] / rho_s[iph]
        Bavg = float(np.mean(1.0 / b[iph]))
        legacy_cnv.append(Bavg * dt * float(np.max(np.abs(eq) / pv)))
        legacy_mb.append(abs(Bavg * dt * float(np.sum(eq))) / pv_total)

    new_cnv, new_mb = [], []
    for iph in range(3):
        cnv, mb = ReservoirModel.cnv_mb_from_residual(residual[iph * nc:(iph + 1) * nc], b[iph], rho_s[iph], pv, dt)
        new_cnv.append(cnv)
        new_mb.append(mb)

    assert np.allclose(new_cnv, legacy_cnv, rtol=1e-13)
    assert np.allclose(new_mb, legacy_mb, rtol=1e-13)
