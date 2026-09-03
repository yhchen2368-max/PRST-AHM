"""Tests for PRSTCore.ad_core.timesteps: rampup_timesteps (MRST-verified,
see test_rampup_timesteps_mrst_parity.py) plus self-checks for
StateChangeTimeStepSelector/FactorTimeStepSelector."""

from __future__ import annotations

import numpy as np

from PRSTCore.ad_core.timesteps import FactorTimeStepSelector, StateChangeTimeStepSelector, rampup_timesteps


def test_rampup_timesteps_sums_to_total_time():
    day = 86400.0
    for time, dt, n in [(365 * day, 30 * day, 8), (100 * day, 10 * day, 3), (45 * day, 30 * day, 8)]:
        dT = rampup_timesteps(time, dt, n)
        assert np.isclose(dT.sum(), time, rtol=1e-10)
        assert np.all(dT > 0)


def test_rampup_timesteps_zero_time_is_empty():
    assert rampup_timesteps(0.0, 100.0).size == 0


def test_rampup_timesteps_initial_steps_are_geometric():
    dT = rampup_timesteps(1000.0, 100.0, n=4)
    # dt/2^4, dt/2^4, dt/2^3, dt/2^2, dt/2^1, then constant dt=100 steps.
    assert np.isclose(dT[0], 100.0 / 16)
    assert np.isclose(dT[1], 100.0 / 16)
    assert np.isclose(dT[2], 100.0 / 8)
    assert np.isclose(dT[3], 100.0 / 4)
    assert np.isclose(dT[4], 100.0 / 2)


def test_state_change_selector_grows_when_change_below_target():
    sel = StateChangeTimeStepSelector(target_change=0.2, growth_factor=2.0, cut_factor=0.5)
    dt_next = sel.compute_next_dt(dt_prev=100.0, relative_change=0.05)
    assert dt_next > 100.0  # well under target -> grow


def test_state_change_selector_shrinks_when_change_exceeds_target():
    sel = StateChangeTimeStepSelector(target_change=0.2, growth_factor=2.0, cut_factor=0.5)
    dt_next = sel.compute_next_dt(dt_prev=100.0, relative_change=0.8)
    assert dt_next < 100.0


def test_state_change_selector_respects_bounds():
    sel = StateChangeTimeStepSelector(target_change=0.2, dt_min=10.0, dt_max=50.0, growth_factor=10.0)
    assert sel.compute_next_dt(dt_prev=40.0, relative_change=0.0) == 50.0
    sel2 = StateChangeTimeStepSelector(target_change=0.2, dt_min=10.0, dt_max=50.0, cut_factor=0.01)
    assert sel2.compute_next_dt(dt_prev=15.0, relative_change=100.0) == 10.0


def test_factor_selector_grows_on_convergence_shrinks_on_failure():
    sel = FactorTimeStepSelector(growth_factor=1.5, cut_factor=0.5, dt_min=1.0, dt_max=1000.0)
    assert np.isclose(sel.compute_next_dt(100.0, converged=True), 150.0)
    assert np.isclose(sel.compute_next_dt(100.0, converged=False), 50.0)
