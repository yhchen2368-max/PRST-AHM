"""RESV wells are controlled on reservoir volume, not surface volume.

The deck reader has always produced ``resv`` and ``resv_history`` control
types; the facility model raised "Unsupported MRST well control type" on
both.  Two correct halves, never joined -- and the seam only showed on decks
that use RESV, which are exactly the large ones: Norne and SPE10 model 2 both
stopped at the first residual.

What joins them is MRST's ``updateRESVControls``: once per report step it
freezes a per-phase surface-to-reservoir conversion (``ControlDensity``) from
the previous state, and converts any ``resv_history`` well to a plain
``resv`` whose target is recomputed through those factors.  The control
equation is then ``sum_phase q_surface * factor - target``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import PRSTCore  # noqa: F401
from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.models.facility_model import FacilityModel

NORNE = REPO_ROOT / 'examples' / 'Norne' / 'NORNE_ATW2013.DATA'


def _rates(values, nvar=None):
    """Three phase surface-rate vectors seeded as separate AD variables."""
    values = np.asarray(values, dtype=float)
    nwell = values.shape[0]
    nvar = nvar if nvar is not None else 3 * nwell
    return {code: SparseADI.variable(values[:, k], nvar, k * nwell)
            for k, code in enumerate('wog')}


def test_resv_equation_weights_surface_rates_by_control_density():
    """The residual is the reservoir rate minus the target."""
    surface = np.array([[2.0, 5.0, 30.0]])
    factors = np.array([1.1, 1.3, 0.004])
    target = 9.0
    well = {'name': 'P1', 'type': 'resv', 'val': target,
            'ControlDensity': factors}

    qs = _rates(surface)
    closure = FacilityModel.compute_control_equations(
        [well], qs_phases=qs, bhp=SparseADI.variable([250.0], 3, 0),
        phase_order=['w', 'o', 'g'])

    assert len(closure) == 1
    expected = float(surface[0] @ factors - target)
    np.testing.assert_allclose(closure[0].val, [expected], rtol=1e-12)

    # The derivative with respect to each surface rate is that phase's factor.
    jacobian = closure[0].jac.toarray().ravel()
    np.testing.assert_allclose(jacobian, factors, rtol=1e-12)


def test_resv_without_control_density_is_refused_by_name():
    """A missing conversion must say so, not silently act like a rate control."""
    well = {'name': 'BADWELL', 'type': 'resv', 'val': 1.0}
    with pytest.raises(ValueError, match='BADWELL'):
        FacilityModel.compute_control_equations(
            [well], qs_phases=_rates(np.array([[1.0, 1.0, 1.0]])),
            bhp=SparseADI.variable([250.0], 3, 0), phase_order=['w', 'o', 'g'])


@pytest.mark.skipif(not NORNE.is_file(), reason='Norne deck is not in this checkout')
def test_norne_assembles_and_converts_its_resv_history_wells():
    """Norne's first system must assemble, which is what RESV blocked.

    Slow -- Norne's corner-point grid takes a couple of minutes to build --
    but this is the case the feature exists for, and a synthetic well cannot
    exercise the report-step ordering that produces the factors.
    """
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    state0, model, schedule, _ = init_eclipse_problem_ad(str(NORNE))
    dt = float(np.atleast_1d(schedule['step']['val'])[0])
    controls = schedule.get('control')
    forces = controls[0] if isinstance(controls, list) and controls else {}

    raw = forces.get('W', []) if isinstance(forces, dict) else []
    resv_before = [w for w in raw
                   if str(w.get('type', '')).lower() in ('resv', 'resv_history')]
    assert resv_before, 'Norne is expected to use RESV controls'

    state = model.validateState(state0)
    model, state = model.prepareReportstep(state, model.validateState(state0),
                                           dt, forces)
    model, state = model.prepareTimestep(state, model.validateState(state0),
                                         dt, forces)

    wells = model._mrst_active_wells(forces, state)
    resv = [w for w in wells if str(w.get('type', '')).lower() == 'resv']
    assert resv, 'RESV wells should still be RESV after preparation'
    assert not any(str(w.get('type', '')).lower() == 'resv_history' for w in wells), \
        'resv_history must have been converted to resv'
    for well in resv:
        factors = np.asarray(well['ControlDensity'], dtype=float)
        assert factors.size >= 2
        assert np.all(np.isfinite(factors)), well.get('name')

    problem, _ = model.get_equations(state, model.validateState(state0), dt, forces)
    residual = np.asarray(problem['Residuals'], dtype=float)
    assert np.all(np.isfinite(residual))
    assert problem['Jacobian'].shape[0] == residual.size
    # The closure equations are the last one per well and must not be trivial.
    closure_rows = [i for i, name in enumerate(problem['equationNames'])
                    if name == 'closureWells']
    assert closure_rows
    assert np.any(np.abs(residual[closure_rows]) > 0.0)
