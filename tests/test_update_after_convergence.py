"""The solver must run MRST's post-convergence state update.

NonLinearSolver.m calls ``model.updateAfterConvergence(state0, state, dt,
forces)`` on every converged ministep. That is where end-of-step state
which is *not* a primary variable gets advanced -- notably the
polymer/surfactant maximum concentrations that drive irreversible
adsorption (PLYROCK item 4 == 2).
"""

import numpy as np
import pytest

from PRSTCore.ad_core.solvers.nonlinear_solver import NonLinearSolver


class _RecordingModel:
    """Minimal model: converges immediately and tracks a running maximum."""

    def __init__(self):
        self.calls = []

    def stepFunction(self, state, state0, dt, **kwargs):
        state = dict(state)
        state['c'] = state['c'] + 1.0
        return state, {'Converged': True, 'Iterations': 1, 'Residuals': [0.0]}

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        self.calls.append((float(dt), float(np.max(state['c']))))
        state['cmax'] = np.maximum(state.get('cmax', state['c']), state['c'])
        return state


class _TupleReturningModel(_RecordingModel):
    """PhysicalModel.updateAfterConvergence returns [state, report]."""

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        state = super().updateAfterConvergence(state0, state, dt, drivingForces)
        return state, {'note': 'final update'}


class _ModelWithoutHook:
    def stepFunction(self, state, state0, dt, **kwargs):
        state = dict(state)
        state['c'] = state['c'] + 1.0
        return state, {'Converged': True, 'Iterations': 1, 'Residuals': [0.0]}


def _solve(model, dt=1.0):
    solver = NonLinearSolver()
    state0 = {'c': np.zeros(3), 'time': 0.0}
    return solver.solveTimestep(state0, dt, model)


def test_hook_is_called_on_a_converged_step():
    model = _RecordingModel()
    state, _report, _ = _solve(model)
    assert len(model.calls) == 1
    assert np.allclose(state['cmax'], state['c'])


def test_running_maximum_persists_across_steps():
    """cmax must not fall back when the concentration later decreases."""
    model = _RecordingModel()
    solver = NonLinearSolver()
    state = {'c': np.zeros(3), 'time': 0.0}
    for _ in range(3):
        state, _r, _m = solver.solveTimestep(state, 1.0, model)
    assert np.allclose(state['c'], 3.0)
    assert np.allclose(state['cmax'], 3.0)

    # A step that lowers c must leave cmax at its historical peak.
    state['c'] = np.full(3, -5.0)
    state, _r, _m = solver.solveTimestep(state, 1.0, model)
    assert np.allclose(state['c'], -4.0)
    assert np.allclose(state['cmax'], 3.0)


def test_tuple_return_is_unpacked_and_reported():
    model = _TupleReturningModel()
    state, report, _ = _solve(model)
    assert np.allclose(state['cmax'], state['c'])
    final = [r.get('FinalUpdate') for r in report['StepReports']]
    assert any(f == {'note': 'final update'} for f in final)


def test_model_without_the_hook_still_solves():
    state, report, _ = _solve(_ModelWithoutHook())
    assert report['Converged']
    assert np.allclose(state['c'], 1.0)


def test_recorded_ministates_reflect_the_update():
    model = _RecordingModel()
    _state, _report, ministates = _solve(model)
    assert ministates, 'expected at least one recorded ministep'
    assert 'cmax' in ministates[-1]
    assert np.allclose(ministates[-1]['cmax'], ministates[-1]['c'])


@pytest.mark.parametrize('dt', [0.5, 2.0])
def test_hook_receives_the_ministep_timestep(dt):
    model = _RecordingModel()
    _solve(model, dt=dt)
    assert model.calls[0][0] == pytest.approx(dt)
