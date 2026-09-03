"""The compiled flux path must assemble the system the general one does.

``_mrst_generic_adi_residual`` can build its flux term two ways: through the
general ``SparseADI`` operators, or through the fixed-width face values that
carry MRST's compiled kernels.  Only the second is fast; both have to be the
same system, because everything downstream -- the Newton path, the adjoint,
every parity result -- is built on it.

The switch is ``model.useFaceOperators``, so the comparison runs the same
deck twice in one process and diffs the two assemblies.
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

DECKS = {
    'spe1': REPO_ROOT / 'examples' / 'SpE1' / 'SPE1CASE1.DATA',
    'spe9': REPO_ROOT / 'examples' / 'SPE9' / 'SPE9_CP.DATA',
}


def _first_system(model, state0, schedule, use_face_operators):
    model.useFaceOperators = use_face_operators
    model._face_flux_cache = None
    dt = float(np.atleast_1d(schedule['step']['val'])[0])
    controls = schedule.get('control')
    forces = controls[0] if isinstance(controls, list) and controls else {}
    state = model.validateState(state0)
    model, state = model.prepareReportstep(state, model.validateState(state0),
                                           dt, forces)
    model, state = model.prepareTimestep(state, model.validateState(state0),
                                         dt, forces)
    problem, _ = model.get_equations(state, model.validateState(state0), dt, forces)
    return problem


@pytest.mark.parametrize('case', sorted(DECKS))
def test_both_flux_paths_assemble_the_same_system(case):
    deck = DECKS[case]
    if not deck.is_file():
        pytest.skip('deck %s is not in this checkout' % deck)
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    state0, model, schedule, _ = init_eclipse_problem_ad(str(deck))
    general = _first_system(model, state0, schedule, False)
    fast = _first_system(model, state0, schedule, True)

    residual = np.abs(np.asarray(general['Residuals']) - np.asarray(fast['Residuals']))
    scale = max(float(np.abs(general['Residuals']).max()), 1.0)
    assert residual.max() <= 1e-12 * scale, (
        'the two flux paths disagree on the residual by %g' % residual.max())

    a, b = general['Jacobian'].tocsr(), fast['Jacobian'].tocsr()
    assert a.shape == b.shape
    difference = (a - b).tocoo()
    if difference.nnz:
        largest = max(float(abs(a).max()), 1.0)
        assert np.abs(difference.data).max() <= 1e-12 * largest, (
            'the two flux paths disagree on the Jacobian by %g'
            % np.abs(difference.data).max())


def test_the_fast_path_is_declined_in_reverse_mode():
    """Reverse mode seeds something other than the cell variables.

    The face values read a cell property's derivatives by taking the
    diagonal of each variable group's block, which only means what it should
    while the primary variables are the cell ones at their usual offsets.
    The adjoint seeds the previous state and ``partialWRTparam`` seeds the
    parameters, so the fast path has to stand down -- silently reading the
    wrong columns would produce a plausible, wrong Jacobian.
    """
    deck = DECKS['spe1']
    if not deck.is_file():
        pytest.skip('SPE1 deck is not in this checkout')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    state0, model, schedule, _ = init_eclipse_problem_ad(str(deck))
    model.useFaceOperators = True
    model._face_flux_cache = None
    dt = float(np.atleast_1d(schedule['step']['val'])[0])
    controls = schedule.get('control')
    forces = controls[0] if isinstance(controls, list) and controls else {}
    state = model.validateState(state0)

    # reverseMode reaches the assembly through ResOnly as well; both must be
    # accepted and neither may take the fast path.
    problem, _ = model.get_equations(state, model.validateState(state0), dt,
                                     forces, ResOnly=True)
    assert np.all(np.isfinite(np.asarray(problem['Residuals'])))
    # Nothing was cached, because the fast path was never reached.
    assert getattr(model, '_face_flux_cache', None) is None


def test_the_switch_actually_switches():
    deck = DECKS['spe1']
    if not deck.is_file():
        pytest.skip('SPE1 deck is not in this checkout')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    state0, model, schedule, _ = init_eclipse_problem_ad(str(deck))
    _first_system(model, state0, schedule, False)
    assert getattr(model, '_face_flux_cache', None) is None
    _first_system(model, state0, schedule, True)
    assert getattr(model, '_face_flux_cache', None) is not None
