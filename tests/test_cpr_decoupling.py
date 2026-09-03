"""MRST's CPR decoupling strategies, all four of them.

The decoupling is what gives CPR a pressure equation to precondition.  A
deck-derived Jacobian has none: its rows are component mass balances, so its
leading block is the water balance differentiated with respect to pressure,
and multigrid on that converges nowhere.  Each strategy answers the same
question -- what weights combine a cell's balances into something elliptic --
and they differ in what they are allowed to look at.

What every one of them must satisfy is here: the combination is invertible,
so the transformed system has the original's solution; and the pressure
block it produces is one a multigrid solver can actually use, which is
checked by solving with it and comparing against a direct solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import PRSTCore  # noqa: F401
from PRSTCore.ad_core import cpr_decoupling as cpr

SPE9 = REPO_ROOT / 'examples' / 'SPE9' / 'SPE9_CP.DATA'
STRATEGIES = ('none', 'quasiIMPES', 'trueIMPES', 'simple')


@pytest.fixture(scope='module')
def spe9():
    """SPE9's first Newton system, its model, and a direct solution."""
    if not SPE9.is_file():
        pytest.skip('SPE9 deck is not in this checkout')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    state0, model, schedule, _ = init_eclipse_problem_ad(str(SPE9))
    dt = float(np.atleast_1d(schedule['step']['val'])[0])
    controls = schedule.get('control')
    forces = controls[0] if isinstance(controls, list) and controls else {}
    state = model.validateState(state0)
    model, state = model.prepareReportstep(state, model.validateState(state0), dt, forces)
    model, state = model.prepareTimestep(state, model.validateState(state0), dt, forces)
    problem, _ = model.get_equations(state, model.validateState(state0), dt, forces)
    A = problem['Jacobian'].tocsr()
    b = -np.asarray(problem['Residuals'], dtype=float).ravel()
    nc = int(np.asarray(problem['State']['pressure']).size)
    ncomp = sum(1 for t in problem['types'] if t == 'cell') // nc
    return problem, model, A, b, nc, ncomp, spla.spsolve(A.tocsc(), b)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_weights_are_finite_and_leave_every_cell_a_voice(spe9, strategy):
    problem, model, A, b, nc, ncomp, _ = spe9
    weights = cpr.decoupling_weights(strategy, A, nc, ncomp, model=model,
                                     state=problem['State'])
    assert weights.shape == (nc, ncomp)
    assert np.all(np.isfinite(weights))
    # A cell whose weights were all zero would contribute an empty pressure
    # equation, and the combination would stop being invertible.
    assert np.all(np.abs(weights).max(axis=1) > 0.0)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_the_combination_preserves_the_solution(spe9, strategy):
    """``M A x = M b`` must be solved by the ``x`` that solves ``A x = b``."""
    problem, model, A, b, nc, ncomp, reference = spe9
    weights = cpr.decoupling_weights(strategy, A, nc, ncomp, model=model,
                                     state=problem['State'])
    M = cpr.decoupling_operator(weights, A.shape[0], nc, ncomp)
    residual = M @ (A @ reference) - M @ b
    scale = max(np.abs(M @ b).max(), np.finfo(float).tiny)
    assert np.abs(residual).max() <= 1e-6 * scale

    # Rows past the cells -- the well equations -- are untouched.
    tail = M[nc:, :].tocsr()
    assert (tail - sp.eye(A.shape[0], format='csr')[nc:, :]).nnz == 0


def test_quasi_impes_zeroes_a_cell_s_own_saturation_coupling(spe9):
    """Its defining property: solve ``D^T w = e_p`` and the saturations go.

    The other strategies do not promise this -- they approximate the same
    end from the fluid state rather than from the matrix -- so it is
    asserted only where it is the definition.
    """
    problem, model, A, b, nc, ncomp, _ = spe9
    weights = cpr.quasi_impes_weights(A, nc, ncomp)
    decoupled = (cpr.decoupling_operator(weights, A.shape[0], nc, ncomp) @ A).tocsr()
    cells = np.arange(nc)
    pressure = np.abs(decoupled[cells, cells].A.ravel())
    for variable in range(1, ncomp):
        coupling = np.abs(decoupled[cells, cells + variable * nc].A.ravel())
        scale = np.maximum(pressure, np.finfo(float).tiny)
        assert np.median(coupling / scale) < 1e-8


def test_true_impes_matches_the_black_oil_reduction_factors(spe9):
    """The weights are ``f_phase / pore_volume`` with MRST's ``f``.

    Recomputed here from the model's own fluid state rather than compared
    against stored numbers, so the test still means something if the PVT
    tables or the surface densities change.
    """
    problem, model, A, b, nc, ncomp, _ = spe9
    state = problem['State']
    weights = cpr.true_impes_weights(model, state, nc, ncomp)

    pressure = np.asarray(state['pressure'], dtype=float).ravel()
    rs = np.asarray(state.get('rs', np.zeros(nc)), dtype=float).ravel()
    pvt = model._phase_pvt(pressure, rs_override=rs if model.disgas else None)
    rho_ws, rho_os, rho_gs = np.asarray(model._mrst_surface_densities(), dtype=float)[:3]
    pv = np.asarray(model.porevolume, dtype=float).ravel()

    # Water is the one component with no dissolution correction, so its
    # weight is exactly the reciprocal density over pore volume.
    expected_water = 1.0 / (np.asarray(pvt['bw'], dtype=float).ravel() * rho_ws) / pv
    np.testing.assert_allclose(weights[:, 0], expected_water, rtol=1e-10)
    assert np.all(weights[:, 0] > 0)


def test_dynamic_row_sum_only_removes_and_never_empties_a_cell(spe9):
    problem, model, A, b, nc, ncomp, _ = spe9
    weights = cpr.quasi_impes_weights(A, nc, ncomp)
    filtered = cpr.apply_dynamic_row_sum(weights, A, nc, ncomp)
    assert filtered.shape == weights.shape
    kept = filtered != 0
    # Every surviving weight is the one it started as; DRS drops, never edits.
    np.testing.assert_allclose(filtered[kept], weights[kept], rtol=0, atol=0)
    assert np.all(np.abs(filtered).max(axis=1) > 0.0)


def test_unknown_strategy_is_refused():
    A = sp.eye(4, format='csr')
    with pytest.raises(ValueError, match='unknown CPR decoupling'):
        cpr.decoupling_weights('impes-ish', A, 2, 2)


def test_fluid_strategies_refuse_to_run_without_a_model():
    """Silently substituting quasiIMPES would be a different preconditioner."""
    A = sp.eye(4, format='csr')
    for strategy in ('trueIMPES', 'simple'):
        with pytest.raises(ValueError, match='needs the model'):
            cpr.decoupling_weights(strategy, A, 2, 2)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_every_strategy_solves_spe9_to_the_direct_answer(spe9, strategy):
    """The point of all of it: CPR converges and gets the same update."""
    from PRSTCore.ad_core.solvers.petsc_solver_ad import PETScSolverAD, check_petsc

    if not check_petsc():
        pytest.skip('petsc4py is not built for this interpreter')
    problem, model, A, b, nc, ncomp, reference = spe9
    solver = PETScSolverAD(strategy='cpr', pressure_precond='gamg',
                           decoupling=strategy, tolerance=1e-8,
                           maxIterations=300)
    try:
        x, relative, report = solver.solveLinearProblem(problem, model=model)
        assert report['Converged'], (
            '%s did not converge: %d iterations, relative residual %g'
            % (strategy, report['Iterations'], relative))
        assert report['Decoupling'].lower() == strategy.lower().replace('-', '')
        error = np.linalg.norm(x - reference) / np.linalg.norm(reference)
        assert error < 1e-4, '%s gave a different Newton update (%g)' % (strategy, error)
    finally:
        solver.destroy()
