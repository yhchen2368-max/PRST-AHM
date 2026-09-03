"""The PETSc solver must return the same Newton update as a direct solve.

An iterative solver is only a solver if it converges on the systems it will
actually be given.  A black-oil Jacobian is not one of the systems multigrid
works on out of the box: its leading block is a component balance
differentiated with respect to pressure, not a pressure operator, and giving
that to a field split converges nowhere -- measured on SPE9, 200 iterations
and a relative residual of 0.8.  What makes it work is the CPR decoupling,
so that is what these tests pin down: with it the answer matches a direct
solve, and the pieces it relies on (an invertible row combination, explicitly
stored diagonals) hold on their own.
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
from PRSTCore.ad_core.solvers.petsc_solver_ad import (PETScSolverAD, check_petsc,
                                                      petsc_has_preconditioner)

petsc_required = pytest.mark.skipif(
    not check_petsc(), reason='petsc4py is not built for this interpreter')

SPE9 = REPO_ROOT / 'examples' / 'SPE9' / 'SPE9_CP.DATA'


@pytest.fixture(scope='module')
def spe9_first_system():
    """The first Newton system of SPE9, with its model."""
    if not SPE9.is_file():
        pytest.skip('SPE9 deck is not in this checkout')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    state0, model, schedule, _ = init_eclipse_problem_ad(str(SPE9))
    dt = float(np.atleast_1d(schedule['step']['val'])[0])
    controls = schedule.get('control')
    forces = controls[0] if isinstance(controls, list) and controls else {}
    problem, _ = model.get_equations(model.validateState(state0),
                                     model.validateState(state0), dt, forces)
    A = problem['Jacobian'].tocsr()
    b = -np.asarray(problem['Residuals'], dtype=float).ravel()
    reference = spla.spsolve(A.tocsc(), b)
    return problem, model, A, b, reference


@petsc_required
@pytest.mark.parametrize('strategy', ['cpr', 'fieldsplit'])
def test_matches_a_direct_solve_on_spe9(spe9_first_system, strategy):
    problem, model, A, b, reference = spe9_first_system
    solver = PETScSolverAD(strategy=strategy, pressure_precond='gamg',
                           tolerance=1e-8, maxIterations=300)
    try:
        x, relative, report = solver.solveLinearProblem(problem, model=model)
        assert report['Converged'], (
            'PETSc %s did not converge: reason %s after %d iterations'
            % (strategy, report.get('ConvergedReason'), report['Iterations']))
        error = np.linalg.norm(x - reference) / np.linalg.norm(reference)
        assert error < 1e-5, 'Newton update differs from the direct solve by %g' % error
        assert relative < 1e-7
    finally:
        solver.destroy()


@petsc_required
def test_setup_reuse_does_not_change_the_answer(spe9_first_system):
    """A second solve on the same pattern reuses the hierarchy, not the answer.

    Agreement is asked for in a norm, to the tolerance the solver was given.
    Element by element it would not hold and should not be expected to: two
    Krylov runs stop at different points inside the same tolerance ball, so
    a component that happens to be near zero has a large *relative*
    difference while contributing nothing to the solution.
    """
    tolerance = 1e-8
    problem, model, A, b, reference = spe9_first_system
    solver = PETScSolverAD(strategy='cpr', tolerance=tolerance, maxIterations=300)
    try:
        first, _, _ = solver.solveLinearProblem(problem, model=model)
        second, _, report = solver.solveLinearProblem(problem, model=model)
        assert report['Converged']
        drift = np.linalg.norm(first - second) / np.linalg.norm(first)
        assert drift < 100 * tolerance, (
            'reusing the setup moved the solution by %g, more than the '
            'tolerance can explain' % drift)
    finally:
        solver.destroy()


@petsc_required
def test_decoupling_is_invertible_and_preserves_the_solution(spe9_first_system):
    """``M A x = M b`` must have the solution ``A x = b`` has.

    The decoupling changes which equations are being solved, not which
    unknowns; if ``M`` were singular the transformed system could be
    satisfied by vectors the original is not.
    """
    problem, model, A, b, reference = spe9_first_system
    nc = int(np.asarray(problem['State']['pressure']).size)
    ncomp = sum(1 for t in problem['types'] if t == 'cell') // nc
    solver = PETScSolverAD(decoupling='quasiIMPES')
    M = solver._decoupling_operator(A, nc, ncomp)

    # M is block diagonal per cell plus an identity tail, so its action on
    # the known solution must reproduce the transformed right-hand side.
    np.testing.assert_allclose(M @ (A @ reference), M @ b, rtol=1e-7,
                               atol=1e-6 * np.abs(M @ b).max())
    # The rows it did not touch are still the identity.
    tail = M[nc:, :].tocsr()
    identity = sp.eye(A.shape[0], format='csr')[nc:, :]
    assert (tail - identity).nnz == 0


@petsc_required
def test_decoupling_removes_the_saturation_dependence(spe9_first_system):
    """The combined equation must not depend on its own cell's saturations.

    That is exactly what the weights are solved for, and it is the property
    that makes the leading block an operator in pressure alone -- the thing
    multigrid can be given.  Checking it says *why* the solver converges,
    which a timing number does not, and it fails loudly if the equation and
    variable orderings are ever not what the weights assume.
    """
    problem, model, A, b, reference = spe9_first_system
    nc = int(np.asarray(problem['State']['pressure']).size)
    ncomp = sum(1 for t in problem['types'] if t == 'cell') // nc
    assert ncomp >= 2, 'SPE9 should present several component balances'

    solver = PETScSolverAD(decoupling='quasiIMPES')
    decoupled = (solver._decoupling_operator(A, nc, ncomp) @ A).tocsr()
    cells = np.arange(nc)
    pressure_coefficient = np.abs(decoupled[cells, cells].A.ravel())

    for variable in range(1, ncomp):
        columns = cells + variable * nc
        coupling = np.abs(decoupled[cells, columns].A.ravel())
        # Relative to the pressure coefficient the same row carries: the
        # rows are not normalised, so an absolute threshold would only be
        # measuring the units the deck is in.
        scale = np.maximum(pressure_coefficient, np.finfo(float).tiny)
        assert np.median(coupling / scale) < 1e-8, (
            'variable group %d still couples into the pressure equation'
            % variable)

    # And the same coupling before decoupling is not small, or there would
    # have been nothing to do.
    raw = A.tocsr()
    raw_coupling = np.abs(raw[cells, cells + nc].A.ravel())
    raw_pressure = np.maximum(np.abs(raw[cells, cells].A.ravel()),
                              np.finfo(float).tiny)
    assert np.median(raw_coupling / raw_pressure) > 1e-8


@petsc_required
def test_explicit_diagonal_inserts_only_missing_entries():
    A = sp.csr_matrix((np.array([1.0, 2.0]),
                       (np.array([0, 1]), np.array([1, 0]))), shape=(3, 3))
    padded = PETScSolverAD._with_explicit_diagonal(A)
    assert padded.nnz == 5, 'expected three diagonal entries to be inserted'
    np.testing.assert_allclose(padded.toarray(), A.toarray())
    # An already-complete diagonal is returned untouched.
    full = sp.eye(3, format='csr')
    assert PETScSolverAD._with_explicit_diagonal(full) is full


@petsc_required
def test_hypre_request_falls_back_when_unavailable():
    """Asking for hypre on a build without it must degrade, not raise."""
    resolved = PETScSolverAD._resolve_pressure_precond('hypre')
    if petsc_has_preconditioner('hypre'):
        assert resolved == 'hypre'
    else:
        assert resolved == 'gamg'


@petsc_required
def test_rejects_an_unknown_strategy():
    with pytest.raises(ValueError):
        PETScSolverAD(strategy='not-a-strategy')


SPE10_1 = REPO_ROOT / 'examples' / 'spe10model1' / 'SPE10_MODEL1_CP.DATA'


@petsc_required
@pytest.mark.skipif(not SPE10_1.is_file(), reason='SPE10 model 1 is not in this checkout')
def test_falls_back_to_boomeramg_when_gamg_stalls():
    """A permeability field GAMG cannot coarsen must not end the run.

    SPE10's permeability spans six orders of magnitude, and PETSc's own
    smoothed-aggregation multigrid stalls on it -- 200 iterations at a
    relative residual of 0.75.  BoomerAMG converges in 26.  On SPE9 the
    preference is the other way round and gamg is five times faster, so the
    solver tries and switches rather than choosing once for every model.
    """
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    if not petsc_has_preconditioner('hypre'):
        pytest.skip('this PETSc has no hypre to fall back to')

    state0, model, schedule, _ = init_eclipse_problem_ad(str(SPE10_1))
    dt = float(np.atleast_1d(schedule['step']['val'])[0])
    controls = schedule.get('control')
    forces = controls[0] if isinstance(controls, list) and controls else {}
    state = model.validateState(state0)
    model, state = model.prepareReportstep(state, model.validateState(state0), dt, forces)
    model, state = model.prepareTimestep(state, model.validateState(state0), dt, forces)
    problem, _ = model.get_equations(state, model.validateState(state0), dt, forces)

    A = problem['Jacobian'].tocsr()
    b = -np.asarray(problem['Residuals'], dtype=float).ravel()
    reference = spla.spsolve(A.tocsc(), b)

    solver = PETScSolverAD(strategy='cpr', pressure_precond='gamg',
                           tolerance=1e-6, maxIterations=200)
    try:
        x, relative, report = solver.solveLinearProblem(problem, model=model)
        assert solver.pressure_precond == 'hypre', (
            'gamg stalls on SPE10 and the solver should have switched')
        assert report['Converged']
        error = np.linalg.norm(x - reference) / np.linalg.norm(reference)
        assert error < 1e-6, error
        # The switch is one-way: the next solve must not pay for it again.
        solver.solveLinearProblem(problem, model=model)
        assert solver.pressure_precond == 'hypre'
    finally:
        solver.destroy()
