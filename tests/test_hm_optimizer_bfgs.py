"""Tests for ``L_BFGS_B`` and ``unitBoxBFGSForEclipseRun``."""

import os

import numpy as np
import pytest
import scipy.sparse as sp

from PRSTCore.hm.utils.optimizer.L_BFGS_B import (L_BFGS_B, boundsType,
                                                  findMaxStep, projectedGrad)
from PRSTCore.hm.utils.optimizer.optimizeLinIneqConstrained import \
    getConstraints
from PRSTCore.hm.utils.optimizer.unitBoxBFGSForEclipseRun import (
    checkFeasible, unitBoxBFGSForEclipseRun)


def quadratic(target):
    target = np.asarray(target, dtype=float)

    def f(u):
        u = np.asarray(u, dtype=float).ravel()
        return float(np.sum((u - target) ** 2)), 2.0 * (u - target)
    return f


# ----------------------------------------------------------- L_BFGS_B --

def test_bounds_type_reports_two_sided_bounds():
    assert list(boundsType(np.zeros(3), np.zeros(3), np.ones(3))) == [2, 2, 2]


def test_bounds_type_reports_a_lower_bound_only():
    assert list(boundsType(np.zeros(2), np.zeros(2), None)) == [1, 1]


def test_bounds_type_reports_an_upper_bound_only():
    assert list(boundsType(np.zeros(2), None, np.ones(2))) == [3, 3]


def test_bounds_type_reports_unbounded():
    assert list(boundsType(np.zeros(2), None, None)) == [0, 0]


@pytest.mark.parametrize('x, g', [
    (0.0, -5.0),      # at the lower bound, heading up (feasible)
    (0.0, 5.0),       # at the lower bound, heading out
    (1.0, -5.0),      # at the upper bound, heading out
    (1.0, 5.0),       # at the upper bound, heading up (infeasible)
    (0.5, -5.0),      # interior
])
def test_projected_gradient_never_clips_inside_the_box(x, g):
    """MRST's projectedGrad has its two branches transposed and is
    therefore the identity on any feasible point -- see the function's
    docstring. Pinning that here so the defect cannot be reintroduced or
    silently corrected."""
    pg = projectedGrad(np.array([x]), np.zeros(1), np.ones(1),
                       np.array([2]), np.array([g]))
    assert pg[0] == pytest.approx(g)


def test_projected_gradient_would_clip_if_the_branches_were_paired_right():
    """The reference algorithm zeroes the gradient of a variable pinned
    at the bound it is heading towards; MRST's does not. This states the
    difference in one place."""
    x, l, u, g = np.array([0.0]), np.zeros(1), np.ones(1), np.array([5.0])
    reference = min(float(x[0] - l[0]), float(g[0]))     # projgr: g >= 0
    mrst = projectedGrad(x, l, u, np.array([2]), g)[0]
    assert reference == pytest.approx(0.0)
    assert mrst == pytest.approx(5.0)


def test_max_step_stops_at_the_nearest_bound():
    smax = findMaxStep(np.array([0.5, 0.5]), np.array([1.0, 0.25]),
                       np.zeros(2), np.ones(2))
    assert smax == pytest.approx(0.5)


def test_max_step_ignores_components_that_do_not_move():
    smax = findMaxStep(np.array([0.5]), np.array([0.0]), np.zeros(1),
                       np.ones(1))
    assert not np.isfinite(smax)


def test_lbfgsb_finds_an_interior_minimum():
    v, u, _ = L_BFGS_B(np.array([0.5, 0.5]), quadratic([0.3, 0.7]),
                       maximize=False, gradTol=1e-9, objChangeTol=1e-14,
                       maxIt=50, verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-5)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_lbfgsb_projects_onto_the_box():
    _, u, _ = L_BFGS_B(np.array([0.5, 0.5]), quadratic([-0.5, 1.5]),
                       maximize=False, gradTol=1e-9, objChangeTol=1e-14,
                       maxIt=50, verbose=False)
    assert u[0] == pytest.approx(0.0, abs=1e-6)
    assert u[1] == pytest.approx(1.0, abs=1e-6)


def test_lbfgsb_maximizes_when_asked():
    _, u, _ = L_BFGS_B(np.array([0.5, 0.5]), quadratic([0.0, 0.0]),
                       maximize=True, gradTol=1e-9, objChangeTol=1e-14,
                       maxIt=50, verbose=False)
    assert np.allclose(u, [1.0, 1.0], atol=1e-3)


def test_lbfgsb_agrees_with_the_linear_inequality_version_unconstrained():
    """With only the box active, the two algorithms must land together."""
    from PRSTCore.hm.utils.optimizer.optimizeLinIneqConstrained import \
        optimizeLinIneqConstrained
    kw = dict(maximize=False, gradTol=1e-9, objChangeTol=1e-14, maxIt=50,
              verbose=False)
    _, u1, _ = L_BFGS_B(np.array([0.5, 0.5]), quadratic([0.3, 0.7]), **kw)
    _, u2, _ = optimizeLinIneqConstrained(np.array([0.5, 0.5]),
                                          quadratic([0.3, 0.7]), **kw)
    assert np.allclose(u1, u2, atol=1e-4)


def test_lbfgsb_history_grows_by_one_per_iteration():
    _, _, h = L_BFGS_B(np.array([0.9, 0.1]), quadratic([0.3, 0.7]),
                       maximize=False, maxIt=3, objChangeTol=0.0,
                       gradTol=0.0, verbose=False)
    assert len(h['val']) == 4          # the initial entry plus three


def test_lbfgsb_saves_history_when_asked(tmp_path):
    path = str(tmp_path / 'hist.npz')
    L_BFGS_B(np.array([0.5, 0.5]), quadratic([0.3, 0.7]), maximize=False,
             maxIt=2, saveHistory=path, verbose=False)
    assert np.load(path)['val'].size >= 2


# ------------------------------------------------------- checkFeasible --

def _c(u):
    return getConstraints(u, {'linIneq': None, 'linEq': None})


def test_a_feasible_point_passes_unchanged():
    u = np.array([0.5, 0.5])
    out, flag, _ = checkFeasible(u.copy(), _c(u), enforce=True)
    assert flag is True and np.allclose(out, u)


def test_a_violating_point_is_pulled_back_inside():
    u = np.array([1.3, 0.5])
    out, flag, fixed = checkFeasible(u, _c(u), enforce=True)
    assert flag is False and fixed is True
    assert out[0] <= 1.0 + 1e-8


def test_without_enforcement_the_point_is_only_reported():
    u = np.array([1.3, 0.5])
    out, flag, fixed = checkFeasible(u.copy(), _c(u), enforce=False)
    assert flag is False and fixed is False
    assert np.allclose(out, u)          # untouched


def test_a_general_inequality_is_also_repaired():
    lin = {'A': sp.csr_matrix(np.array([[1.0, 1.0]])), 'b': np.array([1.0])}
    u = np.array([0.9, 0.9])
    c = getConstraints(u, {'linIneq': lin, 'linEq': None})
    out, flag, _ = checkFeasible(u, c, enforce=True)
    assert flag is False
    assert out.sum() <= 1.0 + 1e-6


# ------------------------------------------ unitBoxBFGSForEclipseRun --

def test_bfgs_run_finds_the_minimum(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'CASE.DATA').write_text('RUNSPEC\n')
    _, u, _ = unitBoxBFGSForEclipseRun(
        np.array([0.5, 0.5]), lambda u, d: quadratic([0.3, 0.7])(u),
        {'base': str(base), 'work': str(tmp_path / 'work')},
        maximize=False, gradTol=1e-9, objChangeTol=1e-14, maxIt=40,
        verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-3)


def test_bfgs_run_copies_the_base_case_for_each_iteration(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'CASE.DATA').write_text('RUNSPEC\n')
    seen = []

    def f(u, case_dir):
        seen.append(case_dir)
        assert os.path.exists(os.path.join(case_dir, 'CASE.DATA'))
        return quadratic([0.3, 0.7])(u)

    unitBoxBFGSForEclipseRun(
        np.array([0.5, 0.5]), f, {'base': str(base),
                                  'work': str(tmp_path / 'work')},
        maximize=False, maxIt=3, verbose=False)
    assert len({os.path.basename(d) for d in seen}) > 1


def test_bfgs_run_honours_a_linear_inequality(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    lin = {'A': sp.csr_matrix(np.array([[1.0, 1.0]])), 'b': np.array([1.0])}
    _, u, _ = unitBoxBFGSForEclipseRun(
        np.array([0.25, 0.25]), lambda u, d: quadratic([0.9, 0.9])(u),
        {'base': str(base), 'work': str(tmp_path / 'work')},
        linIneq=lin, maximize=False, gradTol=1e-9, objChangeTol=1e-14,
        maxIt=40, verbose=False)
    assert u.sum() <= 1.0 + 1e-6
    assert np.allclose(u, [0.5, 0.5], atol=1e-2)


def test_bfgs_run_works_with_a_dense_hessian(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    _, u, _ = unitBoxBFGSForEclipseRun(
        np.array([0.5, 0.5]), lambda u, d: quadratic([0.3, 0.7])(u),
        {'base': str(base), 'work': str(tmp_path / 'work')},
        limitedMemory=False, maximize=False, gradTol=1e-9,
        objChangeTol=1e-14, maxIt=40, verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-3)


def test_bfgs_run_without_the_bfgs_update_is_steepest_descent(tmp_path):
    """Still converges, just more slowly -- so it must not diverge."""
    base = tmp_path / 'base'
    base.mkdir()
    _, _, h = unitBoxBFGSForEclipseRun(
        np.array([0.9, 0.1]), lambda u, d: quadratic([0.3, 0.7])(u),
        {'base': str(base), 'work': str(tmp_path / 'work')},
        useBFGS=False, maximize=False, maxIt=8, objChangeTol=0.0,
        gradTol=0.0, verbose=False)
    vals = np.asarray(h['val'], dtype=float)
    assert vals[-1] <= vals[0]


def test_bfgs_run_checkpoints_its_history(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    work = tmp_path / 'work'
    unitBoxBFGSForEclipseRun(
        np.array([0.5, 0.5]), lambda u, d: quadratic([0.3, 0.7])(u),
        {'base': str(base), 'work': str(work)}, maximize=False, maxIt=2,
        verbose=False)
    assert np.load(work / 'history.npz')['val'].size >= 2
