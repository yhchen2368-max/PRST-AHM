"""Tests for ``optimizeLinIneqConstrained`` and
``optimizeBoundConstrainedForFAHM``.

The problems here have answers that can be written down, so the tests
check the optimiser against the answer rather than against itself.
"""

import os

import numpy as np
import pytest
import scipy.sparse as sp

from PRSTCore.hm.utils.optimizer.optimizeBoundConstrainedForFAHM import \
    optimizeBoundConstrainedForFAHM
from PRSTCore.hm.utils.optimizer.optimizeLinIneqConstrained import (
    classifyConstraints, expandQ, findNextConstraint, getConstraints, hpsolb,
    optimizeLinIneqConstrained, projQ, updateTrustRegion)


def quadratic(target):
    """``|u - target|^2``, whose unconstrained minimum is ``target``."""
    target = np.asarray(target, dtype=float)

    def f(u):
        u = np.asarray(u, dtype=float).ravel()
        return float(np.sum((u - target) ** 2)), 2.0 * (u - target)
    return f


# ---------------------------------------------------------- helpers ----

def test_box_constraints_are_stated_as_A_u_le_b():
    c = getConstraints(np.zeros(2), {'linIneq': None, 'linEq': None})
    A = c['i']['A'].toarray()
    # -u <= 0 is u >= 0; u <= 1 is the upper bound.
    assert np.allclose(A[:2], -np.eye(2))
    assert np.allclose(A[2:], np.eye(2))
    assert np.allclose(c['i']['b'], [0, 0, 1, 1])


def test_supplied_inequalities_are_appended_and_scaled():
    lin = {'A': sp.csr_matrix(np.array([[2.0, 0.0]])), 'b': np.array([4.0])}
    c = getConstraints(np.zeros(2), {'linIneq': lin, 'linEq': None})
    assert c['i']['A'].shape == (5, 2)
    # Scaled by norm(A) == 2, so the row and its rhs are halved.
    assert np.allclose(c['i']['A'].toarray()[-1], [1.0, 0.0])
    assert c['i']['b'][-1] == pytest.approx(2.0)


def test_classify_marks_a_constraint_active_only_when_heading_out():
    A = sp.csr_matrix(np.array([[1.0, 0.0]]))
    b = np.array([1.0])
    _, out = classifyConstraints(A, b, np.array([1.0, 0.0]),
                                 np.array([1.0, 0.0]))
    _, into = classifyConstraints(A, b, np.array([1.0, 0.0]),
                                  np.array([-1.0, 0.0]))
    assert bool(out[0]) and not bool(into[0])


def test_find_next_constraint_returns_the_nearest_one():
    A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    b = np.array([1.0, 0.5])
    ix, s = findNextConstraint(A, b, np.zeros(2), np.array([1.0, 1.0]),
                               np.zeros(2, dtype=bool))
    assert ix == 1 and s == pytest.approx(0.5)


def test_find_next_constraint_reports_none_when_the_ray_escapes():
    A = sp.csr_matrix(np.array([[1.0, 0.0]]))
    ix, s = findNextConstraint(A, np.array([1.0]), np.zeros(2),
                               np.array([-1.0, 0.0]), np.zeros(1, dtype=bool))
    assert ix is None and not np.isfinite(s)


def test_projection_removes_the_component_along_Q():
    Q = np.array([[1.0], [0.0]])
    assert np.allclose(projQ(np.array([3.0, 4.0]), Q), [0.0, 4.0])


def test_projection_is_the_identity_without_constraints():
    v = np.array([3.0, 4.0])
    assert np.allclose(projQ(v, np.zeros((2, 0))), v)


def test_expandQ_appends_an_orthonormal_direction():
    Q = expandQ(np.zeros((2, 0)), np.array([2.0, 0.0]))
    Q = expandQ(Q, np.array([1.0, 5.0]))
    assert Q.shape == (2, 2)
    assert np.allclose(Q.T @ Q, np.eye(2))


def test_expandQ_refuses_a_dependent_direction(capsys):
    Q = expandQ(np.zeros((2, 0)), np.array([1.0, 0.0]))
    out = expandQ(Q, np.array([3.0, 0.0]))
    assert out.shape == (2, 1)
    assert 'linear combination' in capsys.readouterr().out


def test_hpsolb_moves_the_smallest_breakpoint_to_the_end():
    t = np.array([5.0, 1.0, 3.0, 9.0])
    iorder = np.arange(4)
    t, iorder = hpsolb(4, t.copy(), iorder.copy(), 0)
    assert t[3] == pytest.approx(1.0)
    assert iorder[3] == 1


def test_trust_region_shrinks_on_a_poor_model_fit():
    opt = {'ratioThresholds': (0.25, 0.75), 'radiusDecrease': 0.25,
           'radiusIncrease': 2.0}
    assert updateTrustRegion(1.0, 0.0, 1.0, 1.0, opt) == pytest.approx(0.25)


def test_trust_region_grows_on_a_good_one():
    opt = {'ratioThresholds': (0.25, 0.75), 'radiusDecrease': 0.25,
           'radiusIncrease': 2.0}
    assert updateTrustRegion(1.0, 0.9, 2.0, 1.0, opt) == pytest.approx(2.0)


# -------------------------------------------------------- optimiser ----

def test_finds_an_interior_minimum():
    v, u, _ = optimizeLinIneqConstrained(
        np.array([0.5, 0.5]), quadratic([0.3, 0.7]), maximize=False,
        gradTol=1e-8, objChangeTol=1e-12, maxIt=50, verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-4)
    assert v == pytest.approx(0.0, abs=1e-8)


def test_stays_inside_the_unit_box():
    """The unconstrained minimum is outside; the answer must be the
    projection of it onto the box."""
    _, u, _ = optimizeLinIneqConstrained(
        np.array([0.5, 0.5]), quadratic([-0.5, 1.5]), maximize=False,
        gradTol=1e-8, objChangeTol=1e-12, maxIt=50, verbose=False)
    assert np.all(u >= -1e-9) and np.all(u <= 1 + 1e-9)
    assert u[0] == pytest.approx(0.0, abs=1e-6)
    assert u[1] == pytest.approx(1.0, abs=1e-6)


def test_honours_a_linear_inequality():
    """min |u - (0.9, 0.9)|^2 subject to u0 + u1 <= 1.

    The answer is the projection of (0.9, 0.9) onto that line, (0.5, 0.5).
    """
    lin = {'A': sp.csr_matrix(np.array([[1.0, 1.0]])), 'b': np.array([1.0])}
    _, u, _ = optimizeLinIneqConstrained(
        np.array([0.25, 0.25]), quadratic([0.9, 0.9]), linIneq=lin,
        maximize=False, gradTol=1e-8, objChangeTol=1e-12, maxIt=50,
        verbose=False)
    assert u.sum() <= 1 + 1e-6
    assert np.allclose(u, [0.5, 0.5], atol=1e-3)


def test_an_inactive_constraint_does_not_move_the_answer():
    lin = {'A': sp.csr_matrix(np.array([[1.0, 1.0]])), 'b': np.array([10.0])}
    _, u, _ = optimizeLinIneqConstrained(
        np.array([0.5, 0.5]), quadratic([0.3, 0.4]), linIneq=lin,
        maximize=False, gradTol=1e-8, objChangeTol=1e-12, maxIt=50,
        verbose=False)
    assert np.allclose(u, [0.3, 0.4], atol=1e-4)


def test_maximizing_flips_which_corner_is_chosen():
    """The same objective, maximised, runs to the opposite bound."""
    _, u, _ = optimizeLinIneqConstrained(
        np.array([0.5, 0.5]), quadratic([0.0, 0.0]), maximize=True,
        gradTol=1e-8, objChangeTol=1e-12, maxIt=50, verbose=False)
    assert np.allclose(u, [1.0, 1.0], atol=1e-3)


def test_history_records_every_iteration():
    _, _, h = optimizeLinIneqConstrained(
        np.array([0.5, 0.5]), quadratic([0.3, 0.7]), maximize=False,
        maxIt=3, verbose=False)
    assert len(h['val']) == len(h['u']) == len(h['pg'])
    assert np.all(np.isfinite(h['val']))


def test_the_objective_decreases_monotonically():
    _, _, h = optimizeLinIneqConstrained(
        np.array([0.9, 0.1]), quadratic([0.3, 0.7]), maximize=False,
        maxIt=10, verbose=False)
    vals = np.asarray(h['val'], dtype=float)
    assert np.all(np.diff(vals) <= 1e-12)


def test_scaled_bounds_are_mapped_back_to_physical_units():
    """With lb/ub given, the returned u must be in physical units."""
    _, u, _ = optimizeLinIneqConstrained(
        np.array([5.0]), quadratic([3.0]), lb=np.array([0.0]),
        ub=np.array([10.0]), maximize=False, gradTol=1e-10,
        objChangeTol=1e-14, maxIt=50, verbose=False)
    assert u[0] == pytest.approx(3.0, abs=1e-3)


def test_the_iteration_cap_is_respected():
    _, _, h = optimizeLinIneqConstrained(
        np.array([0.9, 0.1]), quadratic([0.3, 0.7]), maximize=False,
        maxIt=2, objChangeTol=0.0, gradTol=0.0, verbose=False)
    assert len(h['val']) <= 3          # initial entry plus two iterations


# ------------------------------------------------------------- FAHM ----

def test_fahm_gives_each_iteration_its_own_case_directory(tmp_path):
    seen = []

    def f(u, case_dir):
        seen.append(case_dir)
        assert os.path.isdir(case_dir), 'the directory must exist on call'
        v, g = quadratic([0.3, 0.7])(u)
        return v, g

    optimizeBoundConstrainedForFAHM(
        np.array([0.5, 0.5]), f, {'work': str(tmp_path)}, maximize=False,
        maxIt=3, verbose=False)
    assert len({os.path.basename(d) for d in seen}) > 1
    assert all(os.path.basename(d).startswith('case') for d in seen)


def test_fahm_checkpoints_the_history(tmp_path):
    optimizeBoundConstrainedForFAHM(
        np.array([0.5, 0.5]), lambda u, d: quadratic([0.3, 0.7])(u),
        {'work': str(tmp_path)}, maximize=False, maxIt=3, verbose=False)
    saved = np.load(tmp_path / 'history.npz')
    assert saved['val'].size >= 2


def test_fahm_minimizes_when_told_to(tmp_path):
    _, u, _ = optimizeBoundConstrainedForFAHM(
        np.array([0.5, 0.5]), lambda u, d: quadratic([0.3, 0.7])(u),
        {'work': str(tmp_path)}, maximize=False, gradTol=1e-8,
        objChangeTol=1e-12, maxIt=50, verbose=False)
    assert np.allclose(u, [0.3, 0.7], atol=1e-4)


def test_fahm_defaults_to_maximizing(tmp_path):
    """FAHM's default differs from every sibling in the module, so a
    caller minimising a mismatch must say so explicitly."""
    from PRSTCore.hm.utils.optimizer.optimizeBoundConstrainedForFAHM import \
        FAHM_DEFAULTS
    assert FAHM_DEFAULTS['maximize'] is True

    _, u, _ = optimizeBoundConstrainedForFAHM(
        np.array([0.5, 0.5]), lambda u, d: quadratic([0.0, 0.0])(u),
        {'work': str(tmp_path)}, gradTol=1e-8, objChangeTol=1e-12,
        maxIt=50, verbose=False)
    assert np.allclose(u, [1.0, 1.0], atol=1e-3)   # ran away, not towards
