"""Parity tests for PRSTCore's SparseADI against MRST's ``core/utils/ADI.m``.

Derivatives are checked against forward finite differences; the branch
semantics (upwind-style selection, max/min tie-breaking) are checked
against the exact rules ADI.m implements, since those are what decide
which operand contributes a derivative when two values coincide.
"""

import numpy as np
import pytest

from PRSTCore.ad_core.adi import (SparseADI, ad_abs, ad_maximum, ad_minimum,
                                  ad_select)


def _fd(f, x0, h=1e-7):
    """Forward-difference Jacobian of ``f`` at ``x0``."""
    base = f(x0)
    cols = [(f(x0 + h * np.eye(x0.size)[i]) - base) / h for i in range(x0.size)]
    return np.array(cols).T


def _var(x0):
    return SparseADI.variable(x0, x0.size, 0)


def _assert_jac(result, f, x0, tol=1e-5):
    assert np.abs(result.jac.toarray() - _fd(f, x0)).max() < tol


X0 = np.array([2.0, 3.0, 1.5])


@pytest.mark.parametrize('exponent', [2.5, 3.0, 0.5, -1.0])
def test_power_scalar_exponent(exponent):
    _assert_jac(_var(X0) ** exponent, lambda v: v ** exponent, X0)


def test_power_vector_exponent():
    """ADI.m's power accepts a vector exponent, not only a scalar."""
    e = np.array([2.0, 3.0, 0.5])
    _assert_jac(_var(X0) ** e, lambda v: v ** e, X0)


def test_power_numeric_base_adi_exponent():
    """``u.^v`` with numeric u: d = u^v * log(u) dv."""
    _assert_jac(2.0 ** _var(X0), lambda v: 2.0 ** v, X0)


def test_power_both_adi():
    """``u.^v`` with both ADI: d = u^v*(v/u) du + u^v*log(u) dv."""
    u = _var(X0)
    _assert_jac(u ** u, lambda v: v ** v, X0, tol=1e-4)


def test_max_against_numeric_keeps_derivative_on_ties():
    """ADI.m routes max(ADI, double) through max(double, ADI), whose flag is
    ``~(double > adi.val)`` -- so equality keeps the ADI derivative."""
    a = _var(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(ad_maximum(a, 2.0).jac.toarray().diagonal(), [0.0, 1.0, 1.0])


def test_max_numeric_left_matches_numeric_right():
    a = _var(np.array([1.0, 2.0, 3.0]))
    left = ad_maximum(2.0, a).jac.toarray()
    right = ad_maximum(a, 2.0).jac.toarray()
    assert np.allclose(left, right)


def test_max_both_adi_gives_tie_to_right_operand():
    """The both-ADI branch is ``inx = u.val > v.val``, so a tie picks v."""
    u = _var(np.array([1.0, 2.0, 3.0]))
    v = SparseADI(np.array([2.0, 2.0, 2.0]), u.jac * 2.0)
    jac = ad_maximum(u, v).jac.toarray().diagonal()
    assert np.allclose(jac, [2.0, 2.0, 1.0])


def test_min_is_negated_max():
    a = _var(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(ad_minimum(a, 2.0).val, [1.0, 2.0, 2.0])


def test_abs_uses_sign_derivative():
    x0 = np.array([-2.0, 3.0])
    r = ad_abs(_var(x0))
    assert np.allclose(r.val, [2.0, 3.0])
    assert np.allclose(r.jac.toarray().diagonal(), [-1.0, 1.0])


@pytest.mark.parametrize('op, fn', [
    ('mul', lambda v: v * v),
    ('div', lambda v: v / (v + 1.0)),
    ('exp', np.exp),
    ('log', np.log),
])
def test_elementary_operations(op, fn):
    u = _var(X0)
    result = {'mul': u * u, 'div': u / (u + 1.0),
              'exp': u.exp(), 'log': u.log()}[op]
    _assert_jac(result, fn, X0)


def test_select_takes_branchwise_derivative():
    u = _var(X0)
    v = SparseADI(X0 * 2.0, u.jac * 3.0)
    mask = np.array([True, False, True])
    jac = ad_select(mask, u, v).jac.toarray().diagonal()
    assert np.allclose(jac, [1.0, 3.0, 1.0])


def test_linear_map_matches_matrix_product():
    """ADI.m mtimes with a numeric left matrix: value and Jacobian both map."""
    import scipy.sparse as sp
    u = _var(X0)
    M = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]))
    r = u.linear_map(M)
    assert np.allclose(r.val, M @ X0)
    assert np.allclose(r.jac.toarray(), (M @ u.jac).toarray())
