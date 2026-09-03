"""Tests for DiagonalADI (DiagonalAutoDiffBackend.m performance-idea port):
correctness against SparseADI on an equivalent computation chain (typical of
a PVT/relperm/mobility evaluation), then a speed benchmark demonstrating the
actual speedup this is for.
"""

from __future__ import annotations

import time

import numpy as np

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.diagonal_adi import DiagonalADI


def _pvt_like_chain(adi_cls, p, sw):
    """A representative chain: mobility ~ (sw^2 / mu(p)) * exp(-c*(p-p0)),
    plus an upstream-gather (mimicking a TPFA flux's cell[c1]/cell[c2]
    indexing) and a division -- exercises every operator DiagonalADI
    implements."""
    p0 = 1.0e7
    c = 1.0e-9
    mu = 1.0e-3 + 2.0e-4 * (p - p0) / p0
    kr = sw**2
    lam = kr / mu
    comp = (-(c * (p - p0))).exp() if hasattr(p, "exp") else np.exp(-(c * (p - p0)))
    lam_c = lam * comp
    n = p.val.size if hasattr(p, "val") else len(p)
    idx = np.roll(np.arange(n), 1)
    gathered = lam_c[idx]
    return gathered / (1.0 + gathered)


def test_diagonal_adi_matches_sparse_adi_on_a_pvt_like_chain():
    rng = np.random.default_rng(0)
    n = 500
    p_val = 1.0e7 + rng.uniform(-1e6, 1e6, n)
    sw_val = rng.uniform(0.1, 0.9, n)
    nvar = 2 * n

    p_s = SparseADI.variable(p_val, nvar, 0)
    sw_s = SparseADI.variable(sw_val, nvar, n)
    result_sparse = _pvt_like_chain(SparseADI, p_s, sw_s)

    p_d = DiagonalADI.variable(p_val, nvar, 0)
    sw_d = DiagonalADI.variable(sw_val, nvar, n)
    result_diag = _pvt_like_chain(DiagonalADI, p_d, sw_d).to_sparse()

    assert np.allclose(result_diag.val, result_sparse.val, rtol=1e-12)
    diff = (result_diag.jac - result_sparse.jac)
    assert np.max(np.abs(diff.toarray())) < 1e-8


def test_diagonal_adi_matches_sparse_adi_through_linear_map_fallback():
    """linear_map falls back to materializing a SparseADI -- check that
    fallback produces the same result as doing the whole chain in SparseADI."""
    rng = np.random.default_rng(1)
    n = 50
    nvar = n
    x_val = rng.uniform(1.0, 5.0, n)
    M = np.eye(n) * 2.0 + np.eye(n, k=1) * 0.5

    x_s = SparseADI.variable(x_val, nvar, 0)
    y_s = (x_s * x_s).linear_map(M)

    x_d = DiagonalADI.variable(x_val, nvar, 0)
    y_d = (x_d * x_d).linear_map(M)

    assert np.allclose(y_d.val, y_s.val, rtol=1e-12)
    assert np.max(np.abs((y_d.jac - y_s.jac).toarray())) < 1e-10


def test_diagonal_adi_constant_and_getitem_are_correct():
    nvar = 6
    x = DiagonalADI.variable(np.array([1.0, 2.0, 3.0]), nvar, 0)
    c = DiagonalADI.constant(np.array([10.0, 20.0, 30.0]), nvar)
    y = (x + c)[::-1]
    assert np.allclose(y.val, [33.0, 22.0, 11.0])
    ys = y.to_sparse()
    # d(y[0])/dx[2] should be 1 (y[0] came from x[2]+10), others zero in that row.
    assert np.isclose(ys.jac[0, 2], 1.0)
    assert np.isclose(ys.jac[0, 0], 0.0)


def test_diagonal_adi_is_faster_than_sparse_adi_for_long_elementwise_chains():
    n = 200_000
    rng = np.random.default_rng(2)
    p_val = 1.0e7 + rng.uniform(-1e6, 1e6, n)
    sw_val = rng.uniform(0.1, 0.9, n)
    nvar = 2 * n

    p_s = SparseADI.variable(p_val, nvar, 0)
    sw_s = SparseADI.variable(sw_val, nvar, n)
    t0 = time.perf_counter()
    _pvt_like_chain(SparseADI, p_s, sw_s)
    t_sparse = time.perf_counter() - t0

    p_d = DiagonalADI.variable(p_val, nvar, 0)
    sw_d = DiagonalADI.variable(sw_val, nvar, n)
    t0 = time.perf_counter()
    _pvt_like_chain(DiagonalADI, p_d, sw_d)
    t_diag = time.perf_counter() - t0

    print(f"\nn={n}: SparseADI={t_sparse*1e3:.1f}ms  DiagonalADI={t_diag*1e3:.1f}ms  speedup={t_sparse/t_diag:.1f}x")
    assert t_diag < t_sparse
