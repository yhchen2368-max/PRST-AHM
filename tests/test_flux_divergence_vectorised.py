"""The vectorised flux divergence must reproduce the per-face loop exactly.

``_assemble_flux_divergence`` used to walk the faces one at a time in
Python.  Replacing that with array operations is only worth anything if the
numbers are unchanged, and the places it is easy to get wrong are all edge
cases: the one-based neighbour table, faces naming inactive cells, the
upstream tie at zero pressure difference, and the ``!= 0`` filter that keeps
zero conductivities out of the assembled triplets.

The original loop is kept here as the oracle.  Comparing against a
transcription rather than against stored numbers means the test still says
something if the operators or the PVT contract change.
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
from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel


class _Stub:
    """The two things ``_assemble_flux_divergence`` reads off the model."""

    def __init__(self, nc, operators):
        self._nc = nc
        self.operators = operators

    def _num_cells(self):
        return self._nc

    _assemble_flux_divergence = GenericBlackOilModel._assemble_flux_divergence


def _loop_reference(nc, operators, p, lamW, lamO, lamG, pvt):
    """The original per-face implementation, verbatim."""
    div_w = np.zeros((nc,), dtype=float)
    div_o = np.zeros((nc,), dtype=float)
    div_g = np.zeros((nc,), dtype=float)
    Lw_rows, Lw_cols, Lw_vals = [], [], []
    Lo_rows, Lo_cols, Lo_vals = [], [], []
    Lg_rows, Lg_cols, Lg_vals = [], [], []

    N = np.asarray(operators.get('N', np.zeros((0, 2), dtype=int)), dtype=int)
    T = np.asarray(operators.get('T', np.zeros((0,), dtype=float)), dtype=float).ravel()
    if N.size == 0 or T.size == 0:
        return (div_w, div_o, div_g,
                (Lw_rows, Lw_cols, Lw_vals),
                (Lo_rows, Lo_cols, Lo_vals),
                (Lg_rows, Lg_cols, Lg_vals))

    for f in range(min(N.shape[0], T.size)):
        c1 = int(N[f, 0]) - 1
        c2 = int(N[f, 1]) - 1
        if c1 < 0 or c2 < 0 or c1 >= nc or c2 >= nc:
            continue
        dp = float(p[c1] - p[c2])
        up = c1 if dp >= 0 else c2
        tf = float(T[f])
        bw_up = float(pvt['bw'][up]) if pvt is not None else 1.0
        bo_up = float(pvt['bo'][up]) if pvt is not None else 1.0
        bg_up = float(pvt['bg'][up]) if pvt is not None else 1.0
        gw = tf * float(lamW[up]) * bw_up
        go = tf * float(lamO[up]) * bo_up
        gg = tf * float(lamG[up]) * bg_up
        fw = gw * dp
        fo = go * dp
        fg = gg * dp
        div_w[c1] += fw
        div_w[c2] -= fw
        div_o[c1] += fo
        div_o[c2] -= fo
        div_g[c1] += fg
        div_g[c2] -= fg

        if gw != 0.0:
            Lw_rows.extend([c1, c1, c2, c2])
            Lw_cols.extend([c1, c2, c1, c2])
            Lw_vals.extend([gw, -gw, -gw, gw])
        if go != 0.0:
            Lo_rows.extend([c1, c1, c2, c2])
            Lo_cols.extend([c1, c2, c1, c2])
            Lo_vals.extend([go, -go, -go, go])
        if gg != 0.0:
            Lg_rows.extend([c1, c1, c2, c2])
            Lg_cols.extend([c1, c2, c1, c2])
            Lg_vals.extend([gg, -gg, -gg, gg])

    return (div_w, div_o, div_g,
            (Lw_rows, Lw_cols, Lw_vals),
            (Lo_rows, Lo_cols, Lo_vals),
            (Lg_rows, Lg_cols, Lg_vals))


def _matrix(triplet, nc):
    """Triplets compared the way the caller uses them: as an assembled matrix.

    Ordering within the triplet is not part of the contract -- the caller
    hands it to ``csr_matrix``, which sums duplicates -- so comparing the
    dense result is the comparison that matters.
    """
    import scipy.sparse as sp

    rows, cols, vals = triplet
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    vals = np.asarray(vals, dtype=float)
    if rows.size == 0:
        return np.zeros((nc, nc))
    return sp.csr_matrix((vals, (rows, cols)), shape=(nc, nc)).toarray()


def _cartesian_faces(nx, ny, nz):
    """A one-based TPFA neighbour table for an nx-by-ny-by-nz box."""
    index = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    pairs = []
    for a, b in ((index[:-1].ravel(), index[1:].ravel()),
                 (index[:, :-1].ravel(), index[:, 1:].ravel()),
                 (index[:, :, :-1].ravel(), index[:, :, 1:].ravel())):
        pairs.append(np.stack([a + 1, b + 1], axis=1))
    return np.concatenate(pairs, axis=0)


def _case(seed, nx=5, ny=4, nz=3, with_pvt=True, corrupt_neighbours=False,
          zero_mobility=False, tied_pressure=False, extra_faces=0):
    rng = np.random.default_rng(seed)
    nc = nx * ny * nz
    N = _cartesian_faces(nx, ny, nz)
    if corrupt_neighbours:
        # Faces naming a cell outside the active set: the loop skips them.
        N = N.copy()
        N[0, 0] = 0            # one-based zero -> index -1
        N[1, 1] = nc + 5       # past the end
    T = rng.uniform(0.5, 2.0, N.shape[0])
    if extra_faces:
        # More neighbour rows than transmissibilities: only the first
        # ``T.size`` faces are considered.
        N = np.concatenate([N, N[:extra_faces]], axis=0)

    p = rng.uniform(200.0, 300.0, nc)
    if tied_pressure:
        # Force exact ties so the dp == 0 upstream tie-break is exercised.
        p[:] = 250.0
    lamW = rng.uniform(0.1, 1.0, nc)
    lamO = rng.uniform(0.1, 1.0, nc)
    lamG = rng.uniform(0.1, 1.0, nc)
    if zero_mobility:
        lamG[:] = 0.0
        lamW[rng.integers(0, nc, nc // 3)] = 0.0
    pvt = None
    if with_pvt:
        pvt = {'bw': rng.uniform(0.8, 1.2, nc),
               'bo': rng.uniform(0.8, 1.2, nc),
               'bg': rng.uniform(0.8, 1.2, nc)}
    return nc, {'N': N, 'T': T}, p, lamW, lamO, lamG, pvt


CASES = {
    'plain': dict(seed=1),
    'no_pvt': dict(seed=2, with_pvt=False),
    'inactive_neighbours': dict(seed=3, corrupt_neighbours=True),
    'zero_mobility': dict(seed=4, zero_mobility=True),
    'tied_pressure': dict(seed=5, tied_pressure=True),
    'more_faces_than_trans': dict(seed=6, extra_faces=7),
    'single_cell_column': dict(seed=7, nx=1, ny=1, nz=6),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_matches_the_per_face_loop(name):
    nc, operators, p, lamW, lamO, lamG, pvt = _case(**CASES[name])
    stub = _Stub(nc, operators)

    got = stub._assemble_flux_divergence(p, lamW, lamO, lamG, pvt=pvt)
    want = _loop_reference(nc, operators, p, lamW, lamO, lamG, pvt)

    for phase, index in (('water', 0), ('oil', 1), ('gas', 2)):
        np.testing.assert_allclose(
            got[index], want[index], rtol=0, atol=1e-12,
            err_msg='%s divergence differs' % phase)

    for phase, index in (('water', 3), ('oil', 4), ('gas', 5)):
        np.testing.assert_allclose(
            _matrix(got[index], nc), _matrix(want[index], nc),
            rtol=0, atol=1e-12,
            err_msg='%s pressure block differs' % phase)


def test_no_faces_returns_independent_zero_vectors():
    """The three divergences must not alias -- callers add into them."""
    stub = _Stub(4, {'N': np.zeros((0, 2), dtype=int), 'T': np.zeros(0)})
    div_w, div_o, div_g, lw, lo, lg = stub._assemble_flux_divergence(
        np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), pvt=None)
    div_w[0] = 1.0
    assert div_o[0] == 0.0 and div_g[0] == 0.0
    for triplet in (lw, lo, lg):
        assert all(np.asarray(part).size == 0 for part in triplet)
