"""Interpolating many small tables at once must equal doing it one at a time.

PVTO and PVTG hold a list of records; each cell picks one and interpolates
in it.  The straightforward implementation loops over the records and
scatters each one's rows into a grid-sized result -- forty-one iterations on
Norne, four grid-sized AD values apiece.  ``_RaggedTables`` replaces that
with one ``searchsorted`` into the records laid end to end, separated by an
offset wide enough that they cannot overlap.

The offset trick is the part that can be wrong in ways that look right, so
the tests push on it: a query below a record's first point, above its last,
exactly on a knot, in a record given in descending order, and in a record
holding a single point.  The reference throughout is
``ad_interp_linear`` -- the routine the loop called.
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
from PRSTCore.ad_core.adi import SparseADI, ad_interp_linear
from PRSTCore.ad_core.initialization.pvt_tables import _interp_ragged, _RaggedTables


def _reference(tables, record, query_ad):
    """What the per-record loop produced: one interpolation per record."""
    n = query_ad.val.size
    out = SparseADI.constant(np.zeros(n), query_ad.nvar)
    for index, (x, y) in enumerate(tables):
        rows = np.flatnonzero(np.asarray(record) == index)
        if rows.size:
            out = out + SparseADI.scatter(
                rows, ad_interp_linear(x, y, query_ad[rows]), n)
    return out


def _compare(tables, record, queries, seed=0):
    record = np.asarray(record, dtype=int)
    queries = np.asarray(queries, dtype=float)
    n = queries.size
    query_ad = SparseADI.variable(queries, n, 0)

    got = _interp_ragged(_RaggedTables(tables), record, query_ad)
    want = _reference(tables, record, query_ad)

    np.testing.assert_allclose(got.val, want.val, rtol=1e-13, atol=1e-13)
    difference = (got.jac - want.jac).tocoo()
    if difference.nnz:
        largest = max(float(abs(want.jac).max()), 1.0)
        assert np.abs(difference.data).max() <= 1e-13 * largest, (
            'derivatives differ by %g' % np.abs(difference.data).max())


def test_two_ordinary_records():
    tables = [(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.8, 0.7])),
              (np.array([0.0, 1.5, 3.0]), np.array([1.2, 0.9, 0.6]))]
    queries = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 0.5])
    record = np.array([0, 0, 1, 1, 1, 0])
    _compare(tables, record, queries)


def test_queries_outside_a_record_extrapolate_the_end_segments():
    """Below the first knot and above the last, both records.

    ``ad_interp_linear`` clips the bin, so it continues the end segment's
    slope rather than flattening.  The flattened lookup has to clip into the
    *same* record, not slide into its neighbour -- which is the failure the
    offset exists to prevent.
    """
    tables = [(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.8, 0.7])),
              (np.array([0.0, 1.5, 3.0]), np.array([1.2, 0.9, 0.6]))]
    queries = np.array([-5.0, -0.1, 2.1, 10.0, -3.0, 7.5])
    record = np.array([0, 0, 0, 0, 1, 1])
    _compare(tables, record, queries)


def test_queries_exactly_on_the_knots():
    tables = [(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.8, 0.7])),
              (np.array([0.0, 2.0, 4.0]), np.array([1.3, 1.0, 0.8]))]
    queries = np.array([0.0, 1.0, 2.0, 0.0, 2.0, 4.0])
    record = np.array([0, 0, 0, 1, 1, 1])
    _compare(tables, record, queries)


def test_records_of_different_lengths():
    tables = [(np.array([0.0, 1.0]), np.array([1.0, 0.9])),
              (np.array([0.0, 0.5, 1.0, 2.0, 5.0]),
               np.array([1.2, 1.1, 1.0, 0.9, 0.7])),
              (np.array([0.0, 3.0, 6.0]), np.array([1.4, 1.0, 0.8]))]
    rng = np.random.default_rng(3)
    queries = rng.uniform(-1.0, 7.0, 40)
    record = rng.integers(0, len(tables), 40)
    _compare(tables, record, queries)


def test_a_record_holding_a_single_point():
    """``ad_interp_linear`` returns that point with no derivative."""
    tables = [(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.8, 0.7])),
              (np.array([1.5]), np.array([0.95]))]
    queries = np.array([0.5, 1.5, 0.0, 99.0])
    record = np.array([0, 1, 1, 1])
    _compare(tables, record, queries)


def test_a_record_given_in_descending_order():
    """The loop sorted each table; so must the flattened form."""
    tables = [(np.array([2.0, 1.0, 0.0]), np.array([0.7, 0.8, 1.0])),
              (np.array([0.0, 1.5, 3.0]), np.array([1.2, 0.9, 0.6]))]
    queries = np.array([0.3, 1.4, 1.9, 2.2])
    record = np.array([0, 0, 1, 0])
    _compare(tables, record, queries)


def test_many_records_at_norne_scale():
    """Forty-one records, which is what a real deck presents."""
    rng = np.random.default_rng(7)
    tables = []
    for _ in range(41):
        knots = np.sort(rng.uniform(0.0, 300.0, rng.integers(3, 12)))
        knots = knots - knots[0]
        tables.append((knots, rng.uniform(0.5, 1.5, knots.size)))
    n = 5000
    record = rng.integers(0, len(tables), n)
    queries = rng.uniform(-20.0, 320.0, n)
    _compare(tables, record, queries)


def test_the_offset_separates_records_that_share_a_range():
    """Every record spans the same numbers -- the case an offset must fix.

    Without the per-record stride a query would find a segment belonging to
    whichever record happened to sit next to it in memory, and the answer
    would be plausible and wrong.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    tables = [(x, x * 1.0), (x, x * 10.0), (x, x * 100.0)]
    queries = np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5])
    record = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    _compare(tables, record, queries)
    # And the values really are a hundred apart, so a mix-up could not hide.
    flat = _RaggedTables(tables)
    query_ad = SparseADI.variable(queries, queries.size, 0)
    got = _interp_ragged(flat, record, query_ad)
    np.testing.assert_allclose(got.val[:3] * 100.0, got.val[6:], rtol=1e-12)
