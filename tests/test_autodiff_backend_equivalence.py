"""The two AD representations must assemble the same system.

``DiagonalAutoDiffBackend`` exists only to make a residual cheaper to build.
If it ever changes a number, everything downstream -- the Newton path, the
adjoint, every MRST parity result -- is affected without any of those tests
naming the backend as the cause.  So the guarantee under test is exact
agreement on a real deck, not agreement to some tolerance.

The Jacobian is allowed to differ by rounding, and only by rounding: the
diagonal form sums a row's contributions as arrays and the sparse form sums
them as matrix entries, which can reorder a few floating-point additions.
The residual is compared bit for bit, because nothing reorders there.
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
from PRSTCore.ad_core.adi import SparseADI, is_ad
from PRSTCore.ad_core.backends import (AutoDiffBackend, DiagonalAutoDiffBackend,
                                       get_backend)
from PRSTCore.ad_core.diagonal_adi import DiagonalADI

DECKS = {
    'spe1': REPO_ROOT / 'examples' / 'SpE1' / 'SPE1CASE1.DATA',
    'spe9': REPO_ROOT / 'examples' / 'SPE9' / 'SPE9_CP.DATA',
}


def _first_problem(deck, backend):
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)

    state0, model, schedule, _solver = init_eclipse_problem_ad(str(deck))
    model.autodiff_backend = backend
    dt = float(np.atleast_1d(schedule['step']['val'])[0])
    controls = schedule.get('control')
    forces = controls[0] if isinstance(controls, list) and controls else {}
    state = model.validateState(state0)
    problem, _ = model.get_equations(state, model.validateState(state0), dt, forces)
    return problem


@pytest.mark.parametrize('case', sorted(DECKS))
def test_backends_assemble_the_same_system(case):
    deck = DECKS[case]
    if not deck.is_file():
        pytest.skip('deck %s is not in this checkout' % deck)

    sparse = _first_problem(deck, 'sparse')
    diagonal = _first_problem(deck, 'diagonal')

    np.testing.assert_array_equal(
        sparse['Residuals'], diagonal['Residuals'],
        err_msg='the diagonal backend changed the residual')

    a, b = sparse['Jacobian'].tocsr(), diagonal['Jacobian'].tocsr()
    assert a.shape == b.shape
    difference = (a - b).tocoo()
    if difference.nnz:
        scale = max(abs(a).max(), 1.0)
        assert np.abs(difference.data).max() <= 1e-12 * scale, (
            'Jacobians differ by more than rounding: %g'
            % np.abs(difference.data).max())


def test_get_backend_accepts_name_class_instance_and_none():
    assert isinstance(get_backend(None), AutoDiffBackend)
    assert type(get_backend(None)) is AutoDiffBackend, 'sparse must stay the default'
    assert isinstance(get_backend('diagonal'), DiagonalAutoDiffBackend)
    assert isinstance(get_backend(DiagonalAutoDiffBackend), DiagonalAutoDiffBackend)
    instance = DiagonalAutoDiffBackend()
    assert get_backend(instance) is instance
    with pytest.raises(ValueError):
        get_backend('no-such-backend')
    with pytest.raises(TypeError):
        get_backend(42)


def test_backends_seed_their_own_representation():
    values = np.array([1.0, 2.0, 3.0])
    assert isinstance(AutoDiffBackend().variable(values, 3, 0), SparseADI)
    assert isinstance(DiagonalAutoDiffBackend().variable(values, 3, 0), DiagonalADI)
    assert is_ad(DiagonalAutoDiffBackend().constant(values, 3))


def test_mixed_representations_promote_to_sparse():
    """Where the two meet, the general one must win rather than raise.

    A residual adds a flux term that has been through ``linear_map`` -- so
    sparse -- to a well source term that has not.  Both orders must work.
    """
    n, nvar = 4, 4
    values = np.arange(1.0, n + 1.0)
    diagonal = DiagonalADI.variable(values, nvar, 0)
    sparse = SparseADI.variable(values, nvar, 0)

    for left, right in ((diagonal, sparse), (sparse, diagonal)):
        for combine in (lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a - b):
            out = combine(left, right)
            assert isinstance(out, SparseADI), 'mixing must produce a sparse value'
            reference = combine(sparse, sparse)
            np.testing.assert_allclose(out.val, reference.val, rtol=0, atol=0)
            np.testing.assert_allclose(out.jac.toarray(), reference.jac.toarray(),
                                       rtol=0, atol=1e-15)


def test_scatter_accumulates_repeated_cells():
    """Two perforations in one cell must add, in the value and the Jacobian.

    The batched well scatter relies on this; with a plain indexed assignment
    the value would keep only the last perforation while its derivative kept
    the sum.
    """
    nvar = 2
    part = SparseADI.variable(np.array([3.0, 5.0]), nvar, 0)
    out = SparseADI.scatter([1, 1], part, 4)
    np.testing.assert_allclose(out.val, [0.0, 8.0, 0.0, 0.0])
    np.testing.assert_allclose(out.jac.toarray(),
                               [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])


def test_diagonal_storage_does_not_grow_with_expression_depth():
    """Storage must track the primary variables, not the chain length.

    Three cell variables sharing one row map is one group holding an
    ``(n, 3)`` array -- the shape MRST's DiagonalJacobian keeps and the one
    the compiled product rule can take whole.  A chain of any depth over
    those three variables has to land back on it; if it does not, every
    later operation costs a pass per accumulated entry and ``to_sparse``
    pays for each one.
    """
    n, nvar = 32, 96
    rng = np.random.default_rng(0)
    p = DiagonalADI.variable(rng.uniform(1.0, 2.0, n), nvar, 0)
    sw = DiagonalADI.variable(rng.uniform(0.1, 0.4, n), nvar, n)
    sg = DiagonalADI.variable(rng.uniform(0.1, 0.3, n), nvar, 2 * n)

    accumulated = 1.0 - sw - sg
    for _ in range(25):
        accumulated = accumulated + (sw * sw) / (1.0 + p) + sg * p

    assert len(accumulated.groups) == 1, (
        'expected the three variables in one group, got %d groups'
        % len(accumulated.groups))
    group, deriv = accumulated.groups[0]
    assert group.offsets == (0, n, 2 * n)
    assert deriv.shape == (n, 3)
    # And it still has to be the right answer.
    reference = accumulated.to_sparse()
    assert reference.jac.shape == (n, nvar)


def test_the_dense_layout_agrees_with_a_finite_difference():
    """The shape is only worth anything if the derivatives are right.

    The rewrite moved every operator from one column at a time to a whole
    array at once, which is exactly the kind of change that can transpose,
    misalign or double a column and still produce a plausible matrix.  A
    central difference on each primary variable in turn is independent of
    all of it.
    """
    n = 12
    nvar = 3 * n
    rng = np.random.default_rng(11)
    values = [rng.uniform(1.0, 2.0, n), rng.uniform(0.1, 0.4, n),
              rng.uniform(0.1, 0.3, n)]

    def expression(raw):
        p, sw, sg = [DiagonalADI.variable(v, nvar, k * n)
                     for k, v in enumerate(raw)]
        so = 1.0 - sw - sg
        return (sw ** 2 * p / (1.0 + sg) + (-sw).exp() * so
                + sg / p - p.log() * so ** 3)

    jac = expression(values).to_sparse().jac.toarray()

    step = 1e-6
    for variable in range(3):
        for cell in range(n):
            up = [v.copy() for v in values]
            down = [v.copy() for v in values]
            up[variable][cell] += step
            down[variable][cell] -= step
            slope = ((expression(up).val[cell] - expression(down).val[cell])
                     / (2.0 * step))
            np.testing.assert_allclose(jac[cell, variable * n + cell], slope,
                                       rtol=1e-6, atol=1e-8)


def test_a_gathered_value_keeps_its_own_group():
    """Two different gathers must not be merged into one another.

    Groups merge on the identity of their row map, so the map is what
    separates ``lam[upstream]`` from ``lam[downstream]``.  Merging those
    would place a derivative on the wrong cell's column -- a wrong answer
    that looks entirely well formed.
    """
    n, nvar = 8, 8
    rng = np.random.default_rng(5)
    x = DiagonalADI.variable(rng.uniform(1.0, 2.0, n), nvar, 0)
    up = np.array([0, 1, 2, 3])
    down = np.array([4, 5, 6, 7])

    combined = x[up] * 2.0 + x[down] * 3.0
    assert len(combined.groups) == 2

    jac = combined.to_sparse().jac.toarray()
    expected = np.zeros((4, nvar))
    expected[np.arange(4), up] = 2.0
    expected[np.arange(4), down] = 3.0
    np.testing.assert_allclose(jac, expected)
