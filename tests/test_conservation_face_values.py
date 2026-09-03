"""A face value must give the divergence the existing path gives.

The point of :mod:`PRSTCore.ad_core.conservation` is to reach a cell-length
Jacobian without ever building the face-length one.  That is only worth
anything if the two agree, so every test here computes the same quantity
both ways: through ``FaceValue.divergence``, and through the sparse route it
replaces -- assemble the face value as an ``nface x nvar`` matrix and
multiply by the divergence operator ``C``.

The expressions exercised are the ones a black-oil flux actually performs: a
potential difference between the two cells, a gravity term averaging their
densities, an upstream-weighted mobility, and products of those.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import PRSTCore  # noqa: F401
from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.conservation import (LEFT, RIGHT, CellVariableLayout,
                                           FaceValue, upwind_flag)
from PRSTCore.ad_core.diagonal_adi import DiagonalADI


def _grid(nx=4, ny=3, nz=2, seed=0):
    """A neighbour table and the divergence operator that goes with it."""
    index = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    pairs = []
    for a, b in ((index[:-1].ravel(), index[1:].ravel()),
                 (index[:, :-1].ravel(), index[:, 1:].ravel()),
                 (index[:, :, :-1].ravel(), index[:, :, 1:].ravel())):
        pairs.append(np.stack([a, b], axis=1))
    neighbours = np.concatenate(pairs, axis=0).astype(np.int64)
    ncell = nx * ny * nz
    nface = neighbours.shape[0]
    C = sp.csr_matrix((np.r_[np.ones(nface), -np.ones(nface)],
                       (np.r_[neighbours[:, 0], neighbours[:, 1]],
                        np.r_[np.arange(nface), np.arange(nface)])),
                      shape=(ncell, nface))
    return ncell, neighbours, C


def _variables(ncell, ngroup, seed=0):
    """``ngroup`` seeded cell variables, in both representations."""
    rng = np.random.default_rng(seed)
    nvar = ngroup * ncell
    values = [rng.uniform(1.0, 2.0, ncell) for _ in range(ngroup)]
    sparse = [SparseADI.variable(v, nvar, k * ncell) for k, v in enumerate(values)]
    diagonal = [DiagonalADI.variable(v, nvar, k * ncell) for k, v in enumerate(values)]
    return values, sparse, diagonal, nvar


def _assert_same(face, reference_ad, ncell, C):
    """Compare ``FaceValue.divergence`` against ``C @ face_sparse``.

    Agreement is asked for relative to the size of the numbers involved.
    The two routes add a cell's faces in different orders -- ``bincount``
    against a sparse matrix product -- so the last bit can differ, and an
    absolute tolerance would only be measuring how large the deck's units
    happen to make the fluxes.
    """
    got = face.divergence(ncell)
    want_value = C @ reference_ad.val
    want_jac = (C @ reference_ad.jac).tocsr()

    scale = max(float(np.abs(want_value).max()), 1.0)
    assert np.abs(got.val - want_value).max() <= 1e-12 * scale, (
        'divergence value differs by %g' % np.abs(got.val - want_value).max())

    difference = (got.jac - want_jac).tocoo()
    if difference.nnz:
        largest = max(float(abs(want_jac).max()), 1.0)
        assert np.abs(difference.data).max() <= 1e-12 * largest, (
            'divergence Jacobian differs by %g' % np.abs(difference.data).max())


@pytest.mark.parametrize('ngroup', [1, 2, 3])
def test_a_plain_difference_matches_the_sparse_route(ngroup):
    ncell, neighbours, C = _grid()
    values, sparse, _, nvar = _variables(ncell, ngroup)
    layout = CellVariableLayout(ncell, ngroup, nvar)
    c1, c2 = neighbours[:, 0], neighbours[:, 1]

    face = (FaceValue.gather(sparse[0], layout, neighbours, RIGHT)
            - FaceValue.gather(sparse[0], layout, neighbours, LEFT))
    reference = sparse[0][c2] - sparse[0][c1]
    _assert_same(face, reference, ncell, C)


def test_gravity_term_averaging_both_cells():
    """``(rho[c1] + rho[c2]) * dz`` -- both sides contribute to one face."""
    ncell, neighbours, C = _grid()
    values, sparse, _, nvar = _variables(ncell, 3, seed=1)
    layout = CellVariableLayout(ncell, 3, nvar)
    c1, c2 = neighbours[:, 0], neighbours[:, 1]
    rng = np.random.default_rng(2)
    dz = rng.standard_normal(neighbours.shape[0])

    rho = sparse[0] * 800.0 + sparse[1] * 3.0
    face = ((FaceValue.gather(rho, layout, neighbours, LEFT)
             + FaceValue.gather(rho, layout, neighbours, RIGHT)) * (0.5 * dz))
    reference = (rho[c1] + rho[c2]) * (0.5 * dz)
    _assert_same(face, reference, ncell, C)


def test_upstream_weighting_picks_one_side_per_face():
    ncell, neighbours, C = _grid()
    values, sparse, _, nvar = _variables(ncell, 3, seed=3)
    layout = CellVariableLayout(ncell, 3, nvar)
    c1, c2 = neighbours[:, 0], neighbours[:, 1]

    potential = sparse[0][c2] - sparse[0][c1]
    flag = upwind_flag(potential)
    upstream = np.where(flag, c1, c2)

    mobility = sparse[1] * sparse[1]
    face = FaceValue.gather(mobility, layout, neighbours, flag)
    reference = mobility[upstream]
    np.testing.assert_allclose(face.val, reference.val, rtol=0, atol=1e-12)
    _assert_same(face, reference, ncell, C)


def test_a_full_flux_expression():
    """Potential times transmissibility times upstream mobility and density."""
    ncell, neighbours, C = _grid(5, 4, 3)
    values, sparse, _, nvar = _variables(ncell, 3, seed=4)
    layout = CellVariableLayout(ncell, 3, nvar)
    c1, c2 = neighbours[:, 0], neighbours[:, 1]
    rng = np.random.default_rng(5)
    T = rng.uniform(0.5, 2.0, neighbours.shape[0])
    dz = rng.standard_normal(neighbours.shape[0])
    g = 9.80665

    pressure, saturation, third = sparse
    rho = pressure * 0.1 + 800.0 * third
    mobility = saturation * saturation * third
    density = third * 1.05

    potential_ref = (pressure[c2] - pressure[c1]
                     - (rho[c1] + rho[c2]) * (0.5 * g * dz))
    flag = upwind_flag(potential_ref)
    upstream = np.where(flag, c1, c2)
    reference = potential_ref * (-T) * mobility[upstream] * density[upstream]

    potential = (FaceValue.gather(pressure, layout, neighbours, RIGHT)
                 - FaceValue.gather(pressure, layout, neighbours, LEFT)
                 - (FaceValue.gather(rho, layout, neighbours, LEFT)
                    + FaceValue.gather(rho, layout, neighbours, RIGHT)) * (0.5 * g * dz))
    face = (potential * (-T)
            * FaceValue.gather(mobility, layout, neighbours, flag)
            * FaceValue.gather(density, layout, neighbours, flag))
    _assert_same(face, reference, ncell, C)


def test_the_diagonal_and_sparse_representations_gather_alike():
    """A cell value in either representation must produce the same face value."""
    ncell, neighbours, C = _grid()
    values, sparse, diagonal, nvar = _variables(ncell, 3, seed=6)
    layout = CellVariableLayout(ncell, 3, nvar)

    from_sparse = FaceValue.gather(sparse[1] * sparse[2], layout, neighbours, LEFT)
    from_diagonal = FaceValue.gather(diagonal[1] * diagonal[2], layout, neighbours, LEFT)
    np.testing.assert_allclose(from_sparse.val, from_diagonal.val, rtol=0, atol=1e-12)
    np.testing.assert_allclose(from_sparse.deriv, from_diagonal.deriv,
                               rtol=0, atol=1e-12)


def test_derivative_width_does_not_grow_with_expression_depth():
    """The reason the class exists: the shape is fixed, whatever is built."""
    ncell, neighbours, _ = _grid()
    values, sparse, _, nvar = _variables(ncell, 3, seed=7)
    layout = CellVariableLayout(ncell, 3, nvar)

    face = FaceValue.gather(sparse[0], layout, neighbours, LEFT)
    for _ in range(40):
        face = (face * FaceValue.gather(sparse[1], layout, neighbours, RIGHT)
                + FaceValue.gather(sparse[2], layout, neighbours, LEFT))
    assert face.deriv.shape == (neighbours.shape[0], 2, 3)


def test_a_constant_face_value_has_no_derivatives():
    ncell, neighbours, C = _grid()
    layout = CellVariableLayout(ncell, 2, 2 * ncell)
    face = FaceValue.constant(np.arange(neighbours.shape[0], dtype=float),
                              layout, neighbours)
    assert not face.deriv.any()
    divergence = face.divergence(ncell)
    assert divergence.jac.nnz == 0 or not divergence.jac.data.any()


# --------------------------------------------------------------- kernel --
from PRSTCore.ad_core import mex  # noqa: E402
from PRSTCore.ad_core.conservation import (DivergenceAssembler,  # noqa: E402
                                           divergence_precomputes)

kernel_required = pytest.mark.skipif(
    mex.load_discrete_divergence() is None,
    reason='the divergence kernel is not built for this interpreter')


@kernel_required
@pytest.mark.parametrize('ngroup', [1, 2, 3])
def test_the_compiled_kernel_agrees_with_the_python_path(ngroup):
    """Two implementations of the same assembly.

    The compiled one is MRST's ``mexDiscreteDivergenceJac``; the Python one
    sums into precomputed slots.  Agreement is relative, not exact: a cell's
    contributions are added in face order by the kernel and in column order
    by ``bincount``, and floating-point addition does not commute across a
    reordering.  Anything beyond rounding means the two disagree about the
    matrix, which is what this is for.
    """
    ncell, neighbours, C = _grid(4, 3, 2)
    values, sparse, _, nvar = _variables(ncell, ngroup, seed=11)
    layout = CellVariableLayout(ncell, ngroup, nvar)
    c1, c2 = neighbours[:, 0], neighbours[:, 1]

    face = (FaceValue.gather(sparse[0], layout, neighbours, RIGHT)
            - FaceValue.gather(sparse[0], layout, neighbours, LEFT))
    if ngroup > 1:
        face = face * FaceValue.gather(sparse[1], layout, neighbours, LEFT)

    compiled = DivergenceAssembler(neighbours, ncell, layout, use_kernel=True)
    python = DivergenceAssembler(neighbours, ncell, layout, use_kernel=False)
    assert compiled._kernel is not None
    assert python._kernel is None

    got = compiled.assemble(face)
    want = python.assemble(face)
    np.testing.assert_array_equal(got.val, want.val)
    difference = (got.jac - want.jac).tocoo()
    if difference.nnz:
        largest = max(float(abs(want.jac).max()), 1.0)
        assert np.abs(difference.data).max() <= 1e-13 * largest, (
            'kernel and Python assembly differ by %g'
            % np.abs(difference.data).max())


@kernel_required
def test_the_kernel_folds_the_accumulation_into_the_diagonal():
    """MRST puts the cell-local term straight on the diagonal, in one pass."""
    ncell, neighbours, C = _grid(3, 3, 2)
    values, sparse, _, nvar = _variables(ncell, 3, seed=12)
    layout = CellVariableLayout(ncell, 3, nvar)
    rng = np.random.default_rng(13)
    accumulation = rng.standard_normal((ncell, 3))

    face = FaceValue.gather(sparse[0], layout, neighbours, LEFT)
    compiled = DivergenceAssembler(neighbours, ncell, layout, use_kernel=True)
    python = DivergenceAssembler(neighbours, ncell, layout, use_kernel=False)

    got = compiled.assemble(face, accumulation=accumulation)
    want = python.assemble(face, accumulation=accumulation)
    difference = (got.jac - want.jac).tocoo()
    largest = max(float(abs(want.jac).max()), 1.0)
    assert not difference.nnz or np.abs(difference.data).max() <= 1e-14 * largest


@kernel_required
def test_precomputes_describe_every_connection_once_per_direction():
    ncell, neighbours, _ = _grid(3, 2, 2)
    face_pos, faces, cells, cell_index = divergence_precomputes(neighbours, ncell)

    assert face_pos.size == ncell + 1
    assert int(face_pos[-1]) == 2 * neighbours.shape[0]
    assert faces.size == cells.size == 2 * neighbours.shape[0]
    # Every face is seen exactly twice, once from each of its cells.
    counts = np.bincount(faces, minlength=neighbours.shape[0])
    assert np.all(counts == 2)
    # The signed neighbour never names the cell it belongs to.
    owners = np.repeat(np.arange(ncell), np.diff(face_pos))
    assert np.all(np.abs(cells) - 1 != owners)
    # The diagonal slot is within the cell's own run of connections.
    assert np.all(cell_index <= np.diff(face_pos))


# ------------------------------------------------------- face arithmetic --
from PRSTCore.ad_core import conservation as _cons  # noqa: E402

face_kernel_required = pytest.mark.skipif(
    mex.load_face_operators() is None,
    reason='the face-operator kernel is not built for this interpreter')


@contextlib.contextmanager
def _without_face_kernel():
    """Run the block with the compiled face arithmetic switched off.

    The numpy twin is the reference: it is the readable statement of what
    each operation means, and the kernel is the thing that has to match it.
    """
    saved = _cons._KERNEL
    _cons._KERNEL = None
    try:
        yield
    finally:
        _cons._KERNEL = saved


def _both_ways(build):
    """``build()`` with the kernel on and off."""
    with_kernel = build()
    with _without_face_kernel():
        without = build()
    return with_kernel, without


@face_kernel_required
@pytest.mark.parametrize('ngroup', [1, 2, 3])
def test_every_face_operator_matches_its_numpy_twin(ngroup):
    ncell, neighbours, _ = _grid(4, 3, 2)
    values, sparse, _, nvar = _variables(ncell, ngroup, seed=21)
    layout = CellVariableLayout(ncell, ngroup, nvar)
    rng = np.random.default_rng(22)
    flag = rng.random(neighbours.shape[0]) < 0.5
    scale = rng.standard_normal(neighbours.shape[0])
    cell = sparse[0] * sparse[-1]

    builders = {
        'left': lambda: FaceValue.gather(cell, layout, neighbours, LEFT),
        'right': lambda: FaceValue.gather(cell, layout, neighbours, RIGHT),
        'upwind': lambda: FaceValue.gather(cell, layout, neighbours, flag),
        'gradient': lambda: FaceValue.gradient(cell, layout, neighbours),
        'average': lambda: FaceValue.average(cell, layout, neighbours),
        'scale': lambda: FaceValue.gather(cell, layout, neighbours, LEFT) * scale,
        'product': lambda: (FaceValue.gather(cell, layout, neighbours, LEFT)
                            * FaceValue.gather(cell, layout, neighbours, RIGHT)),
    }
    for name, build in builders.items():
        got, want = _both_ways(build)
        np.testing.assert_allclose(got.val, want.val, rtol=0, atol=0,
                                   err_msg='%s: values differ' % name)
        np.testing.assert_allclose(got.deriv, want.deriv, rtol=0, atol=0,
                                   err_msg='%s: derivatives differ' % name)


@face_kernel_required
def test_gradient_and_average_agree_with_gathering_both_sides():
    """The fused operators must equal the pair of gathers they replace."""
    ncell, neighbours, C = _grid(4, 3, 2)
    values, sparse, _, nvar = _variables(ncell, 3, seed=23)
    layout = CellVariableLayout(ncell, 3, nvar)
    cell = sparse[1] * 2.0 + sparse[2]

    gradient = FaceValue.gradient(cell, layout, neighbours)
    by_hand = (FaceValue.gather(cell, layout, neighbours, RIGHT)
               - FaceValue.gather(cell, layout, neighbours, LEFT))
    np.testing.assert_allclose(gradient.val, by_hand.val, rtol=0, atol=1e-13)
    np.testing.assert_allclose(gradient.deriv, by_hand.deriv, rtol=0, atol=1e-13)

    average = FaceValue.average(cell, layout, neighbours)
    halved = (FaceValue.gather(cell, layout, neighbours, LEFT)
              + FaceValue.gather(cell, layout, neighbours, RIGHT)) * 0.5
    np.testing.assert_allclose(average.val, halved.val, rtol=0, atol=1e-13)
    np.testing.assert_allclose(average.deriv, halved.deriv, rtol=0, atol=1e-13)


@face_kernel_required
def test_a_full_flux_is_unchanged_by_the_kernel():
    """End to end: the same divergence with the kernel on and off."""
    ncell, neighbours, C = _grid(5, 4, 3)
    values, sparse, _, nvar = _variables(ncell, 3, seed=24)
    layout = CellVariableLayout(ncell, 3, nvar)
    rng = np.random.default_rng(25)
    T = rng.uniform(0.5, 2.0, neighbours.shape[0])
    dz = rng.standard_normal(neighbours.shape[0])
    pressure, saturation, third = sparse
    rho = pressure * 0.1 + 800.0 * third
    mobility = saturation * saturation * third

    def build():
        potential = (FaceValue.gradient(pressure, layout, neighbours)
                     - FaceValue.average(rho, layout, neighbours) * (9.80665 * dz))
        flag = upwind_flag(potential)
        return (potential * (-T)
                * FaceValue.gather(mobility, layout, neighbours, flag))

    got, want = _both_ways(build)
    np.testing.assert_allclose(got.val, want.val, rtol=0, atol=0)
    np.testing.assert_allclose(got.deriv, want.deriv, rtol=0, atol=0)

    # And the assembled divergence agrees with the sparse route it replaces.
    c1, c2 = neighbours[:, 0], neighbours[:, 1]
    potential_ref = (pressure[c2] - pressure[c1]
                     - (rho[c1] + rho[c2]) * (0.5 * 9.80665 * dz))
    flag = upwind_flag(potential_ref)
    reference = potential_ref * (-T) * mobility[np.where(flag, c1, c2)]
    _assert_same(got, reference, ncell, C)
