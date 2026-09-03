"""Parity tests for the general-grid port of MRST ``setupOperatorsTPFA``.

The half-transmissibility kernel itself is covered by
``test_incomp_tpfa_mrst_parity.py``; what is checked here is the operator
layer built on top of it -- the harmonic face reduction, the C/Grad/Div
sign conventions, faceAvg, and faceUpstr's flag semantics -- plus
agreement with the pre-existing logical-Cartesian implementation.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.operators import setup_operators
from PRSTCore.ad_core.operators_tpfa import (get_face_transmissibility,
                                             pore_volume, setup_operators_tpfa)
from PRSTCore.deckformat.grid.init_eclipse_grid import init_eclipse_grid
from PRSTCore.gridprocessing.compute_geometry import compute_geometry
from PRSTCore.gridprocessing.tensor_grid import tensor_grid

NX, NY, NZ = 5, 4, 3
DX, DY, DZ = 10.0, 20.0, 5.0


@pytest.fixture
def grid():
    return compute_geometry(tensor_grid(np.arange(NX + 1) * DX,
                                        np.arange(NY + 1) * DY,
                                        np.arange(NZ + 1) * DZ))


@pytest.fixture
def rock(grid):
    rng = np.random.default_rng(0)
    nc = grid['cells']['num']
    return {'perm': rng.uniform(1e-14, 1e-12, (nc, 3)),
            'poro': rng.uniform(0.1, 0.3, nc)}


def test_matches_logical_cartesian_implementation(grid, rock):
    """The general path must reproduce the Cartesian one it generalises."""
    ops = setup_operators_tpfa(grid, rock)

    deck = {'GRID': {'DXV': np.full(NX, DX), 'DYV': np.full(NY, DY),
                     'DZV': np.full(NZ, DZ), 'cartDims': [NX, NY, NZ]},
            'RUNSPEC': {'cartDims': [NX, NY, NZ]}}
    reference = setup_operators(init_eclipse_grid(deck), rock)

    def order(N):
        s = np.sort(N, axis=1)
        return np.lexsort((s[:, 1], s[:, 0]))

    # setup_operators returns 1-based N; setup_operators_tpfa is 0-based.
    ref_N = reference['N'] - 1
    i, j = order(ops['N']), order(ref_N)
    assert np.array_equal(np.sort(ops['N'], axis=1)[i], np.sort(ref_N, axis=1)[j])
    assert np.abs(ops['T'][i] - reference['T'][j]).max() < 1e-12 * np.abs(reference['T']).max()


def test_pore_volume_is_poro_times_volume_times_ntg(grid, rock):
    assert np.allclose(pore_volume(grid, rock),
                       grid['cells']['volumes'] * rock['poro'])
    with_ntg = dict(rock, ntg=np.full(grid['cells']['num'], 0.5))
    assert np.allclose(pore_volume(grid, with_ntg),
                       grid['cells']['volumes'] * rock['poro'] * 0.5)


def test_only_internal_connections_are_kept(grid, rock):
    ops = setup_operators_tpfa(grid, rock)
    assert np.all(ops['N'] >= 0)
    assert ops['N'].shape[0] == int(np.count_nonzero(ops['internalConn']))
    assert ops['T'].size == ops['N'].shape[0]


def test_harmonic_reduction(grid, rock):
    """T_face = 1 / sum(1/hT) over the face's half-faces."""
    from PRSTCore.solvers.incomp.compute_trans import compute_trans
    hT = compute_trans(grid, rock)
    cf = np.asarray(grid['cells']['faces'])[:, 0]
    expected = 1.0 / np.bincount(cf, weights=1.0 / hT,
                                 minlength=grid['faces']['num'])
    assert np.allclose(get_face_transmissibility(grid, rock), expected)


def test_grad_and_div_sign_conventions(grid, rock):
    """setupOperatorsTPFA: C[f,N1]=+1, C[f,N2]=-1; Grad = -C, Div = C'."""
    ops = setup_operators_tpfa(grid, rock)
    N, C = ops['N'], ops['C'].toarray()
    f = 0
    assert C[f, N[f, 0]] == 1.0 and C[f, N[f, 1]] == -1.0

    x = np.arange(float(grid['cells']['num']))
    assert np.allclose(ops['Grad'](x), x[N[:, 1]] - x[N[:, 0]])

    flux = np.arange(1.0, N.shape[0] + 1.0)
    assert np.allclose(ops['Div'](flux), ops['C'].T @ flux)
    acc = np.ones(grid['cells']['num'])
    assert np.allclose(ops['AccDiv'](acc, flux), acc + ops['C'].T @ flux)


def test_face_avg_is_the_arithmetic_mean_of_the_two_cells(grid, rock):
    ops = setup_operators_tpfa(grid, rock)
    x = np.arange(float(grid['cells']['num']))
    N = ops['N']
    assert np.allclose(ops['faceAvg'](x), 0.5 * (x[N[:, 0]] + x[N[:, 1]]))


def test_face_upstr_flag_selects_first_neighbour(grid, rock):
    """faceUpstr.m: upCell = N(:,2); upCell(flag) = N(flag,1)."""
    ops = setup_operators_tpfa(grid, rock)
    N = ops['N']
    x = np.arange(float(grid['cells']['num']))
    nf = N.shape[0]
    flag = np.zeros(nf, dtype=bool)
    flag[::2] = True
    expected = np.where(flag, x[N[:, 0]], x[N[:, 1]])
    assert np.allclose(ops['faceUpstr'](flag, x), expected)
    # A scalar flag is broadcast to every interface.
    assert np.allclose(ops['faceUpstr'](True, x), x[N[:, 0]])
    assert np.allclose(ops['faceUpstr'](False, x), x[N[:, 1]])


def test_operators_propagate_adi_jacobians(grid, rock):
    """Grad/Div/faceAvg/faceUpstr must carry a SparseADI Jacobian through."""
    ops = setup_operators_tpfa(grid, rock)
    nc = grid['cells']['num']
    u = SparseADI.variable(np.arange(float(nc)), nc, 0)
    N, nf = ops['N'], ops['N'].shape[0]

    assert np.allclose(ops['Grad'](u).jac.toarray(), (-ops['C']).toarray())
    assert np.allclose(ops['faceAvg'](u).jac.toarray(), ops['M'].toarray())

    flag = np.ones(nf, dtype=bool)
    S = sp.csr_matrix((np.ones(nf), (np.arange(nf), N[:, 0])), shape=(nf, nc))
    assert np.allclose(ops['faceUpstr'](flag, u).jac.toarray(), S.toarray())


def test_user_supplied_neighbours_and_trans(grid, rock):
    """The user_provided_trans branch, as the nwm hybrid grid uses it."""
    base = setup_operators_tpfa(grid, rock)
    ops = setup_operators_tpfa(grid, rock, neighbors=base['N'],
                               trans=base['T'], porv=base['pv'])
    assert np.array_equal(ops['N'], base['N'])
    assert np.allclose(ops['T'], base['T'])
    assert np.allclose(ops['pv'], base['pv'])


def test_neighbours_without_trans_select_a_subset_of_the_grid(grid, rock):
    """MRST dispatches on ``trans``, not on ``neighbors``.

    Given cell pairs but no transmissibility, ``grid_based_trans`` computes
    T from the grid and keeps only the faces whose pair was listed --
    MRST-0's ``getNeighborSubsetIndex``, one of its `% edited by zhang`
    changes (2026a asserts the neighbours are empty instead).  This is the
    call HistoryMatching makes with the connections ECLIPSE's INIT file
    reports, and without it the operator setup cannot be built at all.
    """
    base = setup_operators_tpfa(grid, rock)
    keep = np.arange(0, base['N'].shape[0], 2)

    ops = setup_operators_tpfa(grid, rock, neighbors=base['N'][keep, :])

    assert ops['N'].shape[0] == keep.size
    assert np.array_equal(np.sort(ops['N'], axis=1),
                          np.sort(base['N'][keep, :], axis=1))
    # T comes from the grid, not from the caller.
    assert np.allclose(np.sort(ops['T']), np.sort(base['T'][keep]))
    assert ops['T_all'].size == grid['faces']['num']


def test_all_neighbours_without_trans_reproduces_the_plain_setup(grid, rock):
    """Passing every internal pair must change nothing."""
    base = setup_operators_tpfa(grid, rock)
    ops = setup_operators_tpfa(grid, rock, neighbors=base['N'])
    assert np.array_equal(ops['N'], base['N'])
    assert np.allclose(ops['T'], base['T'])
    assert np.allclose(ops['T_all'], base['T_all'])


def test_mismatched_pore_volume_is_rejected(grid, rock):
    with pytest.raises(ValueError):
        setup_operators_tpfa(grid, rock, porv=np.ones(3))


def test_neighbour_index_convention_is_not_guessed_from_the_minimum(grid, rock):
    """A 0-based N whose cell 0 takes part in no connection must not be
    mistaken for a 1-based one and shifted by one."""
    from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel

    nc = grid['cells']['num']
    model = GenericBlackOilModel(grid, rock, {}, water=True, oil=True,
                                 gas=True, mrst_generic_assembly=True)
    T = np.array([1.0, 2.0, 3.0])

    zero_based = np.array([[1, 2], [2, 3], [3, 4]])
    model.operators = {'N': zero_based, 'T': T}
    c1, c2, _ = model._internal_connections()
    assert np.array_equal(c1, zero_based[:, 0])
    assert np.array_equal(c2, zero_based[:, 1])

    # A genuine 1-based list is recognised by reaching nc.
    one_based = np.array([[1, 2], [2, 3], [nc - 1, nc]])
    model.operators = {'N': one_based, 'T': T}
    c1, c2, _ = model._internal_connections()
    assert np.array_equal(c1, one_based[:, 0] - 1)

    # An explicit marker overrides the inference either way.
    model.operators = {'N': zero_based, 'T': T, 'oneBased': True}
    c1, _, _ = model._internal_connections()
    assert np.array_equal(c1, zero_based[:, 0] - 1)


def test_cartesian_setup_operators_declares_its_convention():
    deck = {'GRID': {'DXV': np.full(NX, DX), 'DYV': np.full(NY, DY),
                     'DZV': np.full(NZ, DZ), 'cartDims': [NX, NY, NZ]},
            'RUNSPEC': {'cartDims': [NX, NY, NZ]}}
    nc = NX * NY * NZ
    rock = {'perm': np.full((nc, 3), 1e-13), 'poro': np.full(nc, 0.2)}
    assert setup_operators(init_eclipse_grid(deck), rock)['oneBased'] is True
