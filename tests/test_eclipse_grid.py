"""``initEclipseGrid``: a deck's GRID section as a complete MRST grid.

The grid a deck produces is the foundation every geometry routine
indexes through. PRSTCore's older builder was, by its own docstring, "a
pragmatic, lightweight counterpart": it returned a different shape per
branch and carried no cell-to-face topology at all. A Cartesian deck
came back with ``xfaces``/``yfaces``/``zfaces`` and no ``faces``; a
corner-point deck with ``faces`` but no ``cells.faces``.

Neither carries ``griddim``, ``cells.faces`` or ``cells.facePos``, which
is what ``computeTrans``, ``computeWellIndex`` and the permeability
parameter's ``perm2directionalTrans`` index through. All three were
ported faithfully from MRST and none of them could run on a deck-derived
grid -- not because the ports were wrong, but because nothing produced a
grid they could accept.

These pin the replacement, which dispatches exactly as
``initEclipseGrid.m`` does.
"""

import os

import numpy as np
import pytest

from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from PRSTCore.deckformat.grid.eclipse_grid import init_eclipse_grid

#: One deck per branch of the dispatch. SPE1 states DX/DY/DZ with a
#: constant TOPS (block-centred); the other two state COORD/ZCORN.
CARTESIAN = 'examples/SpE1/SPE1CASE2.DATA'
CORNER_POINT = 'examples/HM/QIEDIE.DATA'


def _grid(path):
    if not os.path.exists(path):
        pytest.skip('%s not present' % path)
    return init_eclipse_grid(convert_deck_units(read_eclipse_deck(path)))


@pytest.fixture(scope='module')
def cartesian():
    return _grid(CARTESIAN)


@pytest.fixture(scope='module')
def corner_point():
    return _grid(CORNER_POINT)


# ------------------------------------------------------- completeness --

@pytest.mark.parametrize('which', ['cartesian', 'corner_point'])
def test_both_branches_carry_the_cell_to_face_topology(which, request):
    """The point of the replacement. Every MRST geometry routine walks
    ``cells.faces`` through ``cells.facePos``; without them a grid is
    only a list of cells that happen to know their volumes."""
    G = request.getfixturevalue(which)
    assert G['griddim'] == 3
    assert 'faces' in G['cells'] and 'facePos' in G['cells']
    assert 'neighbors' in G['faces'] and 'num' in G['faces']

    face_pos = np.asarray(G['cells']['facePos'], dtype=int)
    assert face_pos.size == G['cells']['num'] + 1
    assert face_pos[0] == 0
    assert face_pos[-1] == np.asarray(G['cells']['faces']).shape[0]


@pytest.mark.parametrize('which', ['cartesian', 'corner_point'])
def test_compute_trans_can_run_on_it(which, request):
    """``computeTrans`` was ported from MRST and had no deck-derived
    grid to run on. One half-transmissibility per half face."""
    from PRSTCore.solvers.incomp.compute_trans import compute_trans

    G = request.getfixturevalue(which)
    nc = int(G['cells']['num'])
    T = compute_trans(G, {'perm': np.full((nc, 3), 1e-13)})
    assert T.size == np.asarray(G['cells']['faces']).shape[0]
    assert np.all(np.isfinite(T)) and np.all(T >= 0)


def test_the_two_branches_agree_on_what_a_grid_is(cartesian, corner_point):
    """A structure that varies by branch is what the replacement is for:
    downstream code had to know which kind of deck it came from."""
    assert set(cartesian['cells']) >= {'faces', 'facePos', 'num', 'centroids',
                                       'volumes'}
    assert set(corner_point['cells']) >= set(cartesian['cells']) - {'indexMap'}
    assert set(cartesian['faces']) >= {'neighbors', 'num', 'nodes', 'nodePos',
                                       'areas', 'centroids', 'normals'}


# ------------------------------------------------------------ geometry --

def test_the_cartesian_grid_has_spe1s_dimensions(cartesian):
    """SPE1 is 10x10x3 at 1000x1000x20 ft."""
    assert cartesian['cells']['num'] == 300
    assert list(cartesian['cartDims']) == [10, 10, 3]
    # Six faces per cell on a Cartesian grid, none shared twice.
    assert np.asarray(cartesian['cells']['faces']).shape[0] == 300 * 6


def test_the_corner_point_grid_has_qiedies_dimensions(corner_point):
    assert corner_point['cells']['num'] == 52 * 52 * 20
    assert list(corner_point['cartDims']) == [52, 52, 20]


def test_internal_faces_have_two_neighbours_and_boundary_ones_have_one(
        cartesian):
    neighbors = np.asarray(cartesian['faces']['neighbors'], dtype=int)
    internal = (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0)
    # 10x10x3: 9*10*3 + 10*9*3 + 10*10*2 internal faces.
    assert int(internal.sum()) == 9 * 10 * 3 + 10 * 9 * 3 + 10 * 10 * 2
    assert np.all(neighbors[~internal].max(axis=1) >= 0)


def test_volumes_are_positive_and_sum_to_the_bulk(cartesian):
    """SPE1's deck says it in a comment: 10x1000 ft by 10x1000 ft, with
    the three layers 20, 30 and 50 ft thick. The bulk volume is a
    closed-form number, so this checks the geometry rather than
    restating whatever the code produced."""
    volumes = np.asarray(cartesian['cells']['volumes'], dtype=float)
    assert np.all(volumes > 0)
    ft = 0.3048
    assert volumes.sum() == pytest.approx(
        (10 * 1000 * ft) * (10 * 1000 * ft) * ((20 + 30 + 50) * ft),
        rel=1e-9)


def test_the_layers_have_their_stated_thicknesses(cartesian):
    """20, 30, 50 ft: a uniform 100/3 would give the same total."""
    ft = 0.3048
    volumes = np.asarray(cartesian['cells']['volumes'], dtype=float)
    area = (1000 * ft) ** 2
    per_layer = volumes.reshape(3, 100)[:, 0] / area
    assert np.allclose(per_layer, np.array([20.0, 30.0, 50.0]) * ft)


# -------------------------------------------------------- the dispatch --

def test_a_block_centred_deck_needs_a_constant_tops():
    """``getDeltas``' restriction, kept rather than quietly flattened: a
    varying TOPS describes a geometry a tensor product cannot express."""
    deck = {'RUNSPEC': {'cartDims': [2, 1, 1]},
            'GRID': {'DX': [10.0, 10.0], 'DY': [10.0, 10.0],
                     'DZ': [1.0, 1.0], 'TOPS': [100.0, 200.0]}}
    with pytest.raises(ValueError, match='TOPS'):
        init_eclipse_grid(deck)


def test_a_non_tensor_dx_is_rejected_not_averaged():
    """DX varying along J is not a tensor grid. MRST asserts; taking the
    first row instead would silently build a different reservoir."""
    deck = {'RUNSPEC': {'cartDims': [2, 2, 1]},
            'GRID': {'DX': [10.0, 10.0, 20.0, 20.0],
                     'DY': [10.0] * 4, 'DZ': [1.0] * 4}}
    with pytest.raises(ValueError, match='tensor-grid'):
        init_eclipse_grid(deck)


def test_a_deck_with_no_usable_grid_says_so():
    with pytest.raises(ValueError, match='Grid not implemented'):
        init_eclipse_grid({'RUNSPEC': {'cartDims': [1, 1, 1]}, 'GRID': {}})


def test_dxv_and_dx_describe_the_same_grid():
    """The vector form and the per-cell form are two spellings of one
    thing, and ``getDeltas`` exists to reduce the second to the first."""
    vector = init_eclipse_grid({
        'RUNSPEC': {'cartDims': [2, 2, 1]},
        'GRID': {'DXV': [10.0, 20.0], 'DYV': [5.0, 5.0], 'DZV': [1.0]}})
    per_cell = init_eclipse_grid({
        'RUNSPEC': {'cartDims': [2, 2, 1]},
        'GRID': {'DX': [10.0, 20.0, 10.0, 20.0], 'DY': [5.0] * 4,
                 'DZ': [1.0] * 4}})
    assert np.allclose(vector['cells']['volumes'],
                       per_cell['cells']['volumes'])
    assert np.allclose(vector['cells']['centroids'],
                       per_cell['cells']['centroids'])


def test_actnum_removes_cells_from_a_tensor_grid():
    full = init_eclipse_grid({
        'RUNSPEC': {'cartDims': [2, 2, 1]},
        'GRID': {'DXV': [10.0, 10.0], 'DYV': [10.0, 10.0], 'DZV': [1.0]}})
    holed = init_eclipse_grid({
        'RUNSPEC': {'cartDims': [2, 2, 1]},
        'GRID': {'DXV': [10.0, 10.0], 'DYV': [10.0, 10.0], 'DZV': [1.0],
                 'ACTNUM': [1, 0, 1, 1]}})
    assert full['cells']['num'] == 4
    assert holed['cells']['num'] == 3
