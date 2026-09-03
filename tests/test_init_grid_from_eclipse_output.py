"""Tests for ``init_grid_from_eclipse_output``.

The synthetic cases pin the parts that are easy to get wrong -- region
selection, multipliers, eMap, and the transmissibility units. The tests
against a real ECLIPSE INIT/EGRID run only when one is present.
"""

import os

import numpy as np
import pytest

from PRSTCore.deckformat.resultinput.init_grid_from_eclipse_output import (
    _assign_multipliers, _build_emap, _get_active_neighbors, _get_regions,
    _get_unit_system, init_grid_from_eclipse_output)


def _kw(values):
    return {'values': np.asarray(values, dtype=float)}


# ------------------------------------------------------------- regions --

def test_a_single_region_array_is_not_recorded():
    """One region carries no information, so MRST leaves the field out
    and downstream code can test for its presence."""
    init = {'SATNUM': _kw(np.ones(6)), 'PVTNUM': _kw(np.ones(6))}
    assert 'regions' not in _get_regions({}, init, slice(None))


def test_several_regions_are_recorded():
    init = {'SATNUM': _kw([1, 1, 2, 2, 3, 3])}
    regions = _get_regions({}, init, slice(None))['regions']
    assert list(regions['saturation']) == [1, 1, 2, 2, 3, 3]


def test_every_region_keyword_maps_to_its_own_field():
    init = {'PVTNUM': _kw([1, 2]), 'SATNUM': _kw([1, 2]),
            'IMBNUM': _kw([1, 2]), 'FIPNUM': _kw([1, 2]),
            'EQLNUM': _kw([1, 2])}
    regions = _get_regions({}, init, slice(None))['regions']
    assert set(regions) == {'pvt', 'saturation', 'imbibition', 'fluid',
                            'equilibration'}


def test_rocknum_is_recorded_even_with_one_region():
    """MRST tests only for presence here, not for a second region."""
    regions = _get_regions({}, {'ROCKNUM': _kw(np.ones(4))},
                           slice(None))['regions']
    assert 'rock' in regions


def test_surfactant_pulls_in_satnum_even_when_uniform():
    """Surfactant relative permeability is given as saturation tables, so
    SATNUM is needed whether or not it has more than one region."""
    init = {'SURFNUM': _kw([1, 2]), 'SATNUM': _kw(np.ones(2))}
    regions = _get_regions({}, init, slice(None))['regions']
    assert 'surfactant' in regions and 'saturation' in regions


def test_regions_are_indexed_through_the_emap():
    init = {'SATNUM': _kw([1, 2, 3, 4])}
    regions = _get_regions({}, init, np.array([3, 1]))['regions']
    assert list(regions['saturation']) == [4, 2]


# -------------------------------------------------------- multipliers --

def _G(nc=4, cart=(2, 2, 1)):
    return {'cells': {'num': nc, 'indexMap': np.arange(nc)},
            'cartDims': np.asarray(cart)}


def test_a_multiplier_is_recorded_under_its_direction():
    rock = _assign_multipliers({}, {'MULTX': _kw([1.0, 2.0, 1.0, 1.0])}, _G())
    assert list(rock['multipliers']) == ['x']


def test_the_negative_direction_form_keeps_its_underscore():
    rock = _assign_multipliers({}, {'MULTZ_': _kw([1.0, 0.5, 1.0, 1.0])}, _G())
    assert 'z_' in rock['multipliers']


def test_a_multiplier_of_one_everywhere_is_dropped():
    """It changes nothing and only costs a per-cell array downstream."""
    rock = _assign_multipliers({}, {'MULTX': _kw(np.ones(4))}, _G())
    assert 'multipliers' not in rock


def test_a_full_grid_multiplier_is_mapped_onto_the_active_cells():
    G = {'cells': {'num': 2, 'indexMap': np.array([1, 3])},
         'cartDims': np.asarray([2, 2, 1])}
    rock = _assign_multipliers({}, {'MULTX': _kw([1.0, 2.0, 1.0, 3.0])}, G)
    assert list(rock['multipliers']['x']) == [2.0, 3.0]


def test_a_non_finite_multiplier_warns():
    with pytest.warns(RuntimeWarning, match='non-finite'):
        _assign_multipliers({}, {'MULTX': _kw([1.0, np.nan, 2.0, 1.0])}, _G())


# --------------------------------------------------------------- eMap --

def test_emap_is_a_slice_when_the_grids_agree():
    G = {'cells': {'num': 3, 'indexMap': np.array([0, 1, 2])}}
    e_map, e_map_inv, consistent = _build_emap(G, np.ones(3, bool), 3)
    assert consistent and isinstance(e_map, slice) \
        and isinstance(e_map_inv, slice)


def test_emap_maps_grid_cells_to_init_rows_when_they_differ():
    """Processing the geometry can drop a cell ACTNUM marked active; the
    INIT arrays still have a row for it."""
    act = np.array([True, True, True, True])
    G = {'cells': {'num': 2, 'indexMap': np.array([0, 3])}}
    e_map, e_map_inv, consistent = _build_emap(G, act, 4)
    assert not consistent
    assert list(e_map) == [0, 3]
    assert e_map_inv[0] == 0 and e_map_inv[3] == 1


# ------------------------------------------------------------- units --

def test_transmissibility_is_not_permeability_times_length():
    """ECLIPSE's METRIC transmissibility unit is cP*m^3/(day*bar), which
    is 1.16e-13 -- perm*length would give 1e-25 and every connection
    would be twelve orders of magnitude too tight."""
    u = _get_unit_system('metric')
    assert u['trans'] == pytest.approx(1e-3 / (86400.0 * 1e5), rel=1e-9)


def test_field_units_use_stb_and_psi():
    u = _get_unit_system('field')
    assert u['trans'] == pytest.approx(
        1e-3 * 0.158987294928 / (86400.0 * 6894.757293168361), rel=1e-6)


# ------------------------------------------------------- connections --

def _init_2x1x1():
    """Two cells side by side in I, one connection between them."""
    return {'TRANX': _kw([5.0, 0.0]), 'TRANY': _kw([0.0, 0.0]),
            'TRANZ': _kw([0.0, 0.0])}


def test_a_single_connection_is_found():
    N, T, _ = _get_active_neighbors(_init_2x1x1(), {},
                                    np.ones(2, bool), (2, 1, 1))
    assert N.shape == (1, 2)
    assert sorted(N[0]) == [0, 1]
    assert T[0] == pytest.approx(5.0)


def test_a_zero_transmissibility_connection_is_dropped():
    init = _init_2x1x1()
    init['TRANX'] = _kw([0.0, 0.0])
    N, T, _ = _get_active_neighbors(init, {}, np.ones(2, bool), (2, 1, 1))
    assert N.shape[0] == 0


def test_a_connection_to_an_inactive_cell_is_dropped():
    """Three cells in a row with the middle one inactive. INIT arrays are
    per *active* cell, so TRANX has two entries, and neither active cell
    has a live +I neighbour."""
    act = np.array([True, False, True])
    init = {'TRANX': _kw([5.0, 5.0]), 'TRANY': _kw([0.0, 0.0]),
            'TRANZ': _kw([0.0, 0.0])}
    N, _, _ = _get_active_neighbors(init, {}, act, (3, 1, 1))
    assert N.shape[0] == 0


def test_neighbours_come_back_zero_based():
    N, _, _ = _get_active_neighbors(_init_2x1x1(), {},
                                    np.ones(2, bool), (2, 1, 1))
    assert N.min() >= 0


def test_non_neighbour_connections_are_included():
    init = _init_2x1x1()
    init.update({'NNC1': _kw([1]), 'NNC2': _kw([2]), 'TRANNNC': _kw([7.0])})
    N, T, nnc = _get_active_neighbors(init, {}, np.ones(2, bool), (2, 1, 1))
    assert 7.0 in list(T)
    assert nnc['cells'].shape == (1, 2)


def test_missing_nnc_transmissibilities_warn_and_become_infinite():
    """MRST substitutes inf, which then survives the T > 0 filter and
    lands in the model as an infinite transmissibility."""
    init = _init_2x1x1()
    init.update({'NNC1': _kw([1]), 'NNC2': _kw([2])})
    with pytest.warns(RuntimeWarning, match='NNCs not given'):
        _, T, _ = _get_active_neighbors(init, {}, np.ones(2, bool),
                                        (2, 1, 1))
    assert np.isinf(T).any()


# ------------------------------------------------- against real output --

_PREFIX = os.environ.get('ECLIPSE_INIT_PREFIX')


def _files():
    if not _PREFIX or not os.path.exists(_PREFIX + '.INIT'):
        pytest.skip('set ECLIPSE_INIT_PREFIX to an ECLIPSE run with INIT')
    from PRSTCore.deckformat.resultinput.read_eclipse_output_file_unfmt import \
        read_eclipse_output_file_unfmt as rd
    return rd(_PREFIX + '.INIT'), rd(_PREFIX + '.EGRID')


def test_real_output_gives_a_grid_of_the_declared_size():
    init, grid = _files()
    G, _, _, _ = init_grid_from_eclipse_output(init, grid)
    assert G['cells']['num'] == int(np.prod(G['cartDims']))


def test_real_output_gives_physical_permeability():
    init, grid = _files()
    _, rock, _, _ = init_grid_from_eclipse_output(init, grid)
    # 1 mD is 9.87e-16 m^2; anything outside 0.01 mD - 100 D is not rock.
    assert rock['perm'].min() > 1e-18 and rock['perm'].max() < 1e-10


def test_real_output_gives_physical_transmissibility():
    """The check that caught the unit error: a 100 mD, 50x15 m face at
    50 m spacing has a geometric transmissibility near 1.5e-12 m^3."""
    init, grid = _files()
    _, _, _, T = init_grid_from_eclipse_output(init, grid)
    assert T.min() > 1e-16 and T.max() < 1e-9


def test_real_output_neighbours_are_within_the_grid():
    init, grid = _files()
    G, _, N, _ = init_grid_from_eclipse_output(init, grid)
    assert N.min() >= 0 and N.max() < G['cells']['num']


def test_real_output_carries_net_to_gross():
    init, grid = _files()
    _, rock, _, _ = init_grid_from_eclipse_output(init, grid)
    assert 'ntg' in rock and rock['ntg'].size == rock['perm'].shape[0]
