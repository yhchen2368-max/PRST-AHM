"""Tests for group control (MRST-0's GenericFacilityModel additions).

The allocation cases here are worked out by hand in the docstrings, so
what is checked is the arithmetic a reservoir engineer would do, not
whatever the code happens to produce.
"""

import numpy as np
import pytest

from PRSTCore.ad_core.models.facility_model import FacilityModel
from PRSTCore.ad_core.models.well_group_control import (CONTROL_TYPES,
                                                        get_well_limits,
                                                        get_well_potential,
                                                        update_well_group_control)


def _sols(n):
    return [{'status': True, 'type': 'bhp', 'val': 0.0} for _ in range(n)]


def _wells(n, group='G1', lims=None):
    lims = lims or [{} for _ in range(n)]
    return [{'group': group, 'lims': lims[i]} for i in range(n)]


# --------------------------------------------------------- getWellLimits --

def test_an_undeclared_limit_is_infinite():
    """It must never bind."""
    lims = get_well_limits([{'lims': {}}])
    assert np.all(np.isinf(lims))


def test_limits_land_in_the_declared_column_order():
    lims = get_well_limits([{'lims': dict(zip(CONTROL_TYPES,
                                              [1.0, 2.0, 3.0, 4.0, 5.0,
                                               6.0]))}])
    assert list(lims[0]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_a_well_without_a_lims_field_gets_no_limits():
    assert np.all(np.isinf(get_well_limits([{}])))


def test_one_row_per_well():
    assert get_well_limits([{'lims': {}}] * 4).shape == (4, 6)


# ------------------------------------------------------ getWellPotential --

def test_the_potential_is_the_rate_against_the_bhp_limit():
    """One perforation, unit mobility and b-factor: the potential is
    WI * (p - bhp_lim) for each phase."""
    W = {'WI': np.array([2.0]), 'cstatus': np.array([1]),
         'lims': {'bhp': 100.0}}
    q = get_well_potential(W, {'cdp': np.array([0.0])},
                           bw=np.ones((1, 3)), mob=np.ones((1, 3)),
                           pw=np.array([150.0]))
    # dp = 150 - 100 = 50, Tdp = -2*50 = -100 per phase.
    assert q[0] == pytest.approx(-100.0)


def test_liquid_and_total_columns_are_the_sums_they_should_be():
    W = {'WI': np.array([1.0]), 'cstatus': np.array([1]),
         'lims': {'bhp': 0.0}}
    mob = np.array([[1.0, 2.0, 3.0]])
    q = get_well_potential(W, {'cdp': np.array([0.0])},
                           bw=np.ones((1, 3)), mob=mob,
                           pw=np.array([1.0]))
    assert q[3] == pytest.approx(q[0] + q[1])            # lrat
    assert q[4] == pytest.approx(q[0] + q[1] + q[2])     # rate


def test_a_shut_connection_contributes_nothing():
    W = {'WI': np.array([1.0, 1.0]), 'cstatus': np.array([1, 0]),
         'lims': {'bhp': 0.0}}
    q = get_well_potential(W, {'cdp': np.zeros(2)}, bw=np.ones((2, 3)),
                           mob=np.ones((2, 3)), pw=np.array([1.0, 1.0]))
    assert q[0] == pytest.approx(-1.0)      # one connection, not two


def test_dissolved_gas_is_folded_into_the_gas_rate():
    """Gas that arrives dissolved in oil still counts as gas at surface."""
    W = {'WI': np.array([1.0]), 'cstatus': np.array([1]),
         'lims': {'bhp': 0.0}}
    plain = get_well_potential(W, {'cdp': np.zeros(1)}, bw=np.ones((1, 3)),
                               mob=np.ones((1, 3)), pw=np.array([1.0]))
    with_rs = get_well_potential(W, {'cdp': np.zeros(1)}, bw=np.ones((1, 3)),
                                 mob=np.ones((1, 3)), pw=np.array([1.0]),
                                 rs=np.array([0.5]))
    assert with_rs[2] == pytest.approx(plain[2] * 1.5)


# ------------------------------------------------ updateWellGroupControl --

def _potentials(**columns):
    q = np.zeros((3, 6))
    for name, values in columns.items():
        q[:, CONTROL_TYPES.index(name)] = values
    return q


def test_an_unconstrained_group_shares_pro_rata_by_potential():
    """Target 600 over potentials 100/200/300 gives 100/200/300."""
    q = _potentials(orat=[100.0, 200.0, 300.0])
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'orat',
                                      'val': 600.0}], _wells(3), q)
    assert [w['val'] for w in out] == pytest.approx([100.0, 200.0, 300.0])


def test_a_binding_limit_holds_its_well_and_the_rest_take_up_the_slack():
    """Target 300 over potentials 100/200/300 would give 50/100/150, but
    well 2 is capped at 80. The remaining 220 goes to wells 1 and 3 in
    the ratio 100:300, so 55 and 165."""
    q = _potentials(orat=[100.0, 200.0, 300.0])
    wells = _wells(3, lims=[{}, {'orat': 80.0}, {}])
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'orat',
                                      'val': 300.0}], wells, q)
    assert [w['val'] for w in out] == pytest.approx([55.0, 80.0, 165.0])


def test_the_group_target_is_met_exactly():
    q = _potentials(orat=[100.0, 200.0, 300.0])
    wells = _wells(3, lims=[{}, {'orat': 80.0}, {}])
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'orat',
                                      'val': 300.0}], wells, q)
    assert sum(w['val'] for w in out) == pytest.approx(300.0)


def test_a_well_switches_to_whichever_limit_binds():
    """The group is on total rate, but well 2 hits its *water* limit, so
    it is put on water control at that limit."""
    q = _potentials(wrat=[10.0, 100.0, 30.0], rate=[100.0, 200.0, 300.0])
    wells = _wells(3, lims=[{}, {'wrat': 50.0}, {}])
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'rate',
                                      'val': 600.0}], wells, q)
    assert [w['type'] for w in out] == ['rate', 'wrat', 'rate']
    assert out[1]['val'] == pytest.approx(50.0)


def test_the_slack_from_a_held_well_is_redistributed():
    """Well 2 held at rate 100 leaves 500 for wells 1 and 3, split
    100:300 as 125 and 375."""
    q = _potentials(wrat=[10.0, 100.0, 30.0], rate=[100.0, 200.0, 300.0])
    wells = _wells(3, lims=[{}, {'wrat': 50.0}, {}])
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'rate',
                                      'val': 600.0}], wells, q)
    assert [out[0]['val'], out[2]['val']] == pytest.approx([125.0, 375.0])


def test_the_field_group_covers_every_active_well():
    q = _potentials(orat=[100.0, 200.0, 300.0])
    wells = _wells(3, group='ANYTHING')
    out = update_well_group_control(_sols(3),
                                    [{'name': 'FIELD', 'type': 'orat',
                                      'val': 600.0}], wells, q)
    assert sum(w['val'] for w in out) == pytest.approx(600.0)


def test_a_named_group_covers_only_its_own_wells():
    """The case MRST-0's index mismatch gets wrong: the group's wells are
    not wells 1..N."""
    q = _potentials(orat=[100.0, 200.0, 300.0])
    wells = [{'group': 'OTHER', 'lims': {}}, {'group': 'G1', 'lims': {}},
             {'group': 'G1', 'lims': {}}]
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'orat',
                                      'val': 500.0}], wells, q)
    assert out[0]['val'] == 0.0                 # untouched
    assert out[1]['val'] + out[2]['val'] == pytest.approx(500.0)


def test_a_named_groups_limits_apply_to_the_right_wells():
    """With the group starting at well 2, an index mismatch would read
    well 1's limits for well 2."""
    q = _potentials(orat=[100.0, 200.0, 300.0])
    wells = [{'group': 'OTHER', 'lims': {'orat': 1.0}},
             {'group': 'G1', 'lims': {}},
             {'group': 'G1', 'lims': {'orat': 60.0}}]
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'orat',
                                      'val': 500.0}], wells, q)
    assert out[2]['val'] == pytest.approx(60.0)
    assert out[1]['val'] == pytest.approx(440.0)


def test_a_shut_well_takes_no_share():
    q = _potentials(orat=[100.0, 200.0, 300.0])
    sols = _sols(3)
    sols[1]['status'] = False
    out = update_well_group_control(sols, [{'name': 'FIELD',
                                            'type': 'orat', 'val': 400.0}],
                                    _wells(3), q)
    assert out[1]['val'] == 0.0
    assert out[0]['val'] + out[2]['val'] == pytest.approx(400.0)


def test_no_groups_leaves_everything_alone():
    q = _potentials(orat=[100.0, 200.0, 300.0])
    out = update_well_group_control(_sols(3), [], _wells(3), q)
    assert all(w['val'] == 0.0 for w in out)


def test_an_unknown_group_control_type_is_skipped():
    q = _potentials(orat=[100.0, 200.0, 300.0])
    out = update_well_group_control(_sols(3),
                                    [{'name': 'G1', 'type': 'thp',
                                      'val': 1.0}], _wells(3), q)
    assert all(w['val'] == 0.0 for w in out)


def test_zero_total_potential_does_not_divide_by_zero():
    q = np.zeros((3, 6))
    out = update_well_group_control(_sols(3),
                                    [{'name': 'FIELD', 'type': 'orat',
                                      'val': 100.0}], _wells(3), q)
    assert all(np.isfinite(w['val']) for w in out)


def test_every_well_limited_stops_rather_than_looping():
    """When no well can take the remainder there is nothing left to
    redistribute; the loop must end, not spin."""
    q = _potentials(orat=[100.0, 200.0, 300.0])
    wells = _wells(3, lims=[{'orat': 1.0}, {'orat': 1.0}, {'orat': 1.0}])
    out = update_well_group_control(_sols(3),
                                    [{'name': 'FIELD', 'type': 'orat',
                                      'val': 600.0}], wells, q)
    assert all(w['val'] == pytest.approx(1.0) for w in out)


# ---------------------------------------------------- FacilityModel hooks --

def test_the_facility_model_exposes_both_mrst0_methods():
    assert hasattr(FacilityModel, 'getWellLimits')
    assert hasattr(FacilityModel, 'updateWellGroupControl')


def test_the_facility_hook_reads_groups_from_the_driving_forces():
    q = _potentials(orat=[100.0, 200.0, 300.0])
    forces = {'G': [{'name': 'FIELD', 'type': 'orat', 'val': 600.0}],
              'W': _wells(3)}
    out = FacilityModel.updateWellGroupControl(_sols(3), forces, q)
    assert sum(w['val'] for w in out) == pytest.approx(600.0)


def test_driving_forces_without_groups_are_a_no_op():
    q = _potentials(orat=[100.0, 200.0, 300.0])
    out = FacilityModel.updateWellGroupControl(_sols(3), {'W': _wells(3)}, q)
    assert all(w['val'] == 0.0 for w in out)


# ------------------------------------------- Peaceman WI and producer compi --

def test_the_peaceman_index_reproduces_mrsts_number():
    """SPE1's wells, against the 2.831775e-12 MRST reports. The formula
    accounts for the block's shape and its anisotropy; approximating the
    equivalent radius from the cube root of the cell volume -- which an
    unused legacy path in this codebase still does -- does not."""
    from PRSTCore.core.utils.compute_well_index import compute_well_index

    ft = 0.3048
    G = {'griddim': 3}
    rock = {'perm': np.full((300, 3), 500 * 9.869233e-16)}
    dims = (np.array([1000 * ft]), np.array([1000 * ft]),
            np.array([20 * ft]))
    wi = compute_well_index(G, rock, 0.5 * ft / 2, [0], Dir='z',
                            cellDims=dims)
    assert wi[0] == pytest.approx(2.831775e-12, rel=1e-6)


def test_the_equivalent_radius_follows_the_block_not_its_volume():
    """A long thin block and a cube of the same volume have different
    equivalent radii; taking the cube root of the volume conflates them."""
    from PRSTCore.core.utils.compute_well_index import compute_well_index

    G = {'griddim': 3}
    rock = {'perm': np.full((2, 3), 1e-13)}
    flat = compute_well_index(G, rock, 0.1, [0], Dir='z',
                              cellDims=(np.array([100.0]), np.array([100.0]),
                                        np.array([1.0])))
    cube = compute_well_index(G, rock, 0.1, [0], Dir='z',
                              cellDims=(np.array([21.5]), np.array([21.5]),
                                        np.array([21.5])))
    # atol=0: these are 1e-12 quantities and np.isclose's default
    # absolute tolerance of 1e-8 would call any two of them equal.
    assert not np.isclose(flat[0], cube[0], rtol=0.05, atol=0.0)
    assert flat[0] / cube[0] < 0.1


def test_anisotropy_changes_the_index():
    from PRSTCore.core.utils.compute_well_index import compute_well_index

    G = {'griddim': 3}
    dims = (np.array([100.0]), np.array([100.0]), np.array([10.0]))
    iso = compute_well_index(G, {'perm': np.full((1, 3), 1e-13)}, 0.1, [0],
                             cellDims=dims)
    aniso = compute_well_index(
        G, {'perm': np.array([[1e-13, 1e-14, 1e-13]])}, 0.1, [0],
        cellDims=dims)
    assert not np.isclose(iso[0], aniso[0], rtol=1e-3, atol=0.0)
    assert iso[0] / aniso[0] > 2.0


def test_net_to_gross_scales_a_vertical_perforation():
    from PRSTCore.core.utils.compute_well_index import compute_well_index

    G = {'griddim': 3}
    dims = (np.array([100.0]), np.array([100.0]), np.array([10.0]))
    perm = np.full((1, 3), 1e-13)
    full = compute_well_index(G, {'perm': perm}, 0.1, [0], cellDims=dims)
    half = compute_well_index(G, {'perm': perm, 'ntg': np.array([0.5])},
                              0.1, [0], cellDims=dims)
    assert half[0] == pytest.approx(0.5 * full[0], rel=1e-12)


def test_a_bore_radius_past_the_equivalent_radius_warns():
    """Peaceman assumes the bore is much smaller than the block; past the
    equivalent radius the index changes sign and the well produces
    backwards."""
    from PRSTCore.core.utils.compute_well_index import compute_well_index

    G = {'griddim': 3}
    dims = (np.array([1.0]), np.array([1.0]), np.array([1.0]))
    with pytest.warns(RuntimeWarning, match='Peaceman'):
        compute_well_index(G, {'perm': np.full((1, 3), 1e-13)}, 10.0, [0],
                           cellDims=dims)


@pytest.mark.parametrize('control, compi', [
    ('ORAT', [0.0, 1.0, 0.0]),
    ('WRAT', [1.0, 0.0, 0.0]),
    ('GRAT', [0.0, 0.0, 1.0]),
    ('LRAT', [1.0, 1.0, 0.0]),
    ('RESV', [1.0, 1.0, 1.0]),
    ('BHP', [1.0, 1.0, 1.0]),
])
def test_a_producers_composition_comes_from_its_control_mode(control, compi):
    """MRST's process_wconprod sets it per control type. PRSTCore gave
    every producer a uniform third, which is not what MRST computes for
    any of them."""
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        _wconprod_compi
    assert _wconprod_compi(control) == compi


def test_the_producer_composition_is_left_unnormalised():
    """LRAT is [1, 1, 0], not two halves -- it marks which phases the
    control targets rather than their proportions, and MRST leaves it
    that way."""
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        _wconprod_compi
    assert sum(_wconprod_compi('LRAT')) == 2.0
    assert sum(_wconprod_compi('RESV')) == 3.0
