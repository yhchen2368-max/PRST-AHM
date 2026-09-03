"""Tests for FAHM's parameter-to-deck path and its gradient wrapper."""

import numpy as np
import pytest

from PRSTCore.hm.APP.fahm import (FahmConfig, apply_multipliers,
                                  with_finite_difference_gradient)

DECK = '\n'.join(['RUNSPEC', 'METRIC', 'GRID', 'PERMX', '100 /',
                  'EDIT', 'PROPS', 'SCHEDULE', 'END'])


def test_multiplier_block_lands_at_the_end_of_the_grid_section():
    out = apply_multipliers(DECK, {'permx': 2.0}).splitlines()
    assert out.index('MULTIPLY') > out.index('GRID')
    assert out.index('MULTIPLY') < out.index('EDIT')


def test_multiplier_block_names_the_right_keyword_and_factor():
    out = apply_multipliers(DECK, {'permx': 2.5})
    assert 'PERMX' in out and '2.5' in out
    assert out.rstrip().count('/') >= 2      # the row and the block terminator


def test_pore_volume_is_multiplied_through_multpv():
    """Scaling PORO would also move anything else that reads porosity."""
    assert 'MULTPV' in apply_multipliers(DECK, {'porevolume': 1.2})


def test_a_unit_multiplier_leaves_the_deck_byte_identical():
    assert apply_multipliers(DECK, {'permx': 1.0}) == DECK


def test_several_parameters_share_one_block():
    out = apply_multipliers(DECK, {'permx': 2.0, 'permz': 0.5})
    assert out.count('MULTIPLY') == 1
    assert 'PERMX' in out and 'PERMZ' in out


def test_the_original_deck_text_survives_untouched():
    """The overlay must not disturb any existing line -- that is the whole
    reason it exists rather than a round-trip rewrite."""
    out = apply_multipliers(DECK, {'permx': 2.0})
    for line in DECK.splitlines():
        assert line in out.splitlines()


def test_an_unknown_parameter_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match='No deck keyword'):
        apply_multipliers(DECK, {'nosuch': 2.0})


# ------------------------------------------------------- the endpoints --

# What a saturation table would give: connate water 0.2, water critical
# 0.25, maximum water relperm 0.8.
ENDPOINTS = {'swl': 0.2, 'swcr': 0.25, 'swu': 1.0, 'krw': 0.8,
             'sowcr': 0.4, 'kro': 1.0, 'sgcr': 0.05, 'krg': 0.9}
NC = 100


def _tuned(multipliers, deck=DECK):
    return apply_multipliers(deck, multipliers, endpoints=ENDPOINTS,
                             ncells=NC)


def test_an_endpoint_is_written_as_an_array_not_a_multiplier():
    """SWCR is an optional PROPS array; a deck that leaves its endpoints
    to the saturation table has nothing for MULTIPLY to scale."""
    out = _tuned({'swcr': 1.2}).splitlines()
    assert 'SWCR' in out
    assert 'MULTIPLY' not in out
    assert '  100*0.3  /' in out          # 0.25 * 1.2, over every cell


def test_endpoints_land_in_props_not_grid():
    out = _tuned({'swcr': 1.2}).splitlines()
    assert out.index('PROPS') < out.index('SWCR') < out.index('SCHEDULE')


def test_endscale_is_added_when_the_deck_does_not_ask_for_it():
    """ECLIPSE ignores the endpoint arrays without it."""
    out = _tuned({'swcr': 1.2}).splitlines()
    assert 'ENDSCALE' in out
    assert out.index('RUNSPEC') < out.index('ENDSCALE') < out.index('GRID')


def test_an_existing_endscale_is_not_duplicated():
    deck = DECK.replace('METRIC', 'METRIC\nENDSCALE\n/')
    assert _tuned({'swcr': 1.2}, deck).count('ENDSCALE') == 1


def test_rock_and_endpoint_parameters_go_to_their_own_sections():
    out = _tuned({'permx': 2.0, 'swcr': 1.2}).splitlines()
    assert out.index('MULTIPLY') < out.index('EDIT')
    assert out.index('PROPS') < out.index('SWCR')


def test_a_saturation_endpoint_cannot_leave_the_unit_interval():
    """ECLIPSE rejects a saturation outside [0, 1]."""
    out = _tuned({'swu': 1.5})
    assert '100*1  /' in out


def test_a_relperm_maximum_may_exceed_one():
    """Unlike a saturation, ECLIPSE accepts kr > 1."""
    assert '100*1.6  /' in _tuned({'krw': 2.0})


def test_a_unit_endpoint_factor_leaves_the_deck_alone():
    assert _tuned({'swcr': 1.0}) == DECK


def test_tuning_an_endpoint_without_its_table_value_is_refused():
    """Silently skipping it would tune nothing and report success."""
    with pytest.raises(ValueError, match='saturation-table value'):
        apply_multipliers(DECK, {'swcr': 1.2}, ncells=NC)


def test_tuning_an_endpoint_without_a_cell_count_is_refused():
    with pytest.raises(ValueError, match='cell count'):
        apply_multipliers(DECK, {'swcr': 1.2}, endpoints=ENDPOINTS)


def test_every_gui_parameter_has_a_deck_route():
    """The Parameter tab offers fifteen quantities; each must reach the
    deck, or selecting one would abort the run."""
    from PRSTCore.hm.APP.fahm import _ENDPOINT_KEYWORD, _MULTIPLY_TARGET
    from PRSTCore.hm.APP.fahm_app import PARAMETERS
    from PRSTCore.hm.APP.fahm_parameters import BACKEND_NAME
    routed = set(_MULTIPLY_TARGET) | set(_ENDPOINT_KEYWORD)
    for name in PARAMETERS:
        assert BACKEND_NAME.get(name, name.lower()) in routed, name


def test_every_endpoint_knows_which_table_point_it_scales():
    from PRSTCore.hm.APP.fahm import _ENDPOINT_KEYWORD, _ENDPOINT_SOURCE
    assert set(_ENDPOINT_SOURCE) == set(_ENDPOINT_KEYWORD)


def test_a_deck_without_a_grid_section_is_rejected():
    with pytest.raises(ValueError, match='no GRID section'):
        apply_multipliers('RUNSPEC\nMETRIC\nSCHEDULE\n', {'permx': 2.0})


# ----------------------------------------------------- gradient wrapper --

def test_finite_differences_recover_a_known_gradient():
    f = with_finite_difference_gradient(lambda u: float(np.sum(u ** 2)),
                                        h=1e-4)
    v, g = f(np.array([0.3, 0.4]))
    assert v == pytest.approx(0.25)
    assert np.allclose(g, [0.6, 0.8], atol=1e-3)


def test_the_step_turns_inward_at_the_upper_bound():
    """Stepping past 1.0 would leave the unit box, so the difference is
    taken backwards there instead."""
    seen = []

    def objective(u):
        seen.append(float(u[0]))
        return float(u[0])

    f = with_finite_difference_gradient(objective, h=0.05)
    _, g = f(np.array([0.98]))
    assert max(seen) <= 1.0
    assert g[0] == pytest.approx(1.0, abs=1e-9)     # sign still correct


def test_one_extra_evaluation_per_parameter():
    calls = []
    f = with_finite_difference_gradient(
        lambda u: calls.append(1) or float(np.sum(u)), h=0.01)
    f(np.zeros(3))
    assert len(calls) == 4          # one base plus one per parameter


def test_extra_arguments_reach_the_objective():
    """run_history_match passes the case directory through."""
    seen = []
    f = with_finite_difference_gradient(
        lambda u, d: seen.append(d) or float(np.sum(u)), h=0.01)
    f(np.zeros(1), 'case3')
    assert set(seen) == {'case3'}


# ---------------------------------------------------------- unscaling --

def test_the_unit_box_maps_onto_the_parameter_limits():
    from PRSTCore.hm.APP.fahm import make_objective
    config = FahmConfig(deck_path='x.DATA', work_dir='.', parameters=['permx'])
    lo, hi = config.limits_for('permx')
    unscale = make_objective(config).unscale
    assert unscale([0.0])[0] == pytest.approx(lo)
    assert unscale([1.0])[0] == pytest.approx(hi)


def test_the_default_start_point_is_a_unit_multiplier():
    """run_history_match starts from the untouched deck, so u0 must
    unscale to exactly 1.0."""
    from PRSTCore.hm.APP.fahm import make_objective
    config = FahmConfig(deck_path='x.DATA', work_dir='.', parameters=['permx'])
    lo, hi = config.limits_for('permx')
    u0 = (1.0 - lo) / (hi - lo)
    assert make_objective(config).unscale([u0])[0] == pytest.approx(1.0)
