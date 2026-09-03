"""``fluid.krPts`` against MRST's own numbers.

``PRSTCore.ad_props.kr_points`` ports the ``getPoints`` subfunctions of
MRST's ``assignSWOF``/``assignSGOF``/``assignSWFN``/``assignSGFN``/
``assignSOF2``/``assignSOF3``. The reference values below were produced
by running ``initDeckADIFluid`` on the same two decks in MATLAB
(R2026a, mrst-2026a) and printing ``fluid.krPts`` -- see the block
comment on :data:`MRST`.

Without these points the saturation endpoints have no values, and
history matching's endpoint parameters fall silently back to their
multipliers rather than failing.
"""

import os

import numpy as np
import pytest

from PRSTCore.ad_props.kr_points import get_kr_points

#: ``fluid.krPts`` as MRST reports it, per deck. Regenerate with::
#:
#:     deck  = convertDeckUnits(readEclipseDeck(path));
#:     fluid = initDeckADIFluid(deck);
#:     fluid.krPts.w, fluid.krPts.ow, fluid.krPts.og, fluid.krPts.g
MRST = {
    'examples/SpE1/SPE1CASE2.DATA': {
        'w':  [[0.12, 0.12, 1.0, 1e-05]],
        'ow': [[0.0, 0.16, 1.0, 1.0]],
        'og': [[0.0, 0.18, 1.0, 1.0]],
        'g':  [[0.0, 0.02, 0.88, 0.984]],
    },
    'examples/HM/QIEDIE.DATA': {
        'w':  [[0.311, 0.311, 1.0, 1.0]],
        'ow': [[0.0, 0.238, 1.0, 1.0]],
        'og': [[0.0, 0.371, 1.0, 1.0]],
        'g':  [[0.0, 0.035, 0.354, 0.89]],
    },
}


def _props(path):
    from PRSTCore.deckformat.deckinput.convert_deck_units import \
        convert_deck_units
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import \
        read_eclipse_deck
    return (convert_deck_units(read_eclipse_deck(path)) or {}).get('PROPS', {})


@pytest.mark.parametrize('deck', sorted(MRST))
@pytest.mark.parametrize('curve', ['w', 'ow', 'og', 'g'])
def test_the_points_match_mrst(deck, curve):
    if not os.path.exists(deck):
        pytest.skip('%s not present' % deck)
    ours = get_kr_points(_props(deck))[curve]
    theirs = np.asarray(MRST[deck][curve], dtype=float)
    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs, rtol=1e-10, atol=1e-14)


# --------------------------------------------------- the reading itself --

# A two-region SWOF: the first region is water-wet with connate water
# 0.2, the second has 0.3 and a lower maximum water relperm. Regions are
# stacked as ECLIPSE writes them and split on the drop in column 0.
SWOF = np.array([
    [0.20, 0.00, 1.00, 0.0],
    [0.30, 0.00, 0.60, 0.0],     # still immobile: Swcr is here, not 0.2
    [0.60, 0.20, 0.00, 0.0],     # oil stops moving: Sowcr = 1 - 0.6
    [0.80, 0.40, 0.00, 0.0],
    [0.30, 0.00, 0.90, 0.0],
    [0.70, 0.15, 0.00, 0.0],
    [0.90, 0.25, 0.00, 0.0],
])

SGOF = np.array([
    [0.00, 0.00, 1.00, 0.0],
    [0.05, 0.00, 0.80, 0.0],     # Sgcr = 0.05
    [0.50, 0.30, 0.00, 0.0],     # Sogcr = 1 - 0.5 - Swco
    [0.70, 0.60, 0.00, 0.0],
])


def test_water_takes_the_last_immobile_row():
    """Krw rises with Sw, so the critical saturation is the last row
    still at zero -- taking the first would report connate water."""
    points = get_kr_points({'SWOF': SWOF})
    assert points['w'][0].tolist() == [0.20, 0.30, 0.80, 0.40]


def test_oil_takes_the_first_immobile_row():
    """Krow falls as Sw rises, so oil stops moving at the first zero.
    Reading it the same way as water would put Sowcr at 1 - 0.8 = 0.2
    instead of 0.4."""
    points = get_kr_points({'SWOF': SWOF})
    assert points['ow'][0].tolist() == [0.0, pytest.approx(0.40), 1.0, 1.00]


def test_every_region_gets_its_own_row():
    points = get_kr_points({'SWOF': SWOF})
    assert points['w'].shape == (2, 4)
    assert points['w'][1].tolist() == [0.30, 0.30, 0.90, 0.25]
    assert points['ow'][1][1] == pytest.approx(1 - 0.70)


def test_oil_against_gas_subtracts_connate_water():
    """Krog is tabulated against Sg in a system that also holds connate
    water, so So at that row is 1 - Sg - Swco. Forgetting the Swco term
    overstates Sogcr by the connate water saturation."""
    points = get_kr_points({'SWOF': SWOF, 'SGOF': SGOF})
    assert points['og'][0][1] == pytest.approx(1 - 0.50 - 0.20)


def test_gas_reads_its_own_columns_like_water():
    points = get_kr_points({'SWOF': SWOF, 'SGOF': SGOF})
    assert points['g'][0].tolist() == [0.0, 0.05, 0.70, 0.60]


def test_a_deck_without_saturation_tables_reports_nothing():
    """An analytical fluid has no tabulated points; that is not an
    error, and must not stop a model being built."""
    assert get_kr_points({'PVTW': [[1, 1, 1, 1]]}) == {}
    assert get_kr_points(None) == {}


# ------------------------------------------------------------ Family II --

SWFN = np.array([[0.15, 0.00, 0.0], [0.25, 0.00, 0.0], [0.85, 0.50, 0.0]])
SGFN = np.array([[0.00, 0.00, 0.0], [0.04, 0.00, 0.0], [0.75, 0.70, 0.0]])
SOF3 = np.array([[0.10, 0.00, 0.00], [0.20, 0.05, 0.02], [0.85, 0.90, 0.88]])


def test_family_two_reads_the_separate_keywords():
    points = get_kr_points({'SWFN': SWFN, 'SGFN': SGFN, 'SOF3': SOF3})
    assert points['w'][0].tolist() == [0.15, 0.25, 0.85, 0.50]
    assert points['g'][0].tolist() == [0.0, 0.04, 0.75, 0.70]
    # SOF3 states oil against its own saturation, so it rises with it and
    # the maximum relperm is the last row, not the first.
    assert points['ow'][0].tolist() == [0.0, 0.10, 1.0, 0.90]
    assert points['og'][0].tolist() == [0.0, 0.10, 1.0, 0.88]


def test_sof2_gives_both_oil_curves_the_same_points():
    """assignSOF2 records a single ``krPts.o``, which imposeRelpermScaling
    then copies into ow and og."""
    sof2 = np.array([[0.10, 0.00], [0.30, 0.20], [0.90, 0.95]])
    points = get_kr_points({'SWFN': SWFN, 'SOF2': sof2})
    assert points['ow'].tolist() == points['og'].tolist()
    assert points['ow'][0].tolist() == [0.0, 0.10, 1.0, 0.95]


def test_swof_wins_over_swfn():
    """A deck stating both families is contradictory; MRST assigns from
    Family I, so this reports Family I's points."""
    points = get_kr_points({'SWOF': SWOF, 'SWFN': SWFN})
    assert points['w'][0][0] == 0.20        # SWOF's connate water
