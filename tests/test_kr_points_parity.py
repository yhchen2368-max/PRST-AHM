"""The two places that compute relperm scaling points must agree.

There are two, and that is the point of this file.

``PRSTCore.ad_props.kr_points.get_kr_points`` reads a deck's PROPS
section and reports every region's ``[connate, critical, max-saturation,
max-relperm]`` -- it is what ``fluid.krPts`` is built from, and what
history matching's endpoint parameters take their base values from.

``GenericBlackOilModel._get_relperm_scaling`` computes the same four
sets inline, for the one SATNUM region it is evaluating, and feeds them
to the flow path's endpoint scaling.

Two implementations of one definition drift. They are not merged here:
the inline one carries a guard the shared one does not -- it disables
end-point scaling altogether for a table with no zero-relperm row --
and removing that would change the flow path's answers on such a deck,
which is not a change to make for tidiness. So they stay separate and
this holds them level instead.

They also differ in one detail that has never mattered on a real deck:
the inline version finds the immobile-oil row with ``<= eps`` and the
shared one with ``== 0``. A table with a krow of, say, 1e-17 would part
them. That is worth knowing about rather than worth "fixing" blind.
"""

import io
import os
import tempfile

import numpy as np
import pytest

from PRSTCore.ad_props.kr_points import get_kr_points

CURVES = ('w', 'ow', 'og', 'g')


def _endscale_spe1():
    """SPE1 does not ask for end-point scaling, so the inline path
    returns None for it. A copy that does gives something to compare."""
    source = 'examples/SpE1/SPE1CASE2.DATA'
    if not os.path.exists(source):
        return None
    text = io.open(source, encoding='utf-8', errors='replace').read()
    text = text.replace('RUNSPEC', 'RUNSPEC\n\nENDSCALE\n/\n', 1)
    path = os.path.join(tempfile.gettempdir(), 'prstcore_spe1_endscale.DATA')
    io.open(path, 'w', encoding='utf-8').write(text)
    return path


#: Every deck in the tree that turns end-point scaling on, plus one made
#: to. Norne uses three-point scaling and QIEDIE two, so both branches
#: are covered.
DECKS = ['examples/HM/QIEDIE.DATA',
         'examples/Norne/Norne_simplified/NORNE_ATW2013.DATA']


@pytest.mark.parametrize('deck', DECKS + ['spe1+endscale'])
def test_the_flow_path_and_the_shared_reader_agree(deck):
    if deck == 'spe1+endscale':
        deck = _endscale_spe1()
        if deck is None:
            pytest.skip('SPE1 deck not present')
    elif not os.path.exists(deck):
        pytest.skip('%s not present' % deck)

    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad

    _state0, model, _schedule, _ = init_eclipse_problem_ad(deck)
    nc = int(model.G['cells']['num'])
    inline = model._get_relperm_scaling(nc, model._get_relperm_tables())
    assert inline is not None, 'end-point scaling is off; nothing to compare'

    shared = get_kr_points((model.inputdata or {}).get('PROPS', {}))
    region = model._saturation_region()

    for curve in CURVES:
        theirs = np.asarray(inline['table'][curve], dtype=float)
        table = np.asarray(shared[curve], dtype=float)
        ours = table[min(region, table.shape[0] - 1)]
        assert np.allclose(ours, theirs, rtol=0, atol=0), (deck, curve,
                                                           ours, theirs)


def test_both_report_the_same_four_curves():
    deck = _endscale_spe1()
    if deck is None:
        pytest.skip('SPE1 deck not present')
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        init_eclipse_problem_ad

    _state0, model, _schedule, _ = init_eclipse_problem_ad(deck)
    nc = int(model.G['cells']['num'])
    inline = model._get_relperm_scaling(nc, model._get_relperm_tables())
    shared = get_kr_points((model.inputdata or {}).get('PROPS', {}))
    assert set(inline['table']) == set(shared) == set(CURVES)


def test_the_two_differ_only_where_a_relperm_is_denormal():
    """Pins the one known difference so it is a documented choice rather
    than a surprise. The inline path treats a krow at or below machine
    epsilon as immobile; the shared one requires an exact zero.
    """
    # Sw ascending, krow falling to a denormal rather than to zero.
    swof = np.array([
        [0.10, 0.00, 1.0e+00, 0.0],
        [0.40, 0.10, 1.0e-17, 0.0],     # <= eps, but not == 0
        [0.90, 0.50, 0.0e+00, 0.0],
    ])
    shared = get_kr_points({'SWOF': swof})
    # The shared reader waits for the exact zero at Sw = 0.90.
    assert shared['ow'][0][1] == pytest.approx(1.0 - 0.90)
    # The inline rule would have stopped at Sw = 0.40 instead.
    inline_row = int(np.flatnonzero(swof[:, 2] <= np.finfo(float).eps)[0])
    assert swof[inline_row, 0] == 0.40
