"""Where the saturation endpoints live, and who reads them.

MRST keeps them in exactly one place. ``initRelpermScaling`` fills
``rock.krscale`` from the deck's PROPS, and ``SaturationProperty`` reads
only from there -- ``pts = model.rock.krscale.(type)`` in ``getScalers``,
``model.rock.krscale.drainage.w(cix,1)`` in ``getConnateWater``. PROPS
never reaches the residual by any other route.

PRSTCore had a second route: ``_get_relperm_scaling`` built the scaling
straight from PROPS and ``rock.krscale`` went unread. Both stores then
existed and disagreed, and the endpoint parameters -- whose ``location``
points at krscale -- wrote to the one nobody looked at. A tuned endpoint
changed nothing and reported success.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from PRSTCore.deckformat.params.rock.initRelpermScaling import \
    initRelpermScaling

DECK = Path('examples/SpE1/SPE1CASE2.DATA')


# ------------------------------------------------ initRelpermScaling --

def test_the_table_has_five_columns_not_four():
    """``pts = repmat([NaN NaN NaN NaN NaN], nc, 1)``.

    The fifth is the residual-endpoint relative permeability, which
    three-point scaling needs; four columns have nowhere to put it.
    """
    ks = initRelpermScaling({'PROPS': {}}, 4)
    for phase in ('w', 'ow', 'og', 'g'):
        assert ks['drainage'][phase].shape == (4, 5)


def test_the_water_and_gas_curves_take_their_residual_endpoint_from_krwr_krgr():
    ks = initRelpermScaling({'PROPS': {'KRWR': 0.4, 'KRGR': 0.3}}, 2)
    assert np.allclose(ks['drainage']['w'][:, 4], 0.4)
    assert np.allclose(ks['drainage']['g'][:, 4], 0.3)


def test_the_oil_curves_take_theirs_from_krorw_and_krorg():
    """``crtv = ['KROR', phase(end)]`` -- the two oil curves have separate
    residual endpoints even though they share KRO for the maximum."""
    ks = initRelpermScaling({'PROPS': {'KRO': 0.8, 'KRORW': 0.55,
                                       'KRORG': 0.45}}, 2)
    assert np.allclose(ks['drainage']['ow'][:, 3], 0.8)
    assert np.allclose(ks['drainage']['og'][:, 3], 0.8)
    assert np.allclose(ks['drainage']['ow'][:, 4], 0.55)
    assert np.allclose(ks['drainage']['og'][:, 4], 0.45)


def test_there_are_three_tables_including_the_miscible_one():
    """MRST-0's `% edited by zhang` third table: the saturation functions
    at full surfactant concentration, keyed by the ``S`` prefix.
    ``imposeRelpermScaling`` accepts SSWL/SSWCR/SKRW and needs somewhere
    to put them."""
    ks = initRelpermScaling({'PROPS': {'SSWL': 0.05, 'SSWCR': 0.15,
                                       'ISWL': 0.11}}, 2)
    assert sorted(ks) == ['drainage', 'imbibition', 'miscible']
    assert np.allclose(ks['miscible']['w'][:, 0], 0.05)
    assert np.allclose(ks['miscible']['w'][:, 1], 0.15)
    assert np.allclose(ks['imbibition']['w'][:, 0], 0.11)


def test_an_absent_keyword_stays_nan_rather_than_taking_a_default():
    """The NaN is load-bearing: ``getConnateWater`` patches per entry from
    ``fluid.krPts``, so a zero or a guessed default here would be used as
    if the deck had stated it."""
    ks = initRelpermScaling({'PROPS': {'SWL': 0.1}}, 3)
    w = ks['drainage']['w']
    assert np.allclose(w[:, 0], 0.1)
    assert np.all(np.isnan(w[:, 1:]))


# ----------------------------------------- the model reads that store --

@pytest.fixture(scope='module')
def endscale_model():
    """SPE1 with ENDSCALE switched on -- without it the model has no
    endpoint scaling and everything below is vacuous."""
    if not DECK.exists():
        pytest.skip('SPE1 deck not present')
    from PRSTCore.ad_core.simulators import adjoint_verification as V

    tmp = Path(tempfile.mkdtemp())
    for f in DECK.parent.iterdir():
        if f.is_file():
            shutil.copy2(f, tmp / f.name)
    deck = tmp / 'SPE1_ENDSCALE.DATA'
    deck.write_text(DECK.read_text().replace(
        'RUNSPEC', 'RUNSPEC\n\nENDSCALE\n/\n', 1))

    model, state0, forces, dt = V.build_case(str(deck))
    assert model._get_relperm_tables() is not None
    return model


def _swcr(model):
    tables = model._get_relperm_tables()
    nc = model.G['cells']['num']
    return np.asarray(
        model._get_relperm_scaling(nc, tables)['target']['w'][:, 1]).copy()


def test_writing_krscale_changes_what_the_residual_reads(endscale_model):
    """This is the whole point: before, the residual went on reading the
    deck's value and the tuned endpoint was inert."""
    model = endscale_model
    nc = model.G['cells']['num']
    before = _swcr(model)

    table = np.full((nc, 5), np.nan)
    table[:, 1] = 0.31
    model.rock['krscale'] = {'drainage': {'w': table}}
    after = _swcr(model)

    assert not np.allclose(before, after)
    assert np.allclose(after, 0.31)
    del model.rock['krscale']


def test_a_nan_entry_falls_back_instead_of_propagating(endscale_model):
    """``initRelpermScaling`` leaves an absent keyword NaN, so most of
    krscale is NaN on a deck that states no endpoints. Copying those
    through would replace every defaulted endpoint with a NaN that
    spreads into every saturation it scales."""
    model = endscale_model
    nc = model.G['cells']['num']
    before = _swcr(model)

    model.rock['krscale'] = initRelpermScaling(
        {'PROPS': model.inputdata.get('PROPS', {})}, nc)
    after = _swcr(model)

    assert np.all(np.isfinite(after))
    assert np.allclose(before, after)
    del model.rock['krscale']


def test_the_scaling_cache_notices_a_changed_endpoint(endscale_model):
    """The cache was keyed on the cell count alone, so it would serve the
    pre-tuning values for the rest of the run."""
    model = endscale_model
    nc = model.G['cells']['num']
    _swcr(model)                              # prime the cache

    table = np.full((nc, 5), np.nan)
    table[:, 1] = 0.27
    model.rock['krscale'] = {'drainage': {'w': table}}
    assert np.allclose(_swcr(model), 0.27)

    table2 = np.full((nc, 5), np.nan)
    table2[:, 1] = 0.33
    model.rock['krscale'] = {'drainage': {'w': table2}}
    assert np.allclose(_swcr(model), 0.33)
    del model.rock['krscale']


def test_an_endpoint_parameter_reaches_the_residual(endscale_model):
    """``ModelParameter``'s location for swcr is
    ``rock.krscale.drainage.w[:, 1]``; setting it has to move the value
    the residual scales saturations with."""
    from PRSTCore.optimization.utils.parameters import add_parameter

    model = endscale_model
    nc = model.G['cells']['num']
    table = np.full((nc, 5), np.nan)
    table[:, 1] = float(_swcr(model)[0])
    model.rock['krscale'] = {'drainage': {'w': table}}

    setup = {'model': model,
             'schedule': {'step': {'val': [1.0], 'control': [0]},
                          'control': [{'W': []}]},
             'state0': {}}
    param = add_parameter([], setup, name='swcr',
                          relative_limits=[0.5, 2.0])[0]
    param.set_parameter(setup, np.full(nc, 0.29))

    assert np.allclose(_swcr(model), 0.29)
    del model.rock['krscale']
