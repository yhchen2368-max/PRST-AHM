"""Multi-region saturation-table handling and endpoint-scaling parity.

ECLIPSE repeats SWOF/SGOF once per NTSFUN region and the deck parser
concatenates the blocks. The stacked array is not monotonic in saturation,
so it must be sliced apart before interpolation -- ``interpTable`` sorts by
the saturation column, which interleaves the regions into a single nonsense
curve rather than failing.
"""

import numpy as np
import pytest

from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel
from PRSTCore.ad_props.relperm_tables import (build_swof_sgof_tables,
                                              split_table_regions)

REGION_1 = [[0.2, 0.0, 1.0, 0.0], [0.5, 0.2, 0.4, 0.0], [0.8, 0.6, 0.0, 0.0]]
REGION_2 = [[0.1, 0.0, 1.0, 0.0], [0.4, 0.5, 0.3, 0.0], [0.9, 0.9, 0.0, 0.0]]


def _flat(*blocks):
    return [v for block in blocks for row in block for v in row]


def test_split_finds_each_region():
    stacked = np.array(REGION_1 + REGION_2, dtype=float)
    regions = split_table_regions(stacked)
    assert len(regions) == 2
    assert np.allclose(regions[0], REGION_1)
    assert np.allclose(regions[1], REGION_2)


def test_single_region_table_is_returned_whole():
    single = np.array(REGION_1, dtype=float)
    regions = split_table_regions(single)
    assert len(regions) == 1
    assert np.allclose(regions[0], single)


@pytest.mark.parametrize('region, expected', [(0, REGION_1), (1, REGION_2)])
def test_selected_region_is_monotonic_and_correct(region, expected):
    swof, _ = build_swof_sgof_tables({'SWOF': _flat(REGION_1, REGION_2)},
                                     region=region)
    assert np.allclose(swof, expected)
    assert np.all(np.diff(swof[:, 0]) > 0)


def test_stacked_table_would_otherwise_interleave():
    """The regression this guards: sorting the stacked table by saturation
    produces a non-monotonic krw curve reading ~0.25 at Sw=0.3 where the
    region-1 curve gives ~0.0667."""
    from PRSTCore.ad_core.adi import SparseADI, ad_interp_linear

    stacked = np.array(REGION_1 + REGION_2, dtype=float)
    x = np.r_[stacked[0, 0] - 1.0, stacked[:, 0], stacked[-1, 0] + 1.0]
    y = np.r_[stacked[0, 1], stacked[:, 1], stacked[-1, 1]]
    bad = ad_interp_linear(x, y, SparseADI.variable(np.array([0.3]), 1, 0)).val[0]
    good = np.interp(0.3, np.array(REGION_1)[:, 0], np.array(REGION_1)[:, 1])
    assert not np.isclose(bad, good, rtol=1e-3)

    swof, _ = build_swof_sgof_tables({'SWOF': _flat(REGION_1, REGION_2)}, region=0)
    x = np.r_[swof[0, 0] - 1.0, swof[:, 0], swof[-1, 0] + 1.0]
    y = np.r_[swof[0, 1], swof[:, 1], swof[-1, 1]]
    fixed = ad_interp_linear(x, y, SparseADI.variable(np.array([0.3]), 1, 0)).val[0]
    assert np.isclose(fixed, good)


def _model(deck):
    model = GenericBlackOilModel({'cells': {'num': 1}}, {}, {}, water=True,
                                 oil=True, gas=True, mrst_generic_assembly=True)
    model.inputdata = deck
    return model


def test_uniform_satnum_selects_that_region():
    deck = {'PROPS': {}, 'REGIONS': {'SATNUM': [2, 2, 2]}}
    assert _model(deck)._saturation_region() == 1


def test_absent_satnum_defaults_to_the_first_region():
    assert _model({'PROPS': {}})._saturation_region() == 0
    assert _model({'PROPS': {}, 'REGIONS': {}})._saturation_region() == 0


def test_varying_satnum_is_reported_rather_than_silently_using_region_one():
    deck = {'PROPS': {}, 'REGIONS': {'SATNUM': [1, 2, 1]}}
    with pytest.raises(NotImplementedError, match='per-cell dispatch'):
        _model(deck)._saturation_region()
