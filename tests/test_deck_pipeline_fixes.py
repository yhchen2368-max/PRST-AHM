"""Regression tests for two general (non-EOR-specific) deck-pipeline bugs
found and fixed while wiring the ``ad_eor`` module to run POLYMER.DATA/
SURFACTANT1D.DATA end to end:

1. ``read_props.py`` silently dropped every NTSFUN/NTPVT-region block of a
   PROPS table keyword after the first (e.g. a deck with two stacked SWOF
   tables, one per SATNUM region, only kept the first).
2. ``GenericBlackOilModel.updateForChangedControls`` never invalidated the
   cached ``state['facility_wells']``, so any WCONPROD/WCONINJE/well-keyword
   change at a later report-step control boundary was silently ignored for
   the rest of the simulation.
"""

import numpy as np

from PRSTCore.deckformat.deckinput.read_props import read_props


def test_multiblock_table_keyword_keeps_every_region():
    block = """
SWOF
    0.2000    0.0000    1.0000    0.0
    0.8000    1.0000    0.0000    0.0
/
    0.0500    0.0000    1.0000    0.0
    0.9500    1.0000    0.0000    0.0
/

DENSITY
    800    1000    1   /
"""
    props = read_props(block)
    swof = np.asarray(props['SWOF'], dtype=float).ravel()
    assert swof.size == 16  # two 2-row, 4-column blocks
    table = swof.reshape(-1, 4)
    assert np.allclose(table[:2, 0], [0.2, 0.8])
    assert np.allclose(table[2:, 0], [0.05, 0.95])
    # The keyword after the multi-block one must still parse correctly.
    assert np.allclose(np.asarray(props['DENSITY'], dtype=float).ravel(), [800.0, 1000.0, 1.0])


def test_single_block_keyword_unaffected():
    block = """
PVTW
     300        1.012       4.28e-5        0.61       0. /

ROCK
     300        3.0e-5       /
"""
    props = read_props(block)
    assert np.allclose(np.asarray(props['PVTW'], dtype=float).ravel(),
                        [300.0, 1.012, 4.28e-5, 0.61, 0.0])
    assert np.allclose(np.asarray(props['ROCK'], dtype=float).ravel(), [300.0, 3.0e-5])


def test_updateForChangedControls_invalidates_cached_wells():
    from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel

    model = GenericBlackOilModel(gas=False, mrst_generic_assembly=True)
    state = {'pressure': np.array([100.0]), 'facility_wells': [{'name': 'W1', 'val': 1.0, 'status': True}],
             'facility_well_signature': 'stale-signature'}
    _, state = model.updateForChangedControls(state, {'W': []})
    assert 'facility_wells' not in state
    assert 'facility_well_signature' not in state
