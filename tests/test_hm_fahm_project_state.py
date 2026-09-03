"""Stage 5 oracle tests for FAHM Create Project's imported App state."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from PRSTCore.deckformat.unit_conversion_factors import unit_conversion_factors
from PRSTCore.hm.APP.fahm import (
    FahmConfig,
    _apply_fahm_swatinit_scale,
    _fill_default_aquifer_pressures,
    initialize_fahm_project,
    read_case,
)


REPO = Path(__file__).resolve().parents[1]
ORACLE = REPO / 'tests' / 'fixtures' / 'fahm_oracle' / 'v1'
STAGE5 = ORACLE / 'stage5'
CASE_DECK = REPO / 'examples' / 'SPE9' / 'SPE9_CP.DATA'
CASE_PREFIX = REPO / 'examples' / 'SPE9' / 'RESULTS' / 'SPE9_CP'


@pytest.fixture(scope='module')
def manifest():
    data = json.loads((STAGE5 / 'manifest.json').read_text(encoding='utf-8'))
    data['_records'] = {record['name']: record for record in data['arrays']}
    return data


def oracle_array(manifest, name):
    record = manifest['_records'][name]
    return np.fromfile(STAGE5 / record['path'], dtype=record['dtype']).reshape(
        tuple(record['shape']), order=record['order'])


@pytest.fixture(scope='module')
def project():
    deck = read_case(FahmConfig(deck_path=str(CASE_DECK), work_dir='.'))
    return initialize_fahm_project(deck, str(CASE_PREFIX) + '.DATA')


def test_stage5_oracle_is_bound_to_frozen_fahm_source(manifest):
    fingerprint = json.loads(
        (ORACLE / 'static' / 'source_fingerprint.json').read_text(
            encoding='utf-8'))
    assert manifest['schema_version'] == 'fahm-stage5-oracle-v1'
    assert manifest['reference'] == 'MRST/dev/APP/FAHM.mlapp'
    assert manifest['source_lines'] == 'FAHM.m:1753-1824'
    assert manifest['extracted_source_sha256'] == \
        fingerprint['fahm_m']['sha256']
    assert manifest['matlab_release'] == '2022b'
    for record in manifest['arrays']:
        assert record['order'] == 'F'
        assert (STAGE5 / record['path']).stat().st_size == record['nbytes']


def test_grid_rock_and_init_trans_are_exact_result_arrays(
        project, manifest):
    exact = {
        'grid/cart_dims': np.asarray(project.G['cartDims'], dtype=np.int64),
        'grid/cells_num': np.asarray([[project.G['cells']['num']]],
                                     dtype=np.int64),
        'grid/index_map_0based': project.G['cells']['indexMap'],
        'grid/PORV': project.G['cells']['PORV'],
        'grid/trans_neighbors_0based': project.N,
        'grid/trans_T': project.T,
        'rock/poro': project.rock['poro'],
        'rock/perm': project.rock['perm'],
        'rock/ntg': project.rock['ntg'],
    }
    for name, actual in exact.items():
        expected = oracle_array(manifest, name)
        actual = np.asarray(actual).reshape(expected.shape)
        assert actual.shape == expected.shape, name
        np.testing.assert_array_equal(actual, expected, err_msg=name)

    # Both implementations process the same corner points, but their
    # geometry kernels accumulate floating point products in a different
    # order.  The observed maxima are <1e-9 SI; freeze an explicit bound.
    for name, actual in (
            ('grid/centroids', project.G['cells']['centroids']),
            ('grid/volumes', project.G['cells']['volumes'])):
        expected = oracle_array(manifest, name)
        actual = np.asarray(actual).reshape(expected.shape)
        assert actual.shape == expected.shape, name
        np.testing.assert_allclose(
            actual, expected, rtol=2e-12, atol=2e-9, err_msg=name)


def test_first_restart_state_matches_matlab_arrays_and_phase_order(
        project, manifest):
    state = project.state0
    pairs = {
        'state0/pressure': state['pressure'],
        'state0/s': state['s'],
        'state0/rs': state['rs'],
        'state0/rv': np.atleast_1d(state['rv']),
        'state0/time': np.atleast_1d(state['time']),
    }
    for name, actual in pairs.items():
        expected = oracle_array(manifest, name)
        actual = np.asarray(actual).reshape(expected.shape)
        assert actual.shape == expected.shape, name
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    np.testing.assert_array_equal(state['sW'], state['s'][:, 0])
    np.testing.assert_array_equal(state['sG'], state['s'][:, 2])
    np.testing.assert_array_equal(
        np.asarray([project.model.water, project.model.oil,
                    project.model.gas], dtype=np.uint8).reshape(1, 3),
        oracle_array(manifest, 'model/phases_WOG'))
    np.testing.assert_array_equal(
        np.asarray([project.model.disgas, project.model.vapoil],
                   dtype=np.uint8).reshape(1, 2),
        oracle_array(manifest, 'model/disgas_vapoil'))


def test_phase_pressures_and_black_oil_pvt_match_matlab(
        project, manifest):
    state = project.state0
    pW, pO, pG = project.model._phase_pressures(
        state['pressure'], state['sW'], state['sG'],
        state.get('pcowScale'))
    pvt = project.model._phase_pvt_from_phase_pressures(
        pW, pO, pG, rs_override=state.get('rs'),
        rv_override=state.get('rv'), sG_override=state['sG'])
    arrays = {
        'fluid/phase_pressure_W': pW,
        'fluid/phase_pressure_O': pO,
        'fluid/phase_pressure_G': pG,
        'fluid/shrinkage_W': pvt['bw'],
        'fluid/shrinkage_O': pvt['bo'],
        'fluid/shrinkage_G': pvt['bg'],
        'fluid/viscosity_W': pvt['muw'],
        'fluid/viscosity_O': pvt['muo'],
        'fluid/viscosity_G': pvt['mug'],
        'fluid/rs_max': project.model._phase_pvt(
            state['pressure'])['rs'],
    }
    for name, actual in arrays.items():
        expected = oracle_array(manifest, name)
        actual = np.asarray(actual).reshape(expected.shape)
        assert actual.shape == expected.shape, name
        np.testing.assert_allclose(
            actual, expected, rtol=2e-12, atol=1e-15, err_msg=name)


def test_tpfa_operators_have_exact_shape_order_and_init_porv(
        project, manifest):
    operators = project.model.operators
    np.testing.assert_array_equal(
        operators['N'], oracle_array(manifest, 'operators/N_0based'))
    np.testing.assert_array_equal(
        operators['pv'].reshape(-1, 1),
        oracle_array(manifest, 'operators/pv'))
    np.testing.assert_array_equal(
        operators['pv'], project.G['cells']['PORV'])

    # computeTrans/processGRDECL use different arithmetic kernels.  Pair
    # order is exact and every transmissibility is within 1.3e-4 relative.
    np.testing.assert_allclose(
        operators['T'].reshape(-1, 1),
        oracle_array(manifest, 'operators/T'),
        rtol=1.3e-4, atol=1e-18)
    expected_all = oracle_array(manifest, 'operators/T_all').ravel()
    actual_all = np.asarray(operators['T_all']).ravel()
    assert actual_all.shape == expected_all.shape
    np.testing.assert_allclose(
        np.sort(actual_all), np.sort(expected_all),
        rtol=1.3e-4, atol=1e-18)
    np.testing.assert_array_equal(
        np.asarray(operators['C'].shape, dtype=np.int64).reshape(1, 2),
        oracle_array(manifest, 'operators/C_shape'))
    assert operators['C'].nnz == int(
        oracle_array(manifest, 'operators/C_nnz')[0, 0])


def test_relperm_endpoint_points_and_five_column_storage_are_exact(
        project, manifest):
    scaling = project.model.rock['krscale']
    for name, record in manifest['_records'].items():
        if name.startswith('relperm/model/'):
            _, _, branch, phase = name.split('/')
            actual = scaling[branch][phase]
        elif name.startswith('relperm/input/'):
            keyword = name.rsplit('/', 1)[1]
            from PRSTCore.hm.utils.getRelpermScalingPoints import (
                as_dict, getRelpermScalingPoints)
            actual = as_dict(getRelpermScalingPoints(project.model))[keyword]
        else:
            continue
        expected = oracle_array(manifest, name)
        actual = np.asarray(actual).reshape(expected.shape)
        assert actual.shape == expected.shape, name
        np.testing.assert_allclose(
            actual, expected, rtol=0.0, atol=0.0,
            equal_nan=True, err_msg=name)
    for branch in ('drainage', 'imbibition', 'miscible'):
        for phase in ('w', 'ow', 'og', 'g'):
            assert scaling[branch][phase].shape == (9000, 5)


def _assert_nested_equal(actual, expected, path='deck'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_nested_equal(actual[key], expected[key], path + '.' + str(key))
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            _assert_nested_equal(a, b, '%s[%d]' % (path, index))
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(actual), path
    else:
        assert actual == expected, path


def test_original_deck_and_all_app_model_state_snapshots_have_value_semantics():
    source = read_case(FahmConfig(deck_path=str(CASE_DECK), work_dir='.'))
    source.setdefault('EDIT', {})['MULTPV'] = np.ones(4)
    source['GRID']['MULTPV'] = np.ones(4)
    frozen = copy.deepcopy(source)
    result = initialize_fahm_project(source, str(CASE_PREFIX) + '.DATA')
    _assert_nested_equal(source, frozen)
    assert 'MULTPV' not in result.deck['GRID']
    assert 'MULTPV' not in result.deck['EDIT']

    assert result.model.G is not result.G
    assert result.model.rock is not result.rock
    assert result.model.fluid is not result.fluid
    assert result.model.inputdata is not result.deck
    assert result.N is not result.G['trans']['neighbors']

    base_poro = result.rock['poro'][0]
    result.model.rock['poro'][0] += 1.0
    assert result.rock['poro'][0] == base_poro
    base_x = result.G['cells']['centroids'][0, 0]
    result.model.G['cells']['centroids'][0, 0] += 1.0
    assert result.G['cells']['centroids'][0, 0] == base_x
    result.model.inputdata['PROPS']['_MODEL_ONLY'] = True
    assert '_MODEL_ONLY' not in result.deck['PROPS']
    old = result.state0['s'][0, 0]
    result.state0['sW'][0] += 0.25
    assert result.state0['s'][0, 0] == old
    result.N[0, 0] = -99
    assert result.G['trans']['neighbors'][0, 0] >= 0


def test_restart_core_fields_follow_emap_units_components_and_phase_order():
    from PRSTCore.deckformat.resultinput.convert_restart_to_states import \
        _restart_block_to_state
    ih = np.zeros(100, dtype=int)
    ih[2] = 1                    # METRIC
    ih[14] = 3                   # W/O
    ih[94] = 100                 # ECLIPSE simulator id
    ih[64:67] = [2, 1, 2020]
    block = {
        'INTEHEAD': {'values': ih},
        'PRESSURE': {'values': np.array([10.0, 20.0, 30.0])},
        'SWAT': {'values': np.array([0.1, 0.2, 0.3])},
        'RS': {'values': np.array([1.0, 2.0, 3.0])},
        'PCOW': {'values': np.array([4.0, 5.0, 6.0])},
        'XMF1': {'values': np.array([0.7, 0.8, 0.9])},
        'XMF2': {'values': np.array([0.3, 0.2, 0.1])},
    }
    G = {'cells': {'num': 2, 'eMap': np.array([2, 0])},
         'cartDims': np.array([3, 1, 1])}
    state = _restart_block_to_state(
        block, G, unit_conversion_factors('METRIC'), 0, None, False,
        include_components=True)
    np.testing.assert_array_equal(state['pressure'], [3e6, 1e6])
    np.testing.assert_array_equal(state['s'], [[0.3, 0.7], [0.1, 0.9]])
    np.testing.assert_array_equal(state['rs'], [3.0, 1.0])
    np.testing.assert_array_equal(state['pcow'], [6e5, 4e5])
    np.testing.assert_array_equal(state['x'], [[0.9, 0.1], [0.7, 0.3]])


def _aquifer_model(pressures):
    from PRSTCore.ad_core.models.aquifer_model import AquiferModel
    aquifers = np.zeros((2, 8))
    aquifers[:, 0] = [2, 1]
    aquifers[:, 1] = [1, 0]
    props = {'C': np.ones(2), 'J': np.ones(2),
             'depthaq': np.zeros(2), 'pvttbl': np.ones(2)}
    return AquiferModel(
        aquifers, {'aquid': 0, 'conn': 1, 'pvttbl': 2, 'J': 3, 'C': 4,
                   'alpha': 5, 'depthconn': 6, 'depthaq': 7},
        props, {'pressures': pressures, 'volumes': np.ones(2)})


def test_default_aquifer_pressure_prefers_sorted_restart_and_clamps_negative():
    model = type('M', (), {})()
    model.AquiferModel = _aquifer_model([np.nan, np.nan])
    state = {'pressure': np.array([100.0, 300.0]), 'aquiferSol': [
        {'num': 2, 'pressure': -5.0}, {'num': 1, 'pressure': 30.0}]}
    _fill_default_aquifer_pressures(model, state)
    np.testing.assert_array_equal(
        model.AquiferModel.initvals['pressures'], [30.0, 0.0])


def test_restart_fetkovich_aquifers_are_imported_with_cells_and_si_units():
    from PRSTCore.deckformat.resultinput.convert_restart_to_states import \
        _restart_block_to_state
    ih = np.zeros(100, dtype=int)
    ih[2], ih[14], ih[40], ih[42] = 1, 7, 2, 11
    ih[43], ih[44], ih[45], ih[47], ih[94] = 2, 3, 4, 1, 100
    iaaq = np.zeros(22)
    iaaq[[0, 11]] = 1
    block = {
        'INTEHEAD': {'values': ih},
        'PRESSURE': {'values': np.array([10.0, 20.0])},
        'SWAT': {'values': np.array([0.2, 0.3])},
        'SGAS': {'values': np.array([0.1, 0.1])},
        'IAAQ': {'values': iaaq},
        'SAAQ': {'values': np.array([0.0, 100.0, 0.0, 200.0])},
        'XAAQ': {'values': np.array([2.0, 50.0, 10.0,
                                     3.0, 60.0, 20.0])},
        'ICAQ_1': {'values': np.array([1, 1, 1, 0])},
        'ICAQ_2': {'values': np.array([2, 1, 1, 0])},
        'ACAQ_1': {'values': np.array([4.0])},
        'ACAQ_2': {'values': np.array([5.0])},
        'ACAQNUM_1': {'values': np.array([2])},
        'ACAQNUM_2': {'values': np.array([1])},
    }
    G = {'cells': {'num': 2, 'indexMap': np.array([0, 1]),
                   'eMap': slice(None)},
         'cartDims': np.array([2, 1, 1])}
    units = unit_conversion_factors('METRIC')
    state = _restart_block_to_state(
        block, G, units, 0, None, False, include_aquifers=True)
    aquifers = state['aquiferSol']
    assert [entry['num'] for entry in aquifers] == [2, 1]
    np.testing.assert_array_equal(
        [entry['cells'][0] for entry in aquifers], [0, 1])
    np.testing.assert_array_equal(
        [entry['pressure'] for entry in aquifers], [5e6, 6e6])
    np.testing.assert_allclose(
        [entry['qW'] for entry in aquifers], np.array([2.0, 3.0]) / 86400)
    np.testing.assert_allclose(
        [entry['flux'][0] for entry in aquifers],
        np.array([4.0, 5.0]) / 86400)
    np.testing.assert_array_equal(
        [entry['volume'] for entry in aquifers], [90.0, 180.0])


def test_default_aquifer_pressure_falls_back_to_connected_cell_means():
    model = type('M', (), {})()
    model.AquiferModel = _aquifer_model([np.nan, np.nan])
    _fill_default_aquifer_pressures(
        model, {'pressure': np.array([100.0, 300.0])})
    # Aquifer ids are [2, 1] and connection cells [1, 0].
    np.testing.assert_array_equal(
        model.AquiferModel.initvals['pressures'], [100.0, 300.0])


def test_swatinit_scale_uses_restart_pcow_and_replaces_nonfinite_with_one():
    class Model:
        def __init__(self):
            self.rock = {}

        @staticmethod
        def _phase_pressures(pressure, _sw, _sg, pcow_scale=None):
            assert pcow_scale is None
            return pressure - np.array([2.0, 0.0]), pressure, pressure

    model = Model()
    state = {'pressure': np.array([10.0, 10.0]),
             'sW': np.array([0.2, 0.3]), 'sG': np.zeros(2),
             'pcow': np.array([4.0, 1.0])}
    _apply_fahm_swatinit_scale(
        model, state, {'PROPS': {'SWATINIT': np.array([0.2, 0.3])}})
    np.testing.assert_array_equal(model.rock['pcowScale'], [2.0, 1.0])


def test_capillary_endpoint_scaling_is_consumed_by_phase_pressures():
    """BlackOilCapillaryPressure: SWLPC/SGLPC remap, PCW/PCG rescale."""
    from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel
    from PRSTCore.hm.utils.imposeCapPressScaling import imposeCapPressScaling

    G = {'cells': {'num': 1, 'indexMap': np.array([0])}}
    model = GenericBlackOilModel(G, {}, {}, water=True, oil=True, gas=True)
    model.inputdata = {
        'RUNSPEC': {'ENDSCALE': ['NODIR', 'REVERS', 1, 20, 0]},
        'PROPS': {
            'SCALECRS': ['NO'],
            'SWOF': np.array([[0.2, 0.0, 1.0, 10.0],
                              [0.5, 0.2, 0.5, 4.0],
                              [0.8, 1.0, 0.0, 0.0]]),
            'SGOF': np.array([[0.0, 0.0, 1.0, 2.0],
                              [0.3, 0.2, 0.5, 5.0],
                              [0.8, 1.0, 0.0, 10.0]]),
        },
    }
    imposeCapPressScaling(model, SWLPC=0.3, PCW=20.0,
                          SGLPC=0.1, PCG=4.0)

    pW, pO, pG = model._phase_pressures(
        np.array([100.0]), np.array([0.3]), np.array([0.1]))
    # Both saturations map to the table connate point. PCW/PCG then impose
    # the requested capillary pressure at that point.
    np.testing.assert_allclose(pW, [80.0])
    np.testing.assert_allclose(pO, [100.0])
    np.testing.assert_allclose(pG, [104.0])

    from PRSTCore.ad_core.adi import SparseADI
    p_ad = SparseADI.constant(np.array([100.0]), 2)
    sw_ad = SparseADI.variable(np.array([0.3]), 2, 0)
    sg_ad = SparseADI.variable(np.array([0.1]), 2, 1)
    pW_ad, _, pG_ad = model._phase_pressures_adi(
        p_ad, sw_ad, sg_ad)
    np.testing.assert_allclose(pW_ad.val, [80.0])
    np.testing.assert_allclose(pG_ad.val, [104.0])
    np.testing.assert_allclose(pW_ad.jac.toarray(), [[48.0, 0.0]])
    np.testing.assert_allclose(pG_ad.jac.toarray(), [[0.0, 160.0 / 7.0]])

    # SWATINIT's pcowScale has source-code precedence over PCW.
    model.rock['pcowScale'] = np.array([3.0])
    pW, _, _ = model._phase_pressures(
        np.array([100.0]), np.array([0.3]), np.array([0.1]))
    np.testing.assert_allclose(pW, [70.0])
