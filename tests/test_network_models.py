"""Tests for the network-models port (MRST modules/network-models).

``Network`` builds the graph a GPSNet is generated from, so a missing or
extra edge is a different reservoir model, not a cosmetic difference.
These pin the four topology types against ``Network.m`` and cover the
plotting entry point, which had no port at all.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')

from PRSTCore.network_models.network import Network


def _grid(ncell=8):
    """A grid stub with just the centroids Network reads."""
    return {'griddim': 3,
            'cells': {'num': ncell,
                      'centroids': np.arange(3 * ncell,
                                             dtype=float).reshape(ncell, 3)}}


def _wells():
    """Two wells, the first with two perforations.

    The multi-perforation well is the point: a node is a perforation, not
    a well, so this is where the topology types can differ.
    """
    return [{'name': 'I1', 'cells': [0, 1]}, {'name': 'P1', 'cells': [7]}]


# ------------------------------------------------------------ topology --

def test_all_to_all_is_the_complete_graph_over_perforations():
    """``graph(ones(numNodes) - eye(numNodes))``.

    Every pair of nodes is joined, including two perforations of the same
    well. Excluding same-well pairs -- which this used to do -- drops an
    edge for every multi-perforation well, and GPSNet then builds fewer
    flow paths than MRST would.
    """
    net = Network(_wells(), _grid(), type='all_to_all')
    assert net.num_nodes == 3
    assert net.num_edges == 3
    assert sorted(net.network.edges()) == [(0, 1), (0, 2), (1, 2)]


def test_all_to_all_scales_as_n_choose_two():
    wells = [{'name': 'W%d' % i, 'cells': [i]} for i in range(5)]
    net = Network(wells, _grid(), type='all_to_all')
    assert net.num_edges == 5 * 4 // 2


def test_injectors_to_producers_joins_every_pair_across_the_two_sets():
    wells = [{'name': 'I1', 'cells': [0]}, {'name': 'P1', 'cells': [7]},
             {'name': 'P2', 'cells': [4]}]
    net = Network(wells, _grid(), type='injectors_to_producers',
                  injectors=[0], producers=[1, 2])
    assert sorted(net.network.edges()) == [(0, 1), (0, 2)]


def test_naming_injectors_and_producers_selects_that_type():
    """``if ~isempty(injectors) && ~isempty(producers)`` overrides type."""
    wells = [{'name': 'I1', 'cells': [0]}, {'name': 'P1', 'cells': [7]}]
    net = Network(wells, _grid(), type='all_to_all',
                  injectors=[0], producers=[1])
    assert net.type == 'injectors_to_producers'


def test_supplying_edges_selects_the_user_defined_type():
    net = Network(_wells(), _grid(), edges=np.array([[0, 2], [1, 2]]))
    assert net.type == 'user_defined_edges'
    assert sorted(net.network.edges()) == [(0, 2), (1, 2)]


def test_an_edge_naming_a_node_that_does_not_exist_is_rejected():
    with pytest.raises(ValueError, match='non-existing'):
        Network(_wells(), _grid(), type='user_defined_edges',
                edges=np.array([[0, 2], [1, 2], [9, 0]]))


def test_a_node_with_no_edge_is_rejected():
    """``all(a>0)``: an unreferenced node has no flow path, so it would
    sit in the network contributing nothing while still being counted."""
    with pytest.raises(ValueError, match='at least one edge'):
        Network(_wells(), _grid(), type='user_defined_edges',
                edges=np.array([[0, 1]]))


def test_an_unknown_type_is_rejected():
    with pytest.raises(ValueError, match='not implemented'):
        Network(_wells(), _grid(), type='no_such_topology')


# ------------------------------------------------------------- nodes --

def test_each_perforation_becomes_a_node_carrying_its_well_and_centroid():
    net = Network(_wells(), _grid(), type='all_to_all')
    nodes = net._nodes_data
    assert [n['well'] for n in nodes] == [0, 0, 1]
    assert [n['subwell'] for n in nodes] == [0, 1, 0]
    assert [n['name'] for n in nodes] == ['I1', 'I1', 'P1']
    centroids = _grid()['cells']['centroids']
    assert nodes[2]['XData'] == centroids[7, 0]
    assert nodes[2]['ZData'] == centroids[7, 2]


# ---------------------------------------------------------- plotting --

def test_plot_network_draws_each_layout():
    net = Network(_wells(), _grid(), type='all_to_all')
    for plottype in ('default', 'spacegraph', 'circle'):
        ax, colors = net.plot_network(plottype, on_grid=False)
        assert ax is not None
        assert len(colors) == net.num_edges


def test_plot_network_without_colours_returns_none_for_them():
    net = Network(_wells(), _grid(), type='all_to_all')
    _ax, colors = net.plot_network(on_grid=False, colors=False)
    assert colors is None


def test_plot_network_rejects_data_of_the_wrong_length():
    net = Network(_wells(), _grid(), type='all_to_all')
    with pytest.raises(ValueError, match='does not match number of edges'):
        net.plot_network(data=[1.0], on_grid=False)


def test_plot_network_rejects_an_unknown_plot_type():
    net = Network(_wells(), _grid(), type='all_to_all')
    with pytest.raises(ValueError, match='Plot type not defined'):
        net.plot_network('histogram', on_grid=False)


def test_asking_for_transmissibilities_a_network_lacks_is_an_error():
    """Only a flow-diagnostics network carries T/pv. MRST asserts on the
    edge-table width; plotting nothing instead would look like a network
    with no connections."""
    net = Network(_wells(), _grid(), type='all_to_all')
    with pytest.raises(ValueError, match='no transmissibilities'):
        net.plot_network('transmissibility', on_grid=False)
    with pytest.raises(ValueError, match='no pore volumes'):
        net.plot_network('porevolume', on_grid=False)


def test_edge_widths_scale_with_the_data_and_survive_an_all_zero_field():
    from PRSTCore.network_models.network import _scaled_widths
    assert np.allclose(_scaled_widths([1.0, 2.0, 4.0], 6.0), [1.5, 3.0, 6.0])
    # ``maxWidth*data/max(data)`` is 0/0 here; MATLAB would draw NaN-wide
    # lines, which render as nothing at all.
    assert np.allclose(_scaled_widths([0.0, 0.0], 6.0), [6.0, 6.0])


# ------------------------------------------------- gpsNetSimulationSetup --

class _GPSNetStub:
    def __init__(self, wells):
        self.W = wells
        self.model = {'G': None}
        self.state0 = {'pressure': np.zeros(3)}


def _setup_case():
    from PRSTCore.network_models.utils import gps_net_simulation_setup

    network_well = {'name': 'I1', 'cells': [0, 1], 'refDepth': 1234.0,
                    'dZ': [0.0, 5.0], 'sign': 1, 'compi': [1, 0],
                    'type': 'rate', 'val': 1.0, 'status': True,
                    'WI': [1e-12, 2e-12]}
    schedule = {'step': {'val': [1.0, 2.0], 'control': [0, 0]},
                'control': [{'W': [{'name': 'I1', 'cells': [40, 41, 42],
                                    'type': 'bhp', 'val': 3e7,
                                    'status': False,
                                    'WI': [3e-12, 4e-12, 5e-12],
                                    'group': 'G1'}]}],
                'extraField': 'kept'}
    return (gps_net_simulation_setup(_GPSNetStub([network_well]), schedule),
            schedule)


def test_the_setup_takes_its_control_from_the_fine_schedule():
    """``Wi.type/val/status <- schedule.control(n).W(i)``."""
    from PRSTCore.network_models.utils import gps_net_simulation_setup
    _, schedule = _setup_case()
    setup = gps_net_simulation_setup(
        _GPSNetStub([{'name': 'I1', 'cells': [0, 1], 'refDepth': 1234.0,
                      'dZ': [0.0, 5.0], 'sign': 1, 'type': 'rate',
                      'val': 1.0, 'status': True, 'WI': [1e-12, 2e-12]}]),
        schedule)
    w = setup['schedule']['control'][0]['W'][0]
    assert w['type'] == 'bhp'
    assert w['val'] == 3e7
    assert w['status'] is False


def test_the_setup_sums_the_fine_connection_transmissibilities():
    """``Wi.WI = sum(schedule.control(n).W(i).WI)``: the network well has
    one connection where the fine well had several."""
    from PRSTCore.network_models.utils import gps_net_simulation_setup
    _, schedule = _setup_case()
    setup = gps_net_simulation_setup(
        _GPSNetStub([{'name': 'I1', 'cells': [0], 'type': 'rate',
                      'val': 1.0, 'status': True, 'WI': [1e-12]}]), schedule)
    assert setup['schedule']['control'][0]['W'][0]['WI'] == pytest.approx(1.2e-11)


def test_every_field_of_the_network_well_reaches_the_schedule():
    """``for fn = fieldnames(Wi)', schedule.W(i).(fn) = Wi.(fn)``.

    Rebuilding the well from a chosen field list -- which this used to do
    -- drops refDepth and dZ, which the well model then defaults without
    saying so. Fields only the fine well has must survive too.
    """
    setup, _ = _setup_case()
    w = setup['schedule']['control'][0]['W'][0]
    assert w['cells'] == [0, 1]           # the network well's, not the fine one's
    assert w['refDepth'] == 1234.0
    assert w['dZ'] == [0.0, 5.0]
    assert w['group'] == 'G1'             # present only on the fine well


def test_the_caller_s_schedule_is_not_modified():
    """MATLAB copies the struct by value; a shared reference here would
    let a second call see the first call's wells."""
    setup, schedule = _setup_case()
    assert schedule['control'][0]['W'][0]['cells'] == [40, 41, 42]
    assert setup['schedule']['extraField'] == 'kept'


# ---------------------------------------------------- makeRandomTraining --

def _training_problem(nsteps=12):
    return {'name': 'case',
            'schedule': {'step': {'val': [30.0] * nsteps,
                                  'control': [0] * nsteps},
                         'control': [{'W': [
                             {'name': 'I1', 'type': 'rate', 'val': 100.0,
                              'sign': 1, 'lims': {'bhp': float('inf')}},
                             {'name': 'P1', 'type': 'bhp', 'val': 2.5e7,
                              'sign': -1, 'lims': {'bhp': 2.5e7}}]}]}}


def test_random_training_groups_report_steps_in_fours():
    """``ceil(0.25*(1:nstep))``: one new control per *four* steps.

    Numbering one control per step gives four times as many, each with
    its own random perturbation -- a different training set, not a
    differently-indexed one.
    """
    from PRSTCore.network_models.utils import make_random_training

    out = make_random_training(_training_problem(12), 0.1, 0.05, False)
    assert len(out['schedule']['control']) == 3
    assert list(np.asarray(out['schedule']['step']['control']).ravel()) == \
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert out['name'] == 'case_rand'


def test_the_first_control_of_a_block_is_left_unperturbed():
    from PRSTCore.network_models.utils import make_random_training

    out = make_random_training(_training_problem(12), 0.1, 0.05, False)
    controls = out['schedule']['control']
    assert controls[0]['W'][1]['val'] == 2.5e7
    assert controls[1]['W'][1]['val'] != 2.5e7


def test_a_finite_limit_follows_the_perturbed_target_and_an_infinite_one_does_not():
    """``~isempty(lims) && ~isinf(lims.bhp)``. An infinite limit means
    unconstrained; writing the new target into it would turn that into a
    hard constraint at exactly the target."""
    from PRSTCore.network_models.utils import make_random_training

    out = make_random_training(_training_problem(12), 0.1, 0.05, False)
    for ctrl in out['schedule']['control']:
        assert ctrl['W'][1]['lims']['bhp'] == ctrl['W'][1]['val']
        assert np.isinf(ctrl['W'][0]['lims']['bhp'])


def test_random_training_is_reproducible():
    """``rng(1)`` -- the training set has to be the same on every run, or
    a match cannot be repeated."""
    from PRSTCore.network_models.utils import make_random_training

    a = make_random_training(_training_problem(12), 0.1, 0.05, False)
    b = make_random_training(_training_problem(12), 0.1, 0.05, False)
    va = [c['W'][1]['val'] for c in a['schedule']['control']]
    vb = [c['W'][1]['val'] for c in b['schedule']['control']]
    assert va == vb


# --------------------------------------------------- modifyNorneTest --

def _norne_test(nsteps=9):
    """The shape ``norne_simple_wo`` produces: nine wells, P1 seventh."""
    wells = [{'name': 'I%d' % (i + 1), 'type': 'rate', 'val': 100.0 * (i + 1),
              'sign': 1, 'compi': [1, 0], 'status': True} for i in range(6)]
    wells.append({'name': 'P1', 'type': 'bhp', 'val': 2.5e7, 'sign': -1,
                  'compi': [0, 1], 'status': True})
    wells.append({'name': 'P2', 'type': 'bhp', 'val': 2.4e7, 'sign': -1,
                  'compi': [0, 1], 'status': True})
    return {'name': 'norne', 'schedule': {
        'step': {'val': [30.0] * nsteps, 'control': [0] * nsteps},
        'control': [{'W': wells}]}}


def test_case_zero_changes_nothing_but_still_copies():
    from PRSTCore.network_models.examples.utils import modify_norne_test

    original = _norne_test()
    out = modify_norne_test(original, 0)
    assert out == original
    out['schedule']['control'][0]['W'][0]['val'] = -1.0
    assert original['schedule']['control'][0]['W'][0]['val'] == 100.0


def test_case_one_perturbs_rates_and_pressures_by_their_own_factors():
    """``(.5+rand)`` on a rate, ``(.9+0.2*rand)`` on a bhp -- 25% and 10%
    about the original, not the same spread for both."""
    from PRSTCore.network_models.examples.utils import modify_norne_test

    out = modify_norne_test(_norne_test(), 1)
    assert out['name'] == 'test1'
    W = out['schedule']['control'][0]['W']
    for i in range(6):
        base = 100.0 * (i + 1)
        assert 0.5 * base <= W[i]['val'] <= 1.5 * base
        assert W[i]['val'] != base
    assert 0.9 * 2.5e7 <= W[6]['val'] <= 1.1 * 2.5e7


def test_case_one_is_reproducible():
    from PRSTCore.network_models.examples.utils import modify_norne_test

    a = modify_norne_test(_norne_test(), 1)['schedule']['control'][0]['W']
    b = modify_norne_test(_norne_test(), 1)['schedule']['control'][0]['W']
    assert [w['val'] for w in a] == [w['val'] for w in b]


def test_case_two_shuts_the_dominant_producer_for_the_last_third():
    from PRSTCore.network_models.examples.utils import modify_norne_test

    out = modify_norne_test(_norne_test(9), 2)
    assert out['name'] == 'test2'
    assert len(out['schedule']['control']) == 2
    assert out['schedule']['control'][1]['W'][6]['status'] is False
    assert out['schedule']['control'][0]['W'][6]['status'] is True
    # ``2*end/3 : end`` -- 1-based inclusive, so steps 6..9 of nine.
    assert list(out['schedule']['step']['control']) == [0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_case_four_shuts_it_for_the_middle_third_only():
    from PRSTCore.network_models.examples.utils import modify_norne_test

    out = modify_norne_test(_norne_test(9), 4)
    assert out['name'] == 'test4'
    # ``end/3 : 2*end/3`` -- steps 3..6 of nine.
    assert list(out['schedule']['step']['control']) == [0, 0, 1, 1, 1, 1, 0, 0, 0]


def test_case_three_converts_the_first_two_injectors_into_producers():
    from PRSTCore.network_models.examples.utils import modify_norne_test

    out = modify_norne_test(_norne_test(), 3)
    assert out['name'] == 'test3'
    W = out['schedule']['control'][0]['W']
    for i in range(2):
        # Control, composition *and* sign all follow P1: converting a well
        # means converting what it produces, not only its target.
        assert W[i]['type'] == 'bhp'
        assert W[i]['val'] == 2.5e7
        assert W[i]['compi'] == [0, 1]
        assert W[i]['sign'] == -1
        assert W[i]['name'] == 'PI%d' % (i + 1)
    assert W[2]['type'] == 'rate'          # the untouched injectors
    assert W[6]['status'] is False


def test_a_step_count_that_will_not_divide_into_thirds_is_reported():
    """MATLAB rejects the fractional index outright; rounding it would
    shut the well over a different stretch of the simulation."""
    from PRSTCore.network_models.examples.utils import modify_norne_test

    with pytest.raises(ValueError, match='multiple of'):
        modify_norne_test(_norne_test(10), 2)


def test_an_unknown_case_number_is_rejected():
    from PRSTCore.network_models.examples.utils import modify_norne_test

    with pytest.raises(ValueError, match='0 to 4'):
        modify_norne_test(_norne_test(), 5)
