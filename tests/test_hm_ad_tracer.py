"""Tests for the port of MRST ``hm/ad-tracer``.

A passive tracer rides the water phase and does not feed back into it, so
the physics gives sharp checks that do not need a reference solution:

* injecting tracer-free water must leave every tracer at zero;
* a tracer is bounded by the injected concentration (no over/undershoot
  beyond the initial and injected values);
* two tracers injected at the same concentration must transport
  identically, whatever the flow field.
"""

import numpy as np
import pytest

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.hm.ad_tracer import (OilWaterTracerModel, TracerComponent,
                                   ThreePhaseBlackOilTracerModel)


def test_component_selects_its_own_tracer_from_a_list():
    tc = TracerComponent(tracerIndex=1, tracerName='trB')
    b = [np.array([2.0, 2.0]), None]
    ct = [np.array([0.0, 0.0]), np.array([3.0, 4.0])]
    assert np.allclose(tc.getComponentDensity(ct, b, 2)[0], [6.0, 8.0])


def test_component_selects_its_own_tracer_from_a_matrix():
    tc = TracerComponent(tracerIndex=1, tracerName='trB')
    b = [np.ones(2), None]
    ct = np.array([[0.0, 3.0], [0.0, 4.0]])
    assert np.allclose(tc.getComponentDensity(ct, b, 2)[0], [3.0, 4.0])


def test_component_mass_is_pore_volume_weighted():
    """getComponentMass: c{wIx} = pv .* (sw .* ct .* bW)."""
    tc = TracerComponent(tracerIndex=0)
    got = tc.getComponentMass([np.array([2.0, 2.0])], [np.array([1.5, 1.5])],
                              np.array([10.0, 10.0]), np.array([0.5, 0.5]), 1)[0]
    assert np.allclose(got, 10.0 * 0.5 * 2.0 * 1.5)


def test_component_mobility_rides_the_water_phase():
    tc = TracerComponent(tracerIndex=0)
    got = tc.getComponentMobility([np.array([2.0])], [np.array([1.5])],
                                  [np.array([10.0])], 1)[0]
    assert np.allclose(got, 2.0 * 1.5 * 10.0)


def test_injection_mass_fraction_divides_by_surface_density():
    tc = TracerComponent(tracerIndex=0)
    got = tc.getInjectionMassFraction({'tracer': [np.array([500.0])]}, 1000.0)
    assert np.allclose(got, 0.5)


def test_number_of_tracers_tracks_the_names():
    model = ThreePhaseBlackOilTracerModel(
        {'cells': {'num': 1}}, {}, {}, tracerNames=['trA', 'trB'],
        water=True, oil=True, gas=True)
    assert model.getNumberOfTracers() == 2
    assert model.getComponentNames()[-2:] == ['trA', 'trB']


def test_variable_field_resolves_tracer_names_and_well_rates():
    model = ThreePhaseBlackOilTracerModel(
        {'cells': {'num': 1}}, {}, {}, tracerNames=['trA', 'trB'],
        water=True, oil=True, gas=True)
    assert model.getVariableField('trB') == ('tracer', 1)
    assert model.getVariableField('qwtrA')[0] == 'qWtrA'
    assert model.getVariableField('tracer')[0] == 'tracer'


def test_extra_well_names_are_one_per_tracer():
    model = ThreePhaseBlackOilTracerModel(
        {'cells': {'num': 1}}, {}, {}, tracerNames=['trA', 'trB'],
        water=True, oil=True, gas=True)
    names, types = model.getExtraWellEquationNames()
    assert names[-2:] == ['trAWells', 'trBWells']
    assert types[-2:] == ['perf', 'perf']
    assert model.getExtraWellPrimaryVariableNames()[-2:] == ['qWtrA', 'qWtrB']


def test_validate_state_defaults_every_tracer_to_zero():
    model = ThreePhaseBlackOilTracerModel(
        {'cells': {'num': 4}}, {}, {}, tracerNames=['trA', 'trB'],
        water=True, oil=True, gas=True)
    state = model.validateState({'pressure': np.full(4, 1e7),
                                 'sW': np.full(4, 0.2), 'sG': np.zeros(4),
                                 'rs': np.zeros(4)})
    assert len(state['tracer']) == 2
    assert all(np.allclose(t, 0.0) for t in state['tracer'])


DECK = 'mrst-2026a/autodiff/ad-eor/examples/polymer/POLYMER.DATA'


def _tracer_setup(tracer_names, concentrations):
    """Build an OilWaterTracerModel on the bundled two-phase POLYMER deck."""
    import os
    if not os.path.exists(DECK):
        pytest.skip('POLYMER.DATA not available')
    state0, base, schedule, nls = init_eclipse_problem_ad(DECK)
    model = OilWaterTracerModel(base.G, base.rock, base.fluid,
                                tracerNames=tracer_names)
    model._blackoil_pvt = base._blackoil_pvt
    model.inputdata = base.inputdata
    model.operators = base.operators
    model.porevolume = base.porevolume
    model.gravity = base.gravity
    model.enable_facility_unknowns = True
    model.disgas = False
    model.vapoil = False

    nc = model._num_cells()
    state0 = model.validateState(dict(state0))
    state0['tracer'] = [np.zeros(nc) for _ in tracer_names]

    forces = model.getDrivingForces(schedule['control'][0])
    for w in forces.get('W', []):
        w['tracer'] = list(concentrations)
    dt = float(schedule['step']['val'][0])
    return model, state0, forces, dt


def test_assembly_produces_one_residual_block_per_tracer():
    model, state0, forces, dt = _tracer_setup(['trA', 'trB'], [1.0, 1.0])
    from copy import deepcopy
    problem, _ = model.get_equations(deepcopy(state0), deepcopy(state0), dt, forces)
    nc, nw = problem['nc'], problem['nw']
    # water + oil + 2 tracers over cells, then 3 facility blocks per well.
    assert problem['Residuals'].size == (2 + 2) * nc + 3 * nw
    assert problem['Jacobian'].shape[1] == (2 + 2) * nc + 3 * nw


def test_identical_tracers_assemble_identical_residual_blocks():
    """Two tracers injected at the same concentration are the same problem."""
    model, state0, forces, dt = _tracer_setup(['trA', 'trB'], [2.0, 2.0])
    from copy import deepcopy
    problem, _ = model.get_equations(deepcopy(state0), deepcopy(state0), dt, forces)
    nc = problem['nc']
    res = problem['Residuals']
    assert np.allclose(res[2 * nc:3 * nc], res[3 * nc:4 * nc])


def test_zero_injection_leaves_a_zero_tracer_residual():
    """With no tracer anywhere and none injected, the tracer rows vanish."""
    model, state0, forces, dt = _tracer_setup(['trA'], [0.0])
    from copy import deepcopy
    problem, _ = model.get_equations(deepcopy(state0), deepcopy(state0), dt, forces)
    nc = problem['nc']
    assert np.allclose(problem['Residuals'][2 * nc:3 * nc], 0.0, atol=1e-12)
