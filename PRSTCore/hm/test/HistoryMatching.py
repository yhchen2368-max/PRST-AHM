"""Port of MRST ``HistoryMatching.m`` (mrst-2026a/hm/test).

An adjoint-gradient history match against an external simulator. Like
the MATLAB it is a template: the three paths at the top are placeholders
and must be filled in before it will run.

The point of the NOSIM pass is worth stating, because it is the reason
this script is shaped the way it is. G, rock and state0 are built from
the simulator's *own* INIT/EGRID/UNRST output rather than from the deck,
so the model PRSTCore differentiates is the same model the simulator
integrates. Deriving them independently from the deck would leave the
adjoint gradient describing a slightly different problem than the one
being scored.

Contrast :mod:`PRSTCore.hm.APP.fahm`, which scores the same way but takes
its gradient by finite differences and so needs no model at all.

Run as ``python -m PRSTCore.hm.test.HistoryMatching`` once configured.
"""

import os

import numpy as np

# Fill these in.
PATH_DATA = r'path\to\eclipse\data'
NAME_DATA = 'dataname'
PATH_BASE = r'path\to\base\directory'
PATH_WORK = r'path\to\work\directory'
SIMULATOR = r'C:\Users\dell\Desktop\RFD2022\tNavigator22.1.exe'


def main(path_data=PATH_DATA, name_data=NAME_DATA, path_base=PATH_BASE,
         path_work=PATH_WORK, simulator=SIMULATOR):
    """Run the match. Returns ``(v, u, history)``."""
    from PRSTCore.ad_core.solvers.mumps_solver_ad import MUMPSSolverAD
    from PRSTCore.ad_props.get_normalization_factors import \
        get_normalization_factors
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    from PRSTCore.deckformat.deckinput.convert_deck_units import \
        convert_deck_units
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import \
        read_eclipse_deck
    from PRSTCore.deckformat.deckoutput.write_deck import write_deck
    from PRSTCore.deckformat.resultinput.convert_restart_to_states import \
        convert_restart_to_states
    from PRSTCore.deckformat.resultinput.init_grid_from_eclipse_output import \
        init_grid_from_eclipse_output
    from PRSTCore.deckformat.resultinput.process_eclipse_restart_spec import \
        process_eclipse_restart_spec
    from PRSTCore.deckformat.resultinput.read_eclipse_output_file_unfmt import \
        read_eclipse_output_file_unfmt
    from PRSTCore.hm.utils.evaluate.evaluateMatchFromEclipseRun import \
        evaluateMatchFromEclipseRun
    from PRSTCore.hm.utils.getRelpermScalingPoints import \
        getRelpermScalingPoints
    from PRSTCore.hm.utils.observed.getObservedFromSchedule import \
        getObservedFromSchedule
    from PRSTCore.hm.utils.processEclipseDeck import processEclipseDeck
    # ``matchObservedOW``: the complete port, including the
    # ComputePartials branch the adjoint needs.  Not
    # optimization.objectives.match_observed_ow, which is an earlier
    # partial port of the same MATLAB with no partials at all.
    from PRSTCore.hm.utils.evaluate.matchObservedOW import matchObservedOW
    from PRSTCore.optimization.optim.optimize_bound_constrained import \
        optimize_bound_constrained
    from PRSTCore.optimization.utils.parameters import (
        add_parameter, get_scaled_parameter_vector,
        update_setup_from_scaled_parameters)

    prefix_data = os.path.join(path_data, name_data)
    deck = read_eclipse_deck(prefix_data + '.DATA')
    deck = convert_deck_units(deck)
    deck = processEclipseDeck(deck)

    # ---- write the base case and run it with NOSIM ---------------------
    prefix_base = os.path.join(path_base, name_data)
    if 'eclipse' in simulator.lower():
        write_deck(deck, path_base, filename=name_data, NOSIM=True)
        command = 'eclrun eclipse ' + prefix_base + '.DATA'
    elif 'tnavigator' in simulator.lower():
        write_deck(deck, path_base, filename=name_data, NOSIM=False)
        command = (simulator + ' --no-dump-res --ecl-root -e -i -r -u '
                   '--no-gui --ignore-lock --use-gpu --stop-step=1 '
                   + prefix_base + '.DATA')
    else:
        raise ValueError('Unsupported simulator: %r' % simulator)
    os.system(command)

    # ---- grid, rock and initial state, from the simulator's output -----
    init = read_eclipse_output_file_unfmt(prefix_base + '.INIT')
    grid = read_eclipse_output_file_unfmt(prefix_base + '.EGRID')
    G, rock, N, _ = init_grid_from_eclipse_output(init, grid,
                                                  output_sim_grid=False)

    rsspec = process_eclipse_restart_spec(prefix_base, 'all')
    # ``[states, ~] = convertRestartToStates(...)`` -- the second output
    # is the restart bookkeeping, which nothing here reads.
    states, _ = convert_restart_to_states(
        prefix_base, G, restart_info=rsspec, split_wells_on_sign_change=False,
        remove_closed_wells=False, remove_crossflow=False,
        include_well_sols=True, include_aquifers=True)
    state0 = states[0]

    _fill_defaulted_aquifer_pressure(deck, state0)

    # ---- model ---------------------------------------------------------
    from PRSTCore.ad_core.initialization.init_deck_adi_fluid import \
        init_deck_adi_fluid
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        _select_model_from_deck
    from PRSTCore.ad_core.operators_tpfa import setup_operators_tpfa

    # MRST passes G here so initDeckADIFluid can call handleRegions and
    # build per-region PVT/relperm dispatch. PRSTCore's port takes only
    # the deck and does no region handling, so a deck with more than one
    # SATNUM or PVTNUM region will not be represented faithfully. Flagged
    # rather than worked around, because the fix belongs in the fluid
    # initialiser, not here.
    fluid = init_deck_adi_fluid(deck)
    model = _select_model_from_deck(G, rock, fluid, deck)
    model.operators = setup_operators_tpfa(G, rock, neighbors=N)

    scaling = getRelpermScalingPoints(model)
    # MATLAB expands the pairs: imposeRelpermScaling(model, scaling{:}).
    model = impose_relperm_scaling(model, **dict(scaling))

    if 'SWATINIT' in (deck.get('PROPS') or {}):
        _apply_swatinit_pc_scaling(model, state0)
    model = model.validateModel()

    # ---- schedule and observed data -----------------------------------
    from PRSTCore.nwm._deps import convertDeckScheduleToMRST
    schedule = convertDeckScheduleToMRST(model, deck, ReorderStrategy='origin')
    observed = getObservedFromSchedule(schedule)

    # ---- parameters ----------------------------------------------------
    setup = {'model': model, 'schedule': schedule, 'state0': state0}
    pv = model.operators['pv'] > 0
    kx = model.rock['perm'][:, 0] > 0
    ky = model.rock['perm'][:, 1] > 0
    kz = model.rock['perm'][:, 2] > 0

    params = []
    params = add_parameter(params, setup, name='porevolume',
                           relative_limits=[0.8, 1.2], scaling='linear',
                           uniform_limits=False, subset=pv)
    for name, sub in (('permx', kx), ('permy', ky), ('permz', kz)):
        params = add_parameter(params, setup, name=name,
                               relative_limits=[0.01, 100], scaling='log',
                               uniform_limits=False, subset=sub)
    for name, lims in (('swcr', [1.0, 1.3]), ('sowcr', [0.5, 1.3]),
                       ('krw', [0.5, 2.0]), ('kro', [0.5, 2.0])):
        params = add_parameter(params, setup, name=name,
                               relative_limits=lims, scaling='linear',
                               uniform_limits=False)
    u0 = get_scaled_parameter_vector(setup, params)

    # ---- objective -----------------------------------------------------
    beta = get_normalization_factors(observed)

    weighting = {'WaterRateWeight': beta['ww'],
                 'OilRateWeight': beta['wo'], 'BHPWeight': 0}

    def objh(model_, states_, schedule_, observed_, tt, tstep, state):
        return matchObservedOW(
            model_, states_, schedule_, observed_, **weighting,
            ComputePartials=tt, tStep=tstep, state=state, from_states=False)

    solver = MUMPSSolverAD()

    def func(u):
        return evaluateMatchFromEclipseRun(
            u, objh, setup, params, observed, deck, path_work, name_data,
            AdjointLinearSolver=solver, simulator=simulator)

    v, u, history = optimize_bound_constrained(
        u0, func, grad_tol=1e-6, obj_change_tol=1e-8, max_it=100,
        line_search_max_it=10, lbfgs_num=10,
        save_history=name_data + '_HistoryMatching')

    _report(setup, params, u, schedule, observed, path_work, name_data)
    return v, u, history


def _fill_defaulted_aquifer_pressure(deck, state0):
    """Fill in AQUFETP pressures the deck left defaulted.

    ECLIPSE computes them and writes every aquifer state to the restart
    file; tNavigator writes no aquifer information at all, so with
    tNavigator the pressure has to be specified in the deck.
    """
    aqu = (deck.get('SOLUTION') or {}).get('AQUFETP')
    if aqu is None:
        return
    aqu = np.atleast_2d(np.asarray(aqu, dtype=float))
    bad = ~np.isfinite(aqu[:, 2])
    if not np.any(bad):
        return
    if 'aquiferSol' not in state0:
        raise ValueError('Aquifer pressure is not specified!')

    ID1 = aqu[bad, 0].astype(int)
    ID2 = np.array([int(a['num']) for a in state0['aquiferSol']])
    p = np.array([float(a['pressure']) for a in state0['aquiferSol']])
    p[p < 0] = 0.0
    lookup = {v: i for i, v in enumerate(ID2.tolist())}
    for i in ID1:
        if i in lookup:
            aqu[i, 2] = p[lookup[i]]
    deck['SOLUTION']['AQUFETP'] = aqu


def _apply_swatinit_pc_scaling(model, state0):
    """Scale capillary pressure so the model reproduces SWATINIT."""
    pc = model.getProp(state0, 'capillarypressure')
    pcow = -np.asarray(pc[0], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mult = np.asarray(state0['pcow'], dtype=float) / pcow
    mult[~np.isfinite(mult)] = 1.0
    model.rock['pcowScale'] = mult


def _report(setup, params, u, schedule, observed, path_work, name_data):
    """Plot matched against observed well curves."""
    from PRSTCore.ad_core.plotting.plot_well_sols import plot_well_sols
    from PRSTCore.hm.utils.addFieldRates import addFieldRates
    from PRSTCore.hm.utils.evaluate.getEclipseSimResults import \
        getEclipseSimResults
    from PRSTCore.optimization.utils.parameters import \
        update_setup_from_scaled_parameters

    setupNew = update_setup_from_scaled_parameters(setup, params, u)
    _, wellSols = getEclipseSimResults(path_work, name_data, setupNew)
    wellSols_obs = [o['wellSol'] for o in observed]

    wellSols = addFieldRates(wellSols)
    wellSols_obs = addFieldRates(wellSols_obs)
    T = np.cumsum(np.asarray(schedule['step']['val'], dtype=float))
    plot_well_sols([wellSols_obs, wellSols], T,
                   linestyles=[':', '-'], markerstyles=['o', ''],
                   datasetnames=['Observed', 'Matched'])


if __name__ == '__main__':
    main()
