"""Port of MRST ``CGNetTraining.m`` (mrst-2026a/hm/test).

Calibrates a coarse (CGNet) model against a fine model's simulated well
response. The fine model is read from a simulator's own INIT/EGRID/UNRST
output, coarsened by a 10x10x1 partition, and its pore volumes,
transmissibilities, connection transmissibilities and saturation
endpoints are then tuned until the coarse model reproduces the fine
model's well curves.

The weighting step is worth noting: each phase's weight is set to
``1/sqrt(v)`` where ``v`` is that phase's own mismatch at the starting
point. That puts every term at roughly 1 initially, so no single phase
dominates the objective merely by having larger numbers.

Like the MATLAB this is a template -- ``PATH`` and ``NAME`` are
placeholders. It also needs ``matchObservedOWG``, which MRST calls here
but never defines; see :mod:`PRSTCore.hm.utils.evaluate.matchObservedOWG`.
"""

import os

import numpy as np

# Fill these in.
PATH = r'D:\BaiduSyncdisk\AB model_i'
NAME = 'forecast'
#: The MATLAB hard-codes this many steps; None uses all of them.
N_STEPS = 465


def build_fine_model(path=PATH, name=NAME):
    """Fine model, schedule, observed data and initial state."""
    from PRSTCore.ad_core.initialization.init_deck_adi_fluid import \
        init_deck_adi_fluid
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import \
        _select_model_from_deck
    from PRSTCore.ad_core.operators_tpfa import setup_operators_tpfa
    from PRSTCore.deckformat.deckinput.convert_deck_units import \
        convert_deck_units
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import \
        read_eclipse_deck
    from PRSTCore.deckformat.resultinput.convert_restart_to_states import \
        convert_restart_to_states
    from PRSTCore.deckformat.resultinput.init_grid_from_eclipse_output import \
        init_grid_from_eclipse_output
    from PRSTCore.deckformat.resultinput.process_eclipse_restart_spec import \
        process_eclipse_restart_spec
    from PRSTCore.deckformat.resultinput.read_eclipse_output_file_unfmt import \
        read_eclipse_output_file_unfmt
    from PRSTCore.hm.utils.observed.getObservedFromSchedule import \
        getObservedFromSchedule
    from PRSTCore.nwm._deps import convertDeckScheduleToMRST

    prefix = os.path.join(path, name)
    deck = convert_deck_units(read_eclipse_deck(prefix + '.DATA'))

    init = read_eclipse_output_file_unfmt(prefix + '.INIT')
    grid = read_eclipse_output_file_unfmt(prefix + '.EGRID')
    G, rock, N, T = init_grid_from_eclipse_output(init, grid,
                                                  output_sim_grid=False)

    fluid = init_deck_adi_fluid(deck)
    fmodel = _select_model_from_deck(G, rock, fluid, deck)
    fmodel.operators = setup_operators_tpfa(G, rock, trans=T, neighbors=N)

    fschedule = convertDeckScheduleToMRST(fmodel, deck,
                                          ReorderStrategy='origin')
    observed = getObservedFromSchedule(fschedule)

    rsspec = process_eclipse_restart_spec(prefix, 'all')
    # ``[states, ~] = convertRestartToStates(...)``.
    fstates, _ = convert_restart_to_states(
        prefix, G, restart_info=rsspec, split_wells_on_sign_change=False,
        remove_closed_wells=False, remove_crossflow=False,
        include_well_sols=False, include_aquifers=True, steps=1)
    return fmodel, fschedule, observed, fstates[0]


def build_coarse_model(fmodel, fschedule, fstate0):
    """Coarsen by 10x10x1 and upscale the state and schedule onto it."""
    from PRSTCore.ad_core.upscale.upscale_model_tpfa import upscale_model_tpfa
    from PRSTCore.ad_core.upscale.upscale_schedule import upscale_schedule
    from PRSTCore.ad_core.upscale.upscale_state import upscale_state
    from PRSTCore.ad_props.impose_relperm_scaling import impose_relperm_scaling
    from PRSTCore.coarsegrid.partition_ui import partition_ui
    from PRSTCore.coarsegrid.process_partition import (compress_partition,
                                                       process_partition)
    from PRSTCore.hm.utils.getRelpermScalingPoints import \
        getRelpermScalingPoints

    G = fmodel.G
    cart = np.asarray(G['cartDims'], dtype=int)
    dims = [int(np.ceil(cart[0] / 10)), int(np.ceil(cart[1] / 10)),
            int(cart[2])]

    p = partition_ui(G, dims)
    p = process_partition(G, p)
    p = compress_partition(p)

    cmodel = upscale_model_tpfa(fmodel, p, trans_from_rock=False)
    cstate0 = upscale_state(cmodel, fmodel, fstate0)
    cschedule = upscale_schedule(cmodel, fschedule,
                                 well_upscale_method='sum')

    scaling = getRelpermScalingPoints(cmodel)
    # MATLAB expands the pairs: imposeRelpermScaling(cmodel, scaling{:}).
    cmodel = impose_relperm_scaling(cmodel, **dict(scaling))
    return cmodel, cschedule, cstate0, p


def phase_weights(cmodel, cstates, cschedule, observed, step):
    """One weight per phase, ``1/sqrt(that phase's own mismatch)``.

    Each phase is scored alone -- all other weights zero -- so the weight
    it gets is the reciprocal square root of its own contribution. Every
    term then starts near 1 and no phase dominates by magnitude alone.
    """
    from PRSTCore.hm.utils.evaluate.matchObservedOWG import matchObservedOWG

    names = ('WaterRateWeight', 'OilRateWeight', 'GasRateWeight', 'BHPWeight')
    w = np.zeros(4)
    for i, name in enumerate(names):
        weighting = {n: 0.0 for n in names}
        weighting[name] = 1.0
        val = matchObservedOWG(cmodel, [cstates[k] for k in step], cschedule,
                               [observed[k] for k in step],
                               ComputePartials=False, **weighting)
        total = float(np.sum([np.sum(v) for v in val]))
        w[i] = 1.0 / np.sqrt(total) if total > 0 else 0.0
    return w


def main(path=PATH, name=NAME, n_steps=N_STEPS):
    """Train the coarse model. Returns ``(v, u, history)``."""
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
        simulate_schedule_ad
    from PRSTCore.hm.utils.evaluate.matchObservedOWG import matchObservedOWG
    from PRSTCore.optimization import evaluate_match, unit_box_bfgs
    from PRSTCore.optimization.utils.parameters import (
        add_parameter, get_scaled_parameter_vector,
        update_setup_from_scaled_parameters)

    fmodel, fschedule, observed, fstate0 = build_fine_model(path, name)
    cmodel, cschedule, cstate0, _ = build_coarse_model(fmodel, fschedule,
                                                       fstate0)

    cwellSols, cstates = simulate_schedule_ad(cstate0, cmodel, cschedule)[:2]

    nstep = len(cschedule['step']['val']) if n_steps is None else n_steps
    step = list(range(nstep))
    cschedule['step']['control'] = cschedule['step']['control'][:nstep]
    cschedule['step']['val'] = cschedule['step']['val'][:nstep]

    setup = {'model': cmodel, 'schedule': cschedule, 'state0': cstate0}
    params = []
    params = add_parameter(params, setup, name='porevolume',
                           relative_limits=[0.1, 10], scaling='linear',
                           uniform_limits=False)
    params = add_parameter(params, setup, name='transmissibility',
                           relative_limits=[0.01, 100], scaling='log',
                           uniform_limits=False)
    params = add_parameter(params, setup, name='conntrans',
                           relative_limits=[0.01, 100], scaling='log',
                           uniform_limits=False)
    for pname, lims in (('swl', [0.0, 0.5]), ('swcr', [0.0, 0.5]),
                        ('sowcr', [0.0, 0.5]), ('krw', [0.5, 1.5]),
                        ('kro', [0.5, 1.5])):
        params = add_parameter(params, setup, name=pname, box_lims=lims,
                               scaling='linear')
    u0 = get_scaled_parameter_vector(setup, params)

    w = phase_weights(cmodel, cstates, cschedule, observed, step)

    def obj(model_, states_, schedule_, observed_, tt, tstep, state):
        # Gas and bhp are switched off, as the MATLAB does.
        return matchObservedOWG(
            model_, states_, schedule_, observed_,
            WaterRateWeight=w[0], OilRateWeight=w[1], GasRateWeight=0.0,
            BHPWeight=0.0, ComputePartials=tt, tStep=tstep, state=state,
            from_states=False, mismatchSum=True)

    def f(p):
        return evaluate_match(p, obj, setup, params,
                              [observed[k] for k in step])

    v, u, history = unit_box_bfgs(u0, f, max_it=50, obj_change_tol=1e-6,
                                  grad_tol=1e-6, save_history='AB_i')

    trained = update_setup_from_scaled_parameters(setup, params, u)
    simulate_schedule_ad(trained['state0'], trained['model'],
                         trained['schedule'])
    return v, u, history


if __name__ == '__main__':
    main()
