"""Simulating an Eclipse/DATA input file -- the PRSTCore version of
JutulDarcy's ``examples/introduction/data_input_file.jl``.

Same narrative as the Julia example, on SPE9: read the deck, set up the case,
simulate it, then look at the model in 3D, at the well responses, at the
field measurables, and at the summary curves.

Two differences from the Julia original, both forced by the environment
rather than chosen:

*The deck is the corner-point one.*  ``SPE9.DATA`` is block-centred with a
non-constant ``TOPS``, which PRSTCore's grid reader does not support;
``SPE9_CP.DATA`` is the same case as a corner-point grid.

*It runs in two stages.*  The solver needs ``petsc4py`` and the 3D view needs
``vtk``, and no interpreter here has both -- there is no VTK wheel for
CPython 3.14, where PETSc lives.  So the simulation, the measurables and all
the curves happen in one interpreter, and the 3D view reads what that wrote::

    .conda/python.exe scripts/spe9_data_input.py --steps 20
    python           scripts/spe9_data_input.py --view

``--steps`` is there because the whole 90-step schedule takes about five
minutes; leave it off for the full run.
"""

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import PRSTCore  # noqa: F401  -- puts conda's Library/bin on PATH before MKL

DECK = os.path.join(ROOT, 'examples', 'SPE9', 'SPE9_CP.DATA')

#: The panel groups JutulDarcy's ``plot_summary`` example asks for: what is
#: being produced now, what has been produced in total, where the average
#: pressure has got to, and how much oil is left.
SUMMARY_PLOTS = ('FOPR,FWPR,FGPR', 'FOPT,FWPT,FGPT', 'FPR', 'FOIP')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--deck', default=DECK,
                        help='Eclipse DATA deck (default: SPE9_CP)')
    parser.add_argument('--steps', type=int, default=None,
                        help='report steps to run (default: the whole schedule)')
    parser.add_argument('--outdir', default=None,
                        help='where results and figures go '
                             '(default: results/<deck stem>_data_input)')
    parser.add_argument('--view', action='store_true',
                        help='skip the simulation and open the 3D viewer on '
                             'the states an earlier run saved')
    parser.add_argument('--show', action='store_true',
                        help='show the figures interactively as well as '
                             'writing them to PNG')
    return parser.parse_args()


def truncate_schedule(schedule, nsteps):
    """The first ``nsteps`` report steps, controls untouched."""
    if nsteps is None or nsteps >= len(schedule['step']['val']):
        return schedule
    return {'step': {'val': schedule['step']['val'][:nsteps],
                     'control': schedule['step']['control'][:nsteps]},
            'control': schedule['control']}


def describe_deck(deck_path):
    """## Show the input data

    The deck is a dict of sections, so RUNSPEC can be inspected the same way
    ``case.input_data["RUNSPEC"]`` is in the Julia example.
    """
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck

    deck = read_eclipse_deck(deck_path)
    print('\n=== input data ===')
    print('sections:', ', '.join(sorted(k for k in deck if isinstance(deck[k], dict))))

    runspec = deck.get('RUNSPEC', {})
    print('\nRUNSPEC:')
    for key in sorted(runspec):
        value = runspec[key]
        if isinstance(value, np.ndarray):
            text = 'array%s' % (value.shape,)
        else:
            text = str(value)
        print('  %-12s %s' % (key, text[:70]))
    return deck


def write_measurables_csv(path, measurables):
    """Field vectors as CSV.  Written by hand rather than through pandas,
    which the solver environment does not have."""
    time_days = measurables['time_days']
    names = sorted(k for k, v in measurables.items()
                   if isinstance(v, np.ndarray) and v.ndim == 1
                   and len(v) == len(time_days) and k != 'time_days')
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_days'] + names)
        for row in range(len(time_days)):
            writer.writerow(['%.6g' % time_days[row]]
                            + ['%.6g' % measurables[name][row] for name in names])
    return path


def open_viewer(states_path):
    """## Plot the simulation model

    JutulDarcy's ``plot_reservoir`` / ``plot_explorer``: the grid, coloured by
    a field, with the wells drawn and a timestep slider.
    """
    from PRSTCore.visualization.results_io import load_states
    from PRSTCore.visualization import view_reservoir

    saved = load_states(states_path)
    print('opening the 3D viewer on %s' % states_path)
    print('  cells %d, wells %d, fields %s'
          % (saved['G']['cells']['num'], len(saved['W']),
             ', '.join(sorted(saved['fields']))))
    view_reservoir(saved['G'], W=saved['W'], fields=saved['fields'],
                   title='SPE9 -- PRSTCore')


def main():
    args = parse_args()
    deck_path = args.deck if os.path.isabs(args.deck) else os.path.join(ROOT, args.deck)
    stem = os.path.splitext(os.path.basename(deck_path))[0]
    outdir = args.outdir or os.path.join(ROOT, 'results', stem + '_data_input')
    os.makedirs(outdir, exist_ok=True)
    states_path = os.path.join(outdir, 'states.npz')

    if args.view:
        if not os.path.exists(states_path):
            raise SystemExit('no saved states at %s -- run the simulation '
                             'stage first' % states_path)
        open_viewer(states_path)
        return

    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
    from PRSTCore.ad_core.measurables import well_measurables, field_measurables
    from PRSTCore.ad_core.plotting import (plot_summary, plot_measurables,
                                           plot_well_responses)
    from PRSTCore.visualization.results_io import save_states

    describe_deck(deck_path)

    # ## Set up and run a simulation
    started = time.time()
    state0, model, schedule, solver = init_eclipse_problem_ad(deck_path)
    print('\n=== case ===')
    print('  deck + init   : %.1f s' % (time.time() - started))
    print('  cells         : %d' % model.G['cells']['num'])
    print('  phases        : oil=%s water=%s gas=%s'
          % (model.oil, model.water, model.gas))
    print('  linear solver : %s' % type(getattr(solver, 'LinearSolver', None)).__name__)
    print('  report steps  : %d' % len(schedule['step']['val']))

    schedule = truncate_schedule(schedule, args.steps)
    nsteps = len(schedule['step']['val'])
    dt = np.asarray(schedule['step']['val'], dtype=float)

    print('\n=== simulating %d report steps ===' % nsteps, flush=True)
    started = time.time()
    well_sols, states = simulate_schedule_ad(state0, model, schedule, solver,
                                             verbose=False)
    wall = time.time() - started
    print('  %d steps in %.1f s (%.2f s/step)' % (nsteps, wall, wall / max(nsteps, 1)))

    # ## Measurables -- the layer JutulDarcy has and PRSTCore did not
    wells = well_measurables(well_sols, dt, units='metric')
    field = field_measurables(well_sols, dt, states=states, G=model.G,
                              rock=model.rock, model=model, units='metric')

    print('\n=== field measurables (metric) ===')
    print('  wells         : %d  (%s%s)'
          % (len(wells['names']), ', '.join(wells['names'][:4]),
             ', ...' if len(wells['names']) > 4 else ''))
    for name in ('FOPR', 'FWPR', 'FGPR', 'FPR', 'FOIP', 'FWIT'):
        if name in field:
            values = field[name]
            print('  %-5s %12.4g -> %12.4g   [%s]'
                  % (name, values[0], values[-1],
                     field['unit_labels'][
                         'pressure' if name == 'FPR'
                         else 'volume' if name.endswith(('T', 'IP'))
                         else 'rate']))

    csv_path = write_measurables_csv(os.path.join(outdir, 'field_measurables.csv'),
                                     field)

    # States for the 3D stage.  model.G, not a grid rebuilt from the deck --
    # the run's own active set is the one the states are indexed by.
    # Wells across every control, not just the first: a schedule opens wells
    # as it goes, and the first control alone would draw a fraction of them.
    # (SPE9 opens all 26 at once, but the deck that does not is the one this
    # would be wrong on.)
    drawn_wells = {}
    for control in schedule['control']:
        for well in (control or {}).get('W') or []:
            drawn_wells.setdefault(str(well.get('name', '')), well)

    save_states(states_path, states, model.G, W=list(drawn_wells.values()),
                times=np.cumsum(dt) / 86400.0, model=model,
                meta={'deck': deck_path, 'steps': nsteps})

    # ## Plot the well responses, the field responses and the summary
    figures = {
        'well_responses.png': plot_well_responses(
            wells, title='SPE9 well responses'),
        'field_measurables.png': plot_measurables(
            field, left='FGPR', right='FPR',
            title='SPE9 field gas production against average pressure'),
        'summary.png': plot_summary(
            field, plots=SUMMARY_PLOTS, cols=2, title='SPE9 summary'),
    }
    print('\n=== outputs ===')
    for name, figure in figures.items():
        path = os.path.join(outdir, name)
        figure.savefig(path, dpi=130, bbox_inches='tight')
        print('  %s' % path)
    print('  %s' % csv_path)
    print('  %s' % states_path)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()

    print('\n  for the 3D view, in the interpreter that has vtk:')
    print('    python scripts/spe9_data_input.py --view'
          + ('' if args.outdir is None else ' --outdir %s' % args.outdir))


if __name__ == '__main__':
    main()
