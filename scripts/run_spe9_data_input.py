"""Simulate an Eclipse DATA file and look at the results -- SPE9.

The PRSTCore counterpart of JutulDarcy's
``examples/introduction/data_input_file.jl``, section for section:

===========================================  ==========================================
JutulDarcy                                   PRSTCore
===========================================  ==========================================
``setup_case_from_data_file(pth)``           ``init_eclipse_problem_ad(deck)``
``simulate_reservoir(case)`` -> ``ws, states``  ``simulate_schedule_ad(...)`` -> same
``case.input_data["RUNSPEC"]``               ``read_eclipse_deck(deck)['RUNSPEC']``
``plot_reservoir(model, states)``            ``view_reservoir(G, W=, states=)``
``plot_well_results(ws)``                    ``plot_well_sols`` / the curves here
``plot_reservoir_measurables(..., :fgpr, :pres)``  ``plot_measurables(m, 'FGPR', 'FPR')``
``plot_summary(summary, plots=[...])``       ``plot_summary(m, plots=[...])``
===========================================  ==========================================

Everything is reported in METRIC: rates in sm3/day, volumes in sm3,
pressures in bar.

**Two interpreters.**  Simulating needs PETSc, which lives in ``.conda``
(3.14); the 3D view needs VTK, which has no 3.14 wheel and so lives in
anaconda3 (3.13).  One process cannot do both, so this script simulates,
writes ``states.npz``, and prints the command that opens it in the other
interpreter.  Everything else -- the curves, the measurables -- is matplotlib
and runs in either.

Usage::

    .conda/python.exe scripts/run_spe9_data_input.py                 # whole deck
    .conda/python.exe scripts/run_spe9_data_input.py --steps 10      # a taste
    .conda/python.exe scripts/run_spe9_data_input.py --save-figs     # PNGs, no window
    python scripts/run_spe9_data_input.py --view-only                # 3D, anaconda3
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import PRSTCore  # noqa: F401  -- puts conda's Library/bin on PATH before MKL

#: The corner-point SPE9, not the block-centred ``SPE9.DATA`` next to it.
#: That one describes its grid with DX/DY/DZ and a varying TOPS, which
#: ``init_eclipse_grid`` rejects ("block-centred grids are only supported for
#: constant TOPS"); ``SPE9_CP.DATA`` is the same case as corner point and is
#: what the rest of PRSTCore's SPE9 work uses.
DECK = os.path.join(ROOT, 'examples', 'SPE9', 'SPE9_CP.DATA')
RESULTS = os.path.join(ROOT, 'results', 'spe9_data_input')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--deck', default=DECK, help='Eclipse DATA deck')
    parser.add_argument('--steps', type=int, default=None,
                        help='report steps to run (default: the whole deck)')
    parser.add_argument('--out', default=RESULTS, help='output directory')
    parser.add_argument('--pressure-precond', default='hypre',
                        choices=('hypre', 'gamg', 'ilu', 'lu'),
                        help="PETSc CPR pressure-block preconditioner")
    parser.add_argument('--second-stage', default='ilu',
                        help="PETSc CPR full-system smoother")
    parser.add_argument('--save-figs', action='store_true',
                        help='write the figures as PNG instead of showing them')
    parser.add_argument('--view-only', action='store_true',
                        help='skip the simulation; open the 3D viewer on the '
                             'states.npz left by an earlier run')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def show_input_data(deck_path):
    """`case.input_data` / `case.input_data["RUNSPEC"]`."""
    from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck

    deck = read_eclipse_deck(deck_path)
    print('\n=== input data ===', flush=True)
    print('  sections : %s' % ', '.join(sorted(k for k in deck if k.isupper())))
    runspec = deck.get('RUNSPEC', {})
    if isinstance(runspec, dict):
        keys = sorted(runspec)
        print('  RUNSPEC  : %d keywords' % len(keys))
        print('             %s' % ', '.join(keys[:16]))
        for name in ('DIMENS', 'OIL', 'WATER', 'GAS', 'DISGAS', 'METRIC', 'FIELD'):
            if name in runspec:
                print('             %-8s %s' % (name, runspec[name]))
    return deck


def build_solver(model, args):
    """Force the PETSc CPR solver rather than take whatever is auto-selected.

    ``select_linear_solver_ad`` picks by problem size, and SPE9 (9000 cells,
    three phases) sits near enough the threshold that the choice is not
    obvious from the outside.  Asking for PETSc explicitly makes the run
    reproducible and is what was wanted here.
    """
    from PRSTCore.ad_core.solvers.petsc_solver_ad import PETScSolverAD

    return PETScSolverAD(strategy='cpr',
                         pressure_precond=args.pressure_precond,
                         second_stage=args.second_stage,
                         tolerance=1e-4)


def simulate(args):
    """`setup_case_from_data_file` + `simulate_reservoir`."""
    from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad)
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

    started = time.time()
    state0, model, schedule, solver = init_eclipse_problem_ad(args.deck)
    print('\n=== case ===', flush=True)
    print('  deck + init   : %.1f s' % (time.time() - started))
    print('  active cells  : %d   phases oil=%s water=%s gas=%s'
          % (model.G['cells']['num'], model.oil, model.water, model.gas))

    solver.LinearSolver = build_solver(model, args)
    solver.verbose = not args.quiet
    solver.errorOnFailure = False
    print('  linear solver : %s(cpr, %s, second stage=%s)'
          % (type(solver.LinearSolver).__name__, args.pressure_precond,
             args.second_stage))

    nsteps = len(schedule['step']['val'])
    if args.steps:
        # Trim rather than stop early, so the schedule the solver validates
        # is the one it runs -- a step's control index refers into the
        # control table and must stay consistent with it.
        keep = min(args.steps, nsteps)
        schedule = dict(schedule)
        schedule['step'] = {'val': schedule['step']['val'][:keep],
                            'control': schedule['step']['control'][:keep]}
        nsteps = keep
    print('  report steps  : %d' % nsteps, flush=True)

    started = time.time()
    well_sols, states = simulate_schedule_ad(state0, model, schedule,
                                             nonlinear_solver=solver,
                                             verbose=not args.quiet)
    print('  simulated     : %.1f s (%.1f min)'
          % (time.time() - started, (time.time() - started) / 60.0), flush=True)

    return state0, model, schedule, well_sols, states


def save_states_file(args, model, schedule, states, well_sols):
    """Leave the per-cell results where the 3D viewer can pick them up."""
    from PRSTCore.visualization.results_io import save_states

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'states.npz')

    wells = {}
    for control in schedule['control']:
        for well in control.get('W') or []:
            wells.setdefault(str(well.get('name', '')), well)

    dt = np.asarray(schedule['step']['val'], dtype=float)
    times = np.cumsum(dt) / 86400.0
    save_states(path, states, model.G, W=list(wells.values()),
                times=times, model=model,
                meta={'deck': args.deck, 'steps': len(states)})
    return path


def figures(measurables, per_well, args):
    """The three plots the Julia example draws, minus the 3D one."""
    import matplotlib
    if args.save_figs:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from PRSTCore.ad_core.plotting import (plot_summary, plot_measurables,
                                           plot_well_responses)

    made = []

    # `plot_well_results(ws)`
    made.append(('well_results',
                 plot_well_responses(per_well,
                                     title='SPE9 well responses (METRIC)')))

    # `plot_reservoir_measurables(case, ws, states, left=:fgpr, right=:pres)`
    if 'FPR' in measurables:
        made.append(('field_response',
                     plot_measurables(measurables, left='FGPR', right='FPR',
                                      title='field gas rate against average pressure')))

    # `plot_summary(summary, plots=[...], unit_system="Field", cols=2)`
    made.append(('summary',
                 plot_summary(measurables,
                              plots=['FOPR,FWPR,FGPR', 'FOPT,FWPT,FGPT',
                                     'FPR', 'FOIP'],
                              cols=2, title='SPE9 field summary (METRIC)')))

    if args.save_figs:
        os.makedirs(args.out, exist_ok=True)
        for name, fig in made:
            path = os.path.join(args.out, name + '.png')
            fig.savefig(path, dpi=120)
            print('  figure -> %s' % path, flush=True)
    else:
        plt.show()


def open_viewer(path):
    """`plot_reservoir` / `plot_explorer` -- needs the VTK interpreter."""
    # The guard has to cover the *call*, not just the import: view_saved
    # imports the Qt viewer lazily, so on an interpreter without VTK the
    # ModuleNotFoundError surfaces from inside it rather than from here.
    try:
        from PRSTCore.visualization.results_io import view_saved

        view_saved(path, title='SPE9')
        return True
    except ImportError as err:
        print('\n  3D view needs VTK and Qt, which this interpreter lacks '
              '(%s).' % err)
        print('  Open it with the anaconda3 interpreter:')
        print('    python scripts/run_spe9_data_input.py --view-only',
              flush=True)
        return False


def main():
    args = parse_args()
    path = os.path.join(args.out, 'states.npz')

    if args.view_only:
        if not os.path.exists(path):
            raise SystemExit('no states at %s -- run without --view-only first'
                             % path)
        open_viewer(path)
        return

    show_input_data(args.deck)
    state0, model, schedule, well_sols, states = simulate(args)

    from PRSTCore.ad_core.measurables import field_measurables, well_measurables

    dt = np.asarray(schedule['step']['val'], dtype=float)
    per_well = well_measurables(well_sols, dt, units='metric')
    measurables = field_measurables(well_sols, dt, states=states, G=model.G,
                                    rock=model.rock, model=model, units='metric')

    print('\n=== field measurables (METRIC) ===', flush=True)
    print('  %-6s %s' % ('step', 'time_days'))
    for name in ('FOPR', 'FWPR', 'FGPR', 'FOPT', 'FPR', 'FOIP'):
        if name in measurables:
            values = measurables[name]
            print('  %-6s first %12.4g   last %12.4g' % (name, values[0], values[-1]))
    print('  wells: %d  (%s%s)'
          % (len(per_well['names']), ', '.join(per_well['names'][:8]),
             ', ...' if len(per_well['names']) > 8 else ''), flush=True)

    path = save_states_file(args, model, schedule, states, well_sols)
    print('  states -> %s (%.1f MB)'
          % (path, os.path.getsize(path) / 1024.0 / 1024.0), flush=True)

    figures(measurables, per_well, args)
    open_viewer(path)


if __name__ == '__main__':
    main()
