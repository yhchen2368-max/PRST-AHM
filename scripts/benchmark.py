"""Run a deck and report where the time went, for one or several solvers.

The point is the comparison.  A single total says a run took ninety seconds;
it does not say whether ninety seconds is the linear solver, in which case
the fix is the preconditioner, or the residual assembly, in which case the
fix is the automatic-differentiation representation.  Running the same deck
under two solver configurations and printing both splits side by side is
what makes the next optimisation choose itself.

Examples::

    python scripts/benchmark.py --case spe9 --steps 5
    python scripts/benchmark.py --case spe9 --steps 5 --solver backslash --solver amgcl-cpr
    python scripts/benchmark.py --deck examples/Norne/NORNE_ATW2013.DATA --steps 2 --json out.json

``--solver`` may be repeated; each configuration runs from the same initial
state, and the table at the end compares them.  ``auto`` is whatever
``selectLinearSolverAD`` picks for the model, which is what a normal run
gets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import PRSTCore  # noqa: F401  -- puts conda's Library/bin on PATH before MKL loads
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.ad_core.utils.profiling import profile

#: Short names for the decks that get benchmarked repeatedly.
CASES = {
    'spe1': 'examples/SpE1/SPE1CASE1.DATA',
    'spe9': 'examples/SPE9/SPE9_CP.DATA',
    'egg': 'examples/EGG/Egg_Model_ECL.DATA',
    'norne': 'examples/Norne/NORNE_ATW2013.DATA',
}


def build_solver(name, model, verbose=False):
    """A linear solver by short name, or the model's own automatic choice."""
    from PRSTCore.ad_core.solvers.select_linear_solver_ad import select_linear_solver_ad

    if name == 'auto':
        return select_linear_solver_ad(model, verbose=verbose)
    if name == 'backslash':
        from PRSTCore.ad_core.solvers.backslash_solver_ad import BackslashSolverAD
        return BackslashSolverAD(verbose=verbose)
    if name == 'cpr':
        # CPR with a direct pressure solve: what the selector falls back to
        # when no AMG extension is importable.  Worth measuring explicitly,
        # because that fallback is silent.
        from PRSTCore.ad_core.solvers.cpr_solver_ad import CPRSolverAD
        from PRSTCore.ad_core.solvers.backslash_solver_ad import BackslashSolverAD
        return CPRSolverAD(ellipticSolver=BackslashSolverAD(verbose=verbose), verbose=verbose)
    if name == 'amgcl':
        from PRSTCore.ad_core.solvers.amgcl_solver_ad import AMGCLSolverAD
        return AMGCLSolverAD(tolerance=1e-4, maxIterations=50, verbose=verbose)
    if name == 'amgcl-cpr':
        from PRSTCore.ad_core.solvers.amgcl_cpr_solver_ad import AMGCL_CPRSolverAD
        return AMGCL_CPRSolverAD(tolerance=1e-4, maxIterations=50, verbose=verbose)
    if name == 'amgcl-block-cpr':
        from PRSTCore.ad_core.solvers.amgcl_cpr_solver_block_ad import AMGCL_CPRSolverBlockAD
        return AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=verbose)
    if name.startswith('petsc'):
        from PRSTCore.ad_core.solvers.petsc_solver_ad import PETScSolverAD
        # petsc-<strategy>-<pressure preconditioner>, e.g. petsc-cpr-gamg
        parts = name.split('-')
        strategy = parts[1] if len(parts) > 1 else 'cpr'
        precond = parts[2] if len(parts) > 2 else 'gamg'
        return PETScSolverAD(strategy=strategy, pressure_precond=precond,
                             tolerance=1e-6, maxIterations=200, verbose=verbose)
    raise SystemExit('unknown solver %r' % name)


SOLVER_NAMES = ('auto', 'backslash', 'cpr', 'amgcl', 'amgcl-cpr', 'amgcl-block-cpr',
                'petsc-cpr-gamg', 'petsc-cpr-hypre', 'petsc-fieldsplit-gamg',
                'petsc-fieldsplit-hypre')


def truncate(schedule, nsteps):
    """The first ``nsteps`` report steps, leaving the controls they use."""
    if nsteps is None or nsteps <= 0:
        return schedule
    out = dict(schedule)
    step = dict(schedule['step'])
    step['val'] = np.asarray(schedule['step']['val']).ravel()[:nsteps]
    step['control'] = np.asarray(schedule['step']['control']).ravel()[:nsteps]
    out['step'] = step
    return out


def summarise_wells(well_sols):
    """A scalar fingerprint of the answer, to catch a solver changing it.

    Not a parity check -- it is a number that moves when the well solution
    moves, so that a run which got faster by quietly solving a different
    problem does not read as a win.
    """
    if not well_sols:
        return {}
    last = well_sols[-1]
    out = {}
    for key in ('qWs', 'qOs', 'qGs', 'bhp'):
        values = [float(w.get(key, 0.0) or 0.0) for w in last if isinstance(w, dict)]
        if values:
            out[key] = float(np.sum(np.abs(values)))
    return out


def run_one(deck, solver_name, nsteps, verbose=False):
    """One full measured run: fresh model, fresh state, one solver."""
    print('  [%s] initialising...' % solver_name, flush=True)
    started = time.perf_counter()
    state0, model, schedule, nonlinear = init_eclipse_problem_ad(str(deck))
    init_seconds = time.perf_counter() - started

    linear = build_solver(solver_name, model, verbose=verbose)
    nonlinear.LinearSolver = linear
    schedule = truncate(schedule, nsteps)
    nsteps_actual = len(np.asarray(schedule['step']['val']).ravel())

    print('  [%s] %s, %d report steps...' % (
        solver_name, type(linear).__name__, nsteps_actual), flush=True)
    failure = None
    well_sols, states = [], []
    with profile(model, nonlinear, label=solver_name) as prof:
        try:
            well_sols, states = simulate_schedule_ad(
                state0, model, schedule, NonLinearSolver=nonlinear)
        except Exception as exc:  # a solver that cannot finish is a result
            failure = '%s: %s' % (type(exc).__name__, exc)

    record = prof.as_dict()
    record.update({
        'solver': solver_name,
        'solver_class': type(linear).__name__,
        'init_seconds': init_seconds,
        'report_steps_requested': nsteps_actual,
        'report_steps_completed': len(states),
        'failure': failure,
        'wells': summarise_wells(well_sols),
    })
    print(prof.summary(), flush=True)
    if failure:
        print('  FAILED after %d/%d steps: %s' % (len(states), nsteps_actual, failure), flush=True)
    print(flush=True)
    return record


def print_table(records):
    """One row per configuration, fastest first."""
    header = ('%-18s %-26s %9s %9s %9s %9s %7s %6s' %
              ('solver', 'class', 'total', 'assembly', 'linear', 'other', 'lin_it', 'steps'))
    print(header)
    print('-' * len(header))
    for record in sorted(records, key=lambda r: r['total_seconds']):
        print('%-18s %-26s %8.2fs %8.2fs %8.2fs %8.2fs %7d %6d%s' % (
            record['solver'], record['solver_class'],
            record['total_seconds'], record['assembly_seconds'],
            record['linear_seconds'], record['other_seconds'],
            record['linear_iterations'], record['report_steps_completed'],
            '  FAILED' if record['failure'] else ''))

    finished = [r for r in records if not r['failure']]
    if len(finished) > 1:
        slowest = max(finished, key=lambda r: r['total_seconds'])
        fastest = min(finished, key=lambda r: r['total_seconds'])
        if fastest is not slowest and fastest['total_seconds'] > 0:
            print('\n%s is %.1fx faster than %s overall' % (
                fastest['solver'], slowest['total_seconds'] / fastest['total_seconds'],
                slowest['solver']))
        # Whether the answer moved. Two solvers that agree here differ only
        # in how they got there; two that do not are not comparable at all.
        keys = sorted({k for r in finished for k in r['wells']})
        if keys:
            print('\nwell-solution fingerprint (sum |value| over wells, last step):')
            print('  %-18s %s' % ('solver', '  '.join('%14s' % k for k in keys)))
            for record in finished:
                print('  %-18s %s' % (record['solver'], '  '.join(
                    '%14.6g' % record['wells'].get(k, float('nan')) for k in keys)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--case', choices=sorted(CASES), help='a named example deck')
    source.add_argument('--deck', help='path to a .DATA deck')
    parser.add_argument('--steps', type=int, default=3,
                        help='report steps to run (0 = the whole schedule)')
    parser.add_argument('--solver', action='append', choices=SOLVER_NAMES,
                        help='linear solver to measure; repeat to compare')
    parser.add_argument('--json', help='write the records to this file')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    deck = REPO_ROOT / CASES[args.case] if args.case else Path(args.deck)
    if not deck.is_file():
        raise SystemExit('deck not found: %s' % deck)

    from PRSTCore.ad_core.solvers.select_linear_solver_ad import (
        check_amgcl, check_amgcl_block_cpr)
    print('deck            : %s' % deck)
    print('python          : %d.%d at %s' % (
        sys.version_info[0], sys.version_info[1], sys.prefix))
    print('amgcl scalar    : %s' % check_amgcl())
    print('amgcl block cpr : %s' % check_amgcl_block_cpr())
    print()

    records = [run_one(deck, name, args.steps, verbose=args.verbose)
               for name in (args.solver or ['auto'])]
    print_table(records)

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2), encoding='utf-8')
        print('\nwrote %s' % args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
