"""Run the full T142 model with a per-step breakdown of where the time goes.

Eclipse does this deck in about fifteen minutes.  The point of the trace
below is to say which part of PRSTCore's time would have to change for that
to be within reach, rather than to report one number that cannot be acted
on.

Three levels of detail, all measured on the same run:

* the report step -- wall clock, Newton iterations, convergence;
* inside it -- residual assembly against linear solve against everything
  else, taken by wrapping two calls, so the cost is one function call per
  Newton iteration rather than per cell;
* inside the linear solve -- PETSc's own event log, which separates the
  preconditioner *setup* from its *application*.  That distinction is the
  one that decides what to do next: a solve dominated by setup is fixed by
  reusing the preconditioner across Newton iterations, while one dominated
  by application needs a better preconditioner or fewer iterations.

Usage::

    python scripts/run_t142_full.py [steps] [--backend diagonal|sparse]
                                            [--no-face-operators]
                                            [--quiet]

Outputs ``results/T142_full/well_rates.csv`` as before, and
``results/T142_full/timing.csv`` with one row per report step.
"""
import argparse
import csv
import os
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import PRSTCore  # noqa: F401  -- puts conda's Library/bin on PATH before MKL
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECK = os.path.join(ROOT, 'examples', 'T142', 'T142_E100.DATA')
START = date(1999, 9, 1)

#: PETSc events worth separating.  Anything the build does not record shows
#: up with a zero count and is dropped from the report rather than printed
#: as a confident zero.
#: PETSc events worth separating, as (event, column, label).  PCSetUp and
#: PCApply nest -- a composite preconditioner's own event brackets its
#: children's -- so their totals include the stages beneath them and cannot
#: simply be subtracted from one another.  The leaf events can: MatSolve is
#: the ILU triangular solves and nothing else, MatLUFactorNum the numeric
#: factorisation and nothing else.  Those are what attribute the second
#: stage's share of an apply, which is the question ILU-versus-something-
#: cheaper turns on, so they are recorded rather than printed and dropped.
PETSC_EVENTS = (
    ('PCSetUp', 'pc_setup_s', 'preconditioner setup'),
    ('PCApply', 'pc_apply_s', 'preconditioner apply'),
    ('MatLUFactorNum', 'ilu_factor_s', 'ILU numeric factorisation'),
    ('MatSolve', 'ilu_solve_s', 'ILU triangular solves'),
    ('MatMult', 'matvec_s', 'matrix-vector products'),
    ('KSPSolve', 'ksp_total_s', 'KSP total'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('steps', nargs='?', type=int, default=None,
                        help='report steps to run (default: the whole deck)')
    parser.add_argument('--deck', default=DECK,
                        help='path to the Eclipse DATA deck (default: %s)' % DECK)
    parser.add_argument('--start-date', default=START.isoformat(),
                        help='simulation start date, YYYY-MM-DD (default: '
                             '%s)' % START.isoformat())
    parser.add_argument('--backend', choices=('diagonal', 'sparse'), default='diagonal',
                        help='automatic-differentiation representation for the assembly')
    parser.add_argument('--no-face-operators', action='store_true',
                        help='assemble the flux the general way, for comparison')
    parser.add_argument('--second-stage', default=None,
                        metavar='PC',
                        help="CPR's full-system smoother: 'ilu' (the "
                             "default), or a cheaper one to measure against "
                             "it -- 'bjacobi', 'sor', 'jacobi'")
    parser.add_argument('--pressure-precond', default=None,
                        metavar='PC',
                        help="pressure-block preconditioner: 'gamg', "
                             "'hypre', 'ilu' or 'lu' (default: the "
                             "auto-selected one, hypre above 60k unknowns)")
    parser.add_argument('--use-linesearch', action='store_true',
                        help='enable MRST NonLinearSolver line search '
                             '(residual bisection) once oscillation/'
                             'stagnation is detected')
    parser.add_argument('--enforce-residual-decrease', action='store_true',
                        help="abandon a Newton mini-step whose residual does "
                             "not drop (MRST solveMinistep "
                             "enforceResidualDecrease) so the outer loop can "
                             "cut dt instead of burning iterations")
    parser.add_argument('--acceptance-factor', type=float, default=None,
                        metavar='F',
                        help='when the Newton budget is exhausted, accept '
                             'the mini-step if the residual is below F times '
                             'the tolerance (MRST acceptanceFactor; default '
                             '1 disables the relaxed acceptance)')
    parser.add_argument('--minporo', type=float, default=None, metavar='PHI',
                        help="floor the deck's porosity at PHI (MRST "
                             "setupSPE10_AD 'minporo'; its SPE10 value is "
                             "0.01, spe10.m uses 0.001). Both SPE10 decks "
                             'fill dead cells with PORO=1e-7, which gives '
                             'the trueIMPES CPR weights a 5e6 spread and '
                             'stalls the pressure preconditioner. It moves '
                             'pore volume, so it changes the answer.')
    parser.add_argument('--linear-solver', choices=('auto', 'amgcl-cpr'),
                        default='auto',
                        help="'auto' keeps whatever selectLinearSolverAD "
                             "picked (PETSc where available); 'amgcl-cpr' "
                             'swaps in AMGCL_CPRSolverBlockAD, which is what '
                             'the GUI builds for its "AMGCL CPR" method')
    parser.add_argument('--amgcl-strategy', default='mrst',
                        choices=('mrst', 'mrst_drs', 'amgcl', 'amgcl_drs'),
                        help='CPR strategy for --linear-solver amgcl-cpr')
    parser.add_argument('--amgcl-decoupling', default='trueIMPES',
                        choices=('trueIMPES', 'quasiIMPES', 'none'),
                        help='decoupling for --linear-solver amgcl-cpr')
    parser.add_argument('--quiet', action='store_true',
                        help='suppress the per-mini-step Newton trace')
    parser.add_argument('--save-states', nargs='?', const=True, default=None,
                        metavar='PATH',
                        help='write per-cell pressure and saturation for '
                             'every report step, together with the grid the '
                             'run used, to PATH (default: states.npz beside '
                             'the other outputs). This is what the 3D viewer '
                             'reads; without it only well rates survive the '
                             'run.')
    return parser.parse_args()


class PhaseTimer:
    """Assembly and linear-solve time, taken without touching either."""

    def __init__(self, model, solver):
        self.assembly = 0.0
        self.assembly_calls = 0
        self.linear = 0.0
        self.linear_calls = 0
        self.linear_iterations = 0
        self._restore = []

        original_equations = model.get_equations

        def timed_equations(*a, **k):
            started = time.perf_counter()
            try:
                return original_equations(*a, **k)
            finally:
                self.assembly += time.perf_counter() - started
                self.assembly_calls += 1

        model.get_equations = timed_equations
        self._restore.append((model, 'get_equations', original_equations))

        linear = getattr(solver, 'LinearSolver', None)
        if linear is not None and hasattr(linear, 'solveLinearProblem'):
            original_solve = linear.solveLinearProblem

            def timed_solve(*a, **k):
                started = time.perf_counter()
                try:
                    result = original_solve(*a, **k)
                finally:
                    self.linear += time.perf_counter() - started
                    self.linear_calls += 1
                if isinstance(result, tuple) and len(result) == 3 and isinstance(result[2], dict):
                    self.linear_iterations += int(result[2].get('Iterations', 0) or 0)
                return result

            linear.solveLinearProblem = timed_solve
            self._restore.append((linear, 'solveLinearProblem', original_solve))

    def take(self):
        """The totals since the last call, and reset."""
        out = (self.assembly, self.assembly_calls, self.linear,
               self.linear_calls, self.linear_iterations)
        self.assembly = self.linear = 0.0
        self.assembly_calls = self.linear_calls = self.linear_iterations = 0
        return out


class PetscEvents:
    """PETSc's own event log, read as deltas per report step.

    This is what separates preconditioner setup from preconditioner
    application.  Wrapping ``solveLinearProblem`` cannot see inside a single
    PETSc call, and guessing at the split from iteration counts is how the
    wrong half gets optimised.
    """

    def __init__(self):
        self.available = False
        self._events = {}
        self._previous = {}
        try:
            from petsc4py import PETSc
            PETSc.Log.begin()
            for name, column, label in PETSC_EVENTS:
                try:
                    self._events[name] = (PETSc.Log.Event(name), column, label)
                except Exception:
                    continue
            self.available = bool(self._events)
        except Exception:
            self.available = False
        self.take()

    def take(self):
        """``{column: (seconds, count)}`` since the last call."""
        out = {}
        for name, (event, column, label) in self._events.items():
            try:
                info = event.getPerfInfo()
            except Exception:
                continue
            seconds = float(info.get('time', 0.0))
            count = int(info.get('count', 0))
            before = self._previous.get(name, (0.0, 0))
            out[column] = (seconds - before[0], count - before[1])
            self._previous[name] = (seconds, count)
        return out

    @staticmethod
    def columns():
        return [column for _, column, _ in PETSC_EVENTS]


def main():
    args = parse_args()
    deck = args.deck if os.path.isabs(args.deck) else os.path.join(ROOT, args.deck)
    start = date.fromisoformat(args.start_date)
    stem = os.path.splitext(os.path.basename(deck))[0]
    results = os.path.join(ROOT, 'results', stem + '_full')
    csv_path = os.path.join(results, 'well_rates.csv')
    timing_path = os.path.join(results, 'timing.csv')
    os.makedirs(results, exist_ok=True)

    started = time.time()
    init_kwargs = dict(RemoveZeroPoreVolume=True)
    if args.minporo is not None:
        init_kwargs['minporo'] = float(args.minporo)
    state0, model, schedule, solver = init_eclipse_problem_ad(
        deck, **init_kwargs)
    init_seconds = time.time() - started

    # The fastest assembly this codebase has: the diagonal representation
    # for the per-cell property chain, and the compiled fixed-width face
    # operators for the flux.  Both fall back on their own when a model or a
    # mode cannot use them, so setting them is a request, not an assertion.
    model.autodiff_backend = args.backend
    model.useFaceOperators = not args.no_face_operators
    model._face_flux_cache = None

    if args.linear_solver == 'amgcl-cpr':
        # Exactly what simulator_gui.py builds for method == "AMGCL CPR".
        from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD
        solver.LinearSolver = AMGCL_CPRSolverBlockAD(
            tolerance=1e-4, maxIterations=50,
            strategy=args.amgcl_strategy, decoupling=args.amgcl_decoupling)

    linear = getattr(solver, 'LinearSolver', None)
    if args.second_stage and hasattr(linear, 'second_stage'):
        linear.second_stage = args.second_stage
        # The solver caches its KSP against the sparsity pattern; a
        # different preconditioner has to rebuild it.
        linear._ksp = None
    if args.pressure_precond and hasattr(linear, 'pressure_precond'):
        linear.pressure_precond = args.pressure_precond
        linear._ksp = None
    nc = len(state0['pressure'])
    nsteps = len(schedule['step']['val'])
    max_steps = min(args.steps or nsteps, nsteps)

    print('deck + init      : %.1f s' % init_seconds, flush=True)
    print('active cells     : %d   phases oil=%s water=%s gas=%s'
          % (nc, model.oil, model.water, model.gas), flush=True)
    print('min porosity     : %s'
          % ('off' if args.minporo is None else '%g' % args.minporo), flush=True)
    print('assembly         : backend=%s  face operators=%s'
          % (args.backend, model.useFaceOperators), flush=True)
    print('linear solver    : %s(%s, %s, second stage=%s)'
          % (type(linear).__name__,
             getattr(linear, 'decoupling', '-'),
             getattr(linear, 'pressure_precond', '-'),
             getattr(linear, 'second_stage', '-')), flush=True)

    from PRSTCore.ad_core import mex
    print('compiled kernels : divergence=%s  face ops=%s'
          % (mex.load_discrete_divergence() is not None,
             mex.load_face_operators() is not None), flush=True)

    solver.verbose = not args.quiet
    solver.errorOnFailure = False
    if args.use_linesearch:
        # MRST NonLinearSolver line search: once oscillation/stagnation is
        # detected the Newton increment is bisected (residual reduction).
        solver.useLinesearch = True
    if args.enforce_residual_decrease:
        # NonLinearSolver.solveMinistep: a mini-step that stops making
        # progress is abandoned so the outer loop cuts the timestep rather
        # than burn Newton iterations on a failing step.
        solver.enforceResidualDecrease = True
    if args.acceptance_factor is not None:
        # PhysicalModel.stepFunction: an exhausted-iteration mini-step is
        # accepted when the residual is below acceptanceFactor*tol.
        solver.acceptanceFactor = args.acceptance_factor

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as handle:
            csv.writer(handle).writerow(
                ['step', 'time_days', 'date', 'dt_days', 'converged',
                 'nonlinear_iters', 'wall_s', 'well', 'status',
                 'qO_sm3d', 'qW_sm3d', 'qG_sm3d', 'bhp_bar'])
    with open(timing_path, 'w', newline='') as handle:
        csv.writer(handle).writerow(
            ['step', 'wall_s', 'newton_iters', 'assembly_s', 'assembly_calls',
             'linear_s', 'linear_calls', 'krylov_iters', 'other_s']
            + PetscEvents.columns())

    timer = PhaseTimer(model, solver)
    events = PetscEvents()
    if not events.available:
        print('petsc event log  : unavailable; the linear split will be blank',
              flush=True)

    # Step 0 of the saved history is the initial state, so the viewer opens on
    # the model before anything has been produced rather than on step 1.
    if args.save_states:
        from PRSTCore.visualization.results_io import per_cell_entries

        saved_states = [per_cell_entries(state0, nc)]
        saved_times = [0.0]
        saved_dates = [start.isoformat()]
        saved_wells = {}
    else:
        saved_states = saved_times = saved_dates = saved_wells = None

    totals = dict(wall=0.0, assembly=0.0, linear=0.0, newton=0, krylov=0)
    petsc_totals = {column: 0.0 for column in PetscEvents.columns()}
    print('\n=== running %d / %d report steps ===' % (max_steps, nsteps), flush=True)

    # The report-step driver is shared with the GUI (simulate_schedule); this
    # script only supplies the per-step bookkeeping: the terminal header, the
    # assembly/linear/PETSc timing split, the CSVs and the state collection.
    from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

    def on_solve_start(step_index, meta):
        # Header before the solve, and a reset of the delta instruments so
        # the measurement covers exactly the solve -- the same boundary the
        # loop used to draw around solveTimestep itself.
        if not args.quiet:
            print('\n===== REPORT STEP %d/%d  TIME=%.1f days (%s)  DT=%.1f d ====='
                  % (step_index + 1, nsteps, meta['time_days'],
                     meta['date'].strftime('%d-%b-%Y'),
                     meta['dt'] / 86400.0), flush=True)
        timer.take()
        events.take()

    def on_step(step_index, info):
        state = info['state']
        wall = info['wall']
        converged = info['converged']
        iterations = info['iterations']
        elapsed_days = info['time_days']
        when = info['date']
        dt = info['dt']
        forces = info['forces']

        assembly, assembly_calls, linear_s, linear_calls, krylov = timer.take()
        petsc = events.take()
        setup = petsc.get('pc_setup_s', (0.0, 0))[0]
        applied = petsc.get('pc_apply_s', (0.0, 0))[0]

        totals['wall'] += wall
        totals['assembly'] += assembly
        totals['linear'] += linear_s
        totals['newton'] += iterations
        totals['krylov'] += krylov
        for column in petsc_totals:
            petsc_totals[column] += petsc.get(column, (0.0, 0))[0]

        rows = []
        for well in state.get('wellSol', []):
            def value(key):
                raw = well.get(key)
                if raw is None:
                    return 0.0
                array = np.atleast_1d(np.asarray(raw, dtype=float))
                return float(array[0]) if array.size else 0.0
            rows.append([step_index + 1, '%.3f' % elapsed_days, when.isoformat(),
                         '%.3f' % (dt / 86400.0), int(converged), iterations,
                         '%.2f' % wall, str(well.get('name', '?')),
                         int(bool(well.get('status'))),
                         '%.6f' % (value('qOs') * 86400.0),
                         '%.6f' % (value('qWs') * 86400.0),
                         '%.6f' % (value('qGs') * 86400.0),
                         '%.6f' % (value('bhp') / 1e5)])
        with open(csv_path, 'a', newline='') as handle:
            csv.writer(handle).writerows(rows)
        with open(timing_path, 'a', newline='') as handle:
            csv.writer(handle).writerow(
                [step_index + 1, '%.3f' % wall, iterations,
                 '%.3f' % assembly, assembly_calls, '%.3f' % linear_s,
                 linear_calls, krylov, '%.3f' % (wall - assembly - linear_s)]
                + ['%.3f' % petsc.get(column, (0.0, 0))[0]
                   for column in PetscEvents.columns()])

        if saved_states is not None:
            # Per-cell entries only, found by shape rather than by name:
            # wellSol and the solver's scratch would multiply the file size
            # for nothing a 3D view can draw, and hard-coding the field names
            # is how a file ends up with no saturations in it.
            saved_states.append(per_cell_entries(state, nc))
            saved_times.append(elapsed_days)
            saved_dates.append(when.isoformat())
            # Wells open over the course of the schedule, so the union across
            # controls is the well set worth drawing -- the first control's
            # alone would show one well of a field that ends up with dozens.
            for well in forces.get('W') or []:
                saved_wells.setdefault(str(well.get('name', '')), well)

        other = wall - assembly - linear_s
        line = ('  step %3d  %6.1f s  newton %2d  |  assembly %5.1f s (%2d)  '
                'linear %5.1f s (%2d, %3d krylov)  other %4.1f s'
                % (step_index + 1, wall, iterations, assembly, assembly_calls,
                   linear_s, linear_calls, krylov, other))
        if events.available and linear_s > 0:
            line += '  |  pc setup %4.1f s  apply %4.1f s' % (setup, applied)
        if not converged:
            line += '   NOT CONVERGED'
        print(line, flush=True)

    run = simulate_schedule(model, state0, schedule, solver,
                            max_steps=max_steps, start=start,
                            on_step=on_step, on_solve_start=on_solve_start)
    wall = run['wall']
    print('\n=== %d steps in %.1f s (%.1f min) ==='
          % (max_steps, wall, wall / 60.0), flush=True)

    def share(seconds):
        return 100.0 * seconds / wall if wall > 0 else 0.0

    other = wall - totals['assembly'] - totals['linear']
    print('  assembly            %7.1f s  %5.1f%%   %d Newton systems'
          % (totals['assembly'], share(totals['assembly']), totals['newton']), flush=True)
    print('  linear solve        %7.1f s  %5.1f%%   %d Krylov iterations'
          % (totals['linear'], share(totals['linear']), totals['krylov']), flush=True)
    if events.available:
        setup_total = petsc_totals['pc_setup_s']
        apply_total = petsc_totals['pc_apply_s']
        print('      preconditioner setup   %7.1f s  %5.1f%% of the run'
              % (setup_total, share(setup_total)), flush=True)
        print('        of which ILU factorisation  %7.1f s'
              % petsc_totals['ilu_factor_s'], flush=True)
        print('        the rest is the pressure AMG hierarchy  %7.1f s'
              % (setup_total - petsc_totals['ilu_factor_s']), flush=True)
        print('      preconditioner apply   %7.1f s  %5.1f%% of the run'
              % (apply_total, share(apply_total)), flush=True)
        print('        of which ILU triangular solves  %7.1f s'
              % petsc_totals['ilu_solve_s'], flush=True)
        print('        the rest is the pressure AMG V-cycles and the'
              ' between-stage residual  %7.1f s'
              % (apply_total - petsc_totals['ilu_solve_s']), flush=True)
        print('      matrix-vector products %7.1f s' % petsc_totals['matvec_s'],
              flush=True)
        rest = totals['linear'] - setup_total - apply_total
        print('      other Krylov work      %7.1f s  %5.1f%% of the run'
              % (rest, share(rest)), flush=True)
    print('  everything else     %7.1f s  %5.1f%%' % (other, share(other)), flush=True)

    if max_steps:
        projected = wall / max_steps * nsteps
        print('\n  projected for all %d steps: %.0f s (%.1f min), plus %.0f s of set-up'
              % (nsteps, projected, projected / 60.0, init_seconds), flush=True)
    print('  well rates -> %s' % csv_path, flush=True)
    print('  timings    -> %s' % timing_path, flush=True)

    if saved_states is not None:
        from PRSTCore.visualization.results_io import save_states

        states_path = (os.path.join(results, 'states.npz')
                       if args.save_states is True else args.save_states)
        started = time.time()
        # model.G, not a grid rebuilt from the deck: the run's own options
        # (RemoveZeroPoreVolume here) decide the active set, and a re-read
        # deck can hand back a different one.
        save_states(states_path, saved_states, model.G,
                    W=list(saved_wells.values()),
                    times=saved_times, dates=saved_dates, model=model,
                    meta={'deck': deck, 'backend': args.backend,
                          'steps': max_steps})
        size_mb = os.path.getsize(states_path) / 1024.0 / 1024.0
        print('  states     -> %s  (%d steps, %.1f MB, %.1f s)'
              % (states_path, len(saved_states), size_mb,
                 time.time() - started), flush=True)


if __name__ == '__main__':
    main()
