"""Split a simulation's wall clock into assembly, linear solve and the rest.

Deciding what to make faster needs the split, not the total.  The two halves
answer different questions and have completely different remedies: residual
assembly is Python and numpy and is fixed by changing the automatic
differentiation representation, while the linear solve is compiled code and
is fixed by changing the preconditioner.  Optimising the wrong one is free
to look like progress and change nothing.

Timing is taken by wrapping ``model.get_equations`` and the linear solver's
``solveLinearProblem`` for the duration of a ``profile`` block, rather than
by putting counters in either.  The measurement therefore costs one function
call per Newton iteration -- not per cell, per face or per AD operation --
and the production path carries nothing when nobody is measuring.

Usage::

    with profile(model, solver) as prof:
        well_sols, states = simulate_schedule_ad(state0, model, schedule,
                                                 NonLinearSolver=solver)
    print(prof.summary())
"""

from __future__ import annotations

import time as _time
from contextlib import contextmanager


class SimulationProfile:
    """Accumulated wall time and call counts for one instrumented run."""

    __slots__ = ('assembly_seconds', 'assembly_calls',
                 'linear_seconds', 'linear_calls', 'linear_iterations',
                 'total_seconds', 'label')

    def __init__(self, label=''):
        self.label = str(label)
        self.assembly_seconds = 0.0
        self.assembly_calls = 0
        self.linear_seconds = 0.0
        self.linear_calls = 0
        self.linear_iterations = 0
        self.total_seconds = 0.0

    @property
    def other_seconds(self):
        """Everything that is neither assembly nor linear solve.

        State updates, convergence checks, well limit switching, report
        bookkeeping.  A large value here is itself a finding: it means the
        two measured phases are no longer where the time goes.
        """
        return max(0.0, self.total_seconds - self.assembly_seconds - self.linear_seconds)

    def as_dict(self):
        return {
            'label': self.label,
            'total_seconds': self.total_seconds,
            'assembly_seconds': self.assembly_seconds,
            'assembly_calls': self.assembly_calls,
            'linear_seconds': self.linear_seconds,
            'linear_calls': self.linear_calls,
            'linear_iterations': self.linear_iterations,
            'other_seconds': self.other_seconds,
        }

    def _share(self, seconds):
        if self.total_seconds <= 0.0:
            return 0.0
        return 100.0 * seconds / self.total_seconds

    def summary(self):
        """A few lines naming where the run's time went."""
        def line(name, seconds, calls=None):
            per = '' if not calls else '  (%d calls, %.1f ms each)' % (
                calls, 1000.0 * seconds / calls)
            return '  %-18s %8.2f s  %5.1f%%%s' % (name, seconds, self._share(seconds), per)

        rows = [
            'total               %8.2f s' % self.total_seconds,
            line('assembly', self.assembly_seconds, self.assembly_calls),
            line('linear solve', self.linear_seconds, self.linear_calls),
            line('other', self.other_seconds),
        ]
        if self.linear_iterations:
            rows.append('  %-18s %8d' % ('linear iterations', self.linear_iterations))
        if self.label:
            rows.insert(0, self.label)
        return '\n'.join(rows)


def _linear_solvers(solver):
    """The linear solvers a nonlinear solver will actually call.

    CPR holds a second solver for the pressure block, and its time is
    reported inside the outer solver's own call, so only the outermost one
    is wrapped -- wrapping both would double-count.
    """
    found = []
    for attribute in ('LinearSolver', 'linearSolver'):
        candidate = getattr(solver, attribute, None)
        if candidate is not None and hasattr(candidate, 'solveLinearProblem'):
            found.append(candidate)
    if not found and hasattr(solver, 'solveLinearProblem'):
        found.append(solver)
    return found


@contextmanager
def profile(model, solver=None, label=''):
    """Instrument ``model`` and ``solver`` for the duration of the block.

    Both objects are restored on the way out, including when the body
    raises -- a failed run that leaves a monkeypatched model behind would
    silently corrupt every later measurement in the same process.
    """
    prof = SimulationProfile(label)
    restore = []

    def patch(owner, name, wrapper):
        """Install ``wrapper`` on the instance, remembering how to undo it.

        The method normally lives on the class, so the wrapper goes into the
        instance dictionary and removing it again is a ``delattr``.  If the
        instance already had its own, that one has to be put back instead.
        """
        had_own = name in vars(owner)
        original = getattr(owner, name)
        setattr(owner, name, wrapper)
        restore.append((owner, name, original, had_own))
        return original

    if getattr(model, 'get_equations', None) is not None:
        def timed_equations(*args, **kwargs):
            started = _time.perf_counter()
            try:
                return original_equations(*args, **kwargs)
            finally:
                prof.assembly_seconds += _time.perf_counter() - started
                prof.assembly_calls += 1

        original_equations = patch(model, 'get_equations', timed_equations)

    def make_timed_solve(original):
        def timed_solve(*args, **kwargs):
            started = _time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                prof.linear_seconds += _time.perf_counter() - started
                prof.linear_calls += 1
            # (dx, residual, report) -- the report carries the Krylov count,
            # which separates "the preconditioner is weak" from "the solve is
            # slow", and only the solver knows it.
            if isinstance(result, tuple) and len(result) == 3 and isinstance(result[2], dict):
                prof.linear_iterations += int(result[2].get('Iterations', 0) or 0)
            return result
        return timed_solve

    for linear_solver in (_linear_solvers(solver) if solver is not None else []):
        # Bound through a factory rather than a closure over the loop
        # variable: every wrapper would otherwise call the last solver's
        # method, and with one solver the bug is invisible.
        patch(linear_solver, 'solveLinearProblem',
              make_timed_solve(linear_solver.solveLinearProblem))

    started = _time.perf_counter()
    try:
        yield prof
    finally:
        prof.total_seconds = _time.perf_counter() - started
        for owner, name, original, had_own in restore:
            if had_own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
