"""The report-step driver shared by the command line runner and the GUI.

The loop that marches a model through its schedule -- update controls, pick
the step length, run the non-linear solve, report -- used to live inside
``scripts/run_t142_full.py``.  The interactive ``SimulatorWindow`` needs the
same loop with a progress callback instead of a terminal print, and two
copies of the driver is how one of them drifts.  This module is the single
driver: ``scripts/run_t142_full.py`` and
:mod:`PRSTCore.visualization.simulator_gui` both call :func:`simulate_schedule`
and differ only in what their ``on_step`` callback does with each finished
report step.

The driver is deliberately dependency-free: it does not import results_io,
well_curves or any visualisation code.  Callers collect what they need out of
the per-step ``info`` dict in their ``on_step`` callback.
"""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import timedelta


def simulate_schedule(model, state0, schedule, solver, max_steps=None,
                      start=None, on_step=None, on_solve_start=None,
                      should_stop=None):
    """Run the report steps of ``schedule`` with ``solver`` on ``model``.

    This is the exact loop MRST's ``simulateScheduleAD`` drives and
    ``scripts/run_t142_full.py`` used to own: for each report step, update
    the active controls, run ``solveTimestep``, and hand the finished step
    to ``on_step``.

    Parameters
    ----------
    model : GenericBlackOilModel
    state0 : dict
        The initial state returned by ``init_eclipse_problem_ad``.
    schedule : dict
        The deck's schedule (``step.val``/``step.control``/``control``).
    solver : NonLinearSolver
    max_steps : int, optional
        Run at most this many report steps (default: the whole schedule).
    start : datetime.date, optional
        Simulation start date; used to label each step with a calendar date.
    on_step : callable(index, info), optional
        Called after every finished report step.  ``info`` is a dict with
        ``index`` (0-based), ``state`` (the raw state, including ``wellSol``),
        ``report``, ``wall`` (seconds), ``time_days``, ``date``, ``dt``,
        ``converged``, ``iterations``, ``wellSol`` and ``forces``.
    on_solve_start : callable(index, meta), optional
        Called immediately before each report step's solve.  ``meta`` is the
        step's bookkeeping (``index``, ``time_days``, ``date``, ``dt``,
        ``forces``).  Callers that keep delta instruments (assembly/linear
        timers, PETSc event logs) reset them here so the measurement covers
        exactly the solve, as the standalone runner used to.
    should_stop : callable() -> bool, optional
        Checked between report steps; return True to stop the run early.

    Returns
    -------
    dict
        ``steps`` -- list of per-step info dicts (without the raw ``state``,
        which is only passed to ``on_step`` so callers keep a compact copy),
        ``wall`` -- total wall time in seconds, ``nsteps`` -- steps actually
        run, ``schedule_steps`` -- report steps in the schedule.
    """
    nsteps = int(len(schedule['step']['val']))
    if max_steps is None:
        max_steps = nsteps
    else:
        max_steps = int(min(max_steps, nsteps))

    state = state0
    elapsed_days = 0.0
    run_started = time.time()
    steps = []

    for step_index in range(max_steps):
        if should_stop is not None and should_stop():
            break
        control = int(schedule['step']['control'][step_index])
        forces = model.getDrivingForces(schedule['control'][control])
        model, state = model.updateForChangedControls(state, forces)
        dt = float(schedule['step']['val'][step_index])
        elapsed_days += dt / 86400.0
        when = (start + timedelta(days=elapsed_days)) if start is not None \
            else None

        step_started = time.time()
        meta = {'index': step_index, 'time_days': elapsed_days, 'date': when,
                'dt': dt, 'forces': forces}
        if on_solve_start is not None:
            on_solve_start(step_index, meta)
        state, report, _ = solver.solveTimestep(
            deepcopy(state), dt, model, drivingForces=forces,
            initialGuess=deepcopy(state), controlId=control)
        wall = time.time() - step_started

        converged = bool(report.get('Converged'))
        iterations = int(report.get('Iterations', 0))
        info = dict(meta, state=state, report=report, wall=wall,
                    converged=converged, iterations=iterations,
                    wellSol=state.get('wellSol', []))
        steps.append(info)
        if on_step is not None:
            on_step(step_index, info)

    return {
        'steps': steps,
        'wall': time.time() - run_started,
        'nsteps': len(steps),
        'schedule_steps': nsteps,
    }
