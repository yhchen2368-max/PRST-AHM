import numpy as _np
import time as _time
from copy import deepcopy

from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel
from PRSTCore.ad_core.solvers import NonLinearSolver


def _wrap_minimal_model(model):
    if hasattr(model, 'validateState') and hasattr(model, 'stepFunction'):
        return model
    wrapper = GenericBlackOilModel()
    if isinstance(model, dict):
        for key, value in model.items():
            setattr(wrapper, key, value)
    return wrapper


def simulate_schedule_ad(state0, model, schedule, nonlinear_solver=None, verbose=False, **kwargs):
    if state0 is None:
        raise ValueError('Initial state state0 must be provided')
    if model is None:
        raise ValueError('Model must be provided')
    if schedule is None:
        raise ValueError('Schedule must be provided')

    model = _wrap_minimal_model(model)
    state = model.validateState(state0)
    schedule = model.validateSchedule(schedule)

    nsteps = len(schedule['step']['val'])
    well_sols = []
    states = []
    return_report = bool(kwargs.pop('return_report', False))
    timer = _time.perf_counter()

    if nonlinear_solver is None and 'NonLinearSolver' in kwargs:
        nonlinear_solver = kwargs.pop('NonLinearSolver')

    solver = nonlinear_solver
    if isinstance(solver, dict):
        solver = NonLinearSolver(
            maxIterations=solver.get('maxIterations', 25),
            minIterations=solver.get('minIterations', 1),
            maxTimestepCuts=solver.get('maxTimestepCuts', 6),
            verbose=solver.get('verbose', verbose),
            errorOnFailure=solver.get('errorOnFailure', True),
            continueOnFailure=solver.get('continueOnFailure', False),
            linearSolver=solver.get('linearSolver', solver.get('LinearSolver', None)),
        )
    if solver is None:
        solver = NonLinearSolver(verbose=verbose)
    if hasattr(solver, 'timeStepSelector') and hasattr(solver.timeStepSelector, 'reset'):
        solver.timeStepSelector.reset()

    output_ministeps = bool(kwargs.pop('OutputMinisteps', False))
    process_output_fn = kwargs.pop('processOutputFn', None)
    control_logic_fn = kwargs.pop('controlLogicFn', None)
    after_step_fn = kwargs.pop('afterStepFn', None)
    restart_step = int(kwargs.pop('restartStep', 1))

    prev_control = None
    control_step_reports = []

    # Time-varying pore volume and transmissibility, if the schedule
    # carries them. A deck with MULTPV or MULTFLT changing over the
    # schedule needs the operators rebuilt at each control step, and the
    # multipliers accumulate: step 3's value is the product of steps 1
    # to 3, not step 3 alone. Nothing happens unless the schedule has
    # these fields, which is why an ordinary run is unaffected.
    base_pv = _base_operator(model, 'pv') if 'multpv' in schedule else None
    base_trans = _base_operator(model, 'T') if 'multipliers' in schedule \
        else None

    for step_idx in range(restart_step - 1, nsteps):
        ctrl_idx = int(schedule['step']['control'][step_idx])
        control = schedule['control'][ctrl_idx]

        if base_pv is not None:
            _apply_step_multiplier(model, 'pv', base_pv,
                                   schedule['multpv'], ctrl_idx)
        if base_trans is not None:
            _apply_step_multiplier(model, 'T', base_trans,
                                   schedule['multipliers'], ctrl_idx)

        forces = model.getDrivingForces(control)

        if ctrl_idx != prev_control:
            model, state = model.updateForChangedControls(state, forces)
            prev_control = ctrl_idx

        dt = float(schedule['step']['val'][step_idx])
        state0 = deepcopy(state)
        state, report, ministates = solver.solveTimestep(state0, dt, model,
                                                         drivingForces=forces,
                                                         initialGuess=deepcopy(state),
                                                         controlId=ctrl_idx,
                                                         **kwargs)

        substates = ministates if output_ministeps else [state]

        well_sols_step = [s.get('wellSol', []) for s in substates]
        if process_output_fn is not None:
            substates, well_sols_step, report = process_output_fn(substates, well_sols_step, report)

        if control_logic_fn is not None:
            state, schedule, report, altered = control_logic_fn(state, schedule, report, step_idx)
            if altered:
                prev_control = None
                if substates:
                    substates[-1] = state
                if well_sols_step:
                    well_sols_step[-1] = state.get('wellSol', [])

        well_sols.extend(well_sols_step)
        states.extend(deepcopy(substates))

        control_report = {
            'ControlStep': int(step_idx + 1),
            'ControlId': int(ctrl_idx),
            'Timestep': float(dt),
            'Converged': bool(report.get('Converged', True)),
            'Iterations': int(report.get('Iterations', 0)),
            'SimulationTime': float(report.get('SimulationTime', 0.0)),
            'StepReports': list(report.get('StepReports', [])),
            'NonlinearReport': list(report.get('NonlinearReport', [])),
            'MinistepCount': int(len(report.get('StepReports', []))),
        }
        control_step_reports.append(control_report)

        if after_step_fn is not None:
            model, states, well_sols, ok = after_step_fn(model, states, well_sols, solver, schedule, [])
            if not ok:
                break

        if verbose:
            print(f'Step {step_idx+1}/{nsteps}: dt={dt}, converged={report.get("Converged", True)}')

        if not report.get('Converged', True):
            raise RuntimeError('Nonlinear solver failed to converge at step %d' % (step_idx + 1))

    schedulereport = {
        'Converged': all(r.get('Converged', False) for r in control_step_reports) if control_step_reports else True,
        'NumControlSteps': int(len(control_step_reports)),
        'SimulationTime': float(_time.perf_counter() - timer),
        'ControlstepReports': control_step_reports,
    }
    if return_report:
        return well_sols, states, schedulereport
    return well_sols, states


def _base_operator(model, name):
    """The unmultiplied operator, kept so each step multiplies the
    original rather than compounding on the previous step's result."""
    ops = getattr(model, 'operators', None)
    if ops is None and isinstance(model, dict):
        ops = model.get('operators')
    if not isinstance(ops, dict) or name not in ops:
        return None
    return _np.array(ops[name], dtype=float, copy=True)


def _apply_step_multiplier(model, name, base, multipliers, ctrl_idx):
    """Port of MRST-0's ``updateStepPV`` / ``updateStepTrans``.

    The multiplier for a control step is the product of every step up to
    and including it -- MRST's ``getCurrentMultipliers`` takes
    ``indices = 1:step`` -- so a MULTPV applied at step 2 stays applied
    at step 5.
    """
    factor = _current_multiplier(multipliers, ctrl_idx, base.size)
    if factor is None:
        return
    ops = getattr(model, 'operators', None)
    if ops is None and isinstance(model, dict):
        ops = model.get('operators')
    ops[name] = base * factor


def _current_multiplier(multipliers, step, n):
    """Port of ``getCurrentMultipliers``: the cumulative product up to
    ``step``, or None when there is nothing to apply."""
    if multipliers is None:
        return None
    values = multipliers[:step + 1] if not isinstance(multipliers, dict) \
        else None
    if values is None:
        # A dict of per-field multiplier lists; only the flat form is
        # used by anything here, so a dict is left alone rather than
        # guessed at.
        return None
    factor = _np.ones(n)
    for entry in values:
        entry = _np.atleast_1d(_np.asarray(entry, dtype=float)).ravel()
        if entry.size == 1:
            factor = factor * float(entry[0])
        elif entry.size == n:
            factor = factor * entry
    return factor
