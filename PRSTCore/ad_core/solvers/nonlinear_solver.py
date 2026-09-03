import numpy as _np
from copy import deepcopy
import time as _time

from .linear_solver_ad import LinearSolverAD


class NonLinearSolver:
    """Simple MRST-style nonlinear solver with timestep cutting support."""

    def __init__(self, maxIterations=12, minIterations=1, maxTimestepCuts=6,
                 verbose=False, timeStepSelector=None, errorOnFailure=True,
                 continueOnFailure=False, linearSolver=None,
                 useRelaxation=False, relaxationParameter=1.0,
                 relaxationType='dampen', relaxationIncrement=0.1,
                 relaxationDecrement=None, minRelaxation=0.5,
                 maxRelaxation=1.0, oscillationThreshold=1.0,
                 useLinesearch=False, alwaysUseStabilization=False,
                 stagnateTol=1.0e-2, linesearchReductionFactor=0.5,
                 linesearchDecreaseFactor=1.0, linesearchMaxIterations=10,
                 linesearchConvergenceNames=None,
                 linesearchResidualScaling=None, linesearchReductionFn=None,
                 enforceResidualDecrease=False, acceptanceFactor=1.0):
        self.maxIterations = int(maxIterations)
        self.minIterations = int(minIterations)
        self.maxTimestepCuts = int(maxTimestepCuts)
        self.verbose = verbose
        self.errorOnFailure = errorOnFailure
        self.continueOnFailure = continueOnFailure
        self.timeStepSelector = timeStepSelector or SimpleTimeStepSelector()
        self.LinearSolver = linearSolver or LinearSolverAD(verbose=verbose)
        self.previousStepReport = None
        # Direct counterparts of the stabilization properties in MRST's
        # ``NonLinearSolver``.  ``initEclipseProblemAD`` enables the
        # relaxation branch by default for deck cases.
        self.useRelaxation = bool(useRelaxation)
        self.relaxationParameter = float(relaxationParameter)
        self.relaxationType = str(relaxationType)
        self.relaxationIncrement = float(relaxationIncrement)
        self.relaxationDecrement = (None if relaxationDecrement is None
                                    else float(relaxationDecrement))
        self.minRelaxation = float(minRelaxation)
        self.maxRelaxation = float(maxRelaxation)
        self.oscillationThreshold = float(oscillationThreshold)
        self.useLinesearch = bool(useLinesearch)
        self.alwaysUseStabilization = bool(alwaysUseStabilization)
        self.stagnateTol = float(stagnateTol)
        # NonLinearSolver.m line-search properties (lines 55--63).
        self.linesearchReductionFactor = float(linesearchReductionFactor)
        self.linesearchDecreaseFactor = float(linesearchDecreaseFactor)
        self.linesearchMaxIterations = int(linesearchMaxIterations)
        self.linesearchConvergenceNames = list(linesearchConvergenceNames or [])
        self.linesearchResidualScaling = (
            None if linesearchResidualScaling is None
            else _np.asarray(linesearchResidualScaling, dtype=float))
        self.linesearchReductionFn = linesearchReductionFn
        self.enforceResidualDecrease = bool(enforceResidualDecrease)
        # PhysicalModel.stepFunction: when the Newton iteration budget is
        # exhausted, a mini-step whose residual is below
        # ``acceptanceFactor*tol`` is still accepted (default 1 disables the
        # relaxed acceptance, exactly like MRST).
        self.acceptanceFactor = float(acceptanceFactor)
        self.convergenceIssues = False
        self.previousIncrement = None
        self._residualHistory = []
        self.iterationTraceReference = None
        self._traceMinistep = 0
        self._traceDt = _np.nan

    def setIterationTraceReference(self, reference):
        """Attach an optional MRST-style nonlinear trace for verbose output.

        ``reference`` is a list of dictionaries, one per ministep.  Each
        dictionary may contain ``dt``, ``iterations``, ``converged`` and a
        ``residuals`` list.  The residual list follows MRST's
        ``NonlinearReport`` convention: entry zero is the initial
        pre-update residual, entry one is after the first Newton update,
        and so on.
        """
        self.iterationTraceReference = reference
        return self

    def _reference_ministep(self, ministep=None):
        ref = self.iterationTraceReference
        if not ref:
            return None
        ix = int(self._traceMinistep if ministep is None else ministep) - 1
        if ix < 0 or ix >= len(ref):
            return None
        return ref[ix]

    @staticmethod
    def _max_abs(value):
        arr = _np.asarray(value, dtype=float).ravel()
        return float(_np.max(_np.abs(arr))) if arr.size else 0.0

    def printNewtonTrace(self, newton, values, converged, linearReport=None, names=None):
        """Print one Python Newton residual row, optionally against MRST."""
        if not self.verbose:
            return
        values = _np.asarray(values, dtype=float).ravel()
        py_max = self._max_abs(values)
        names = list(names or [])
        worst_py = int(_np.argmax(_np.abs(values))) if values.size else -1
        worst_name = names[worst_py] if 0 <= worst_py < len(names) else ''
        linear_time = 0.0
        linear_res = _np.nan
        linear_iter = None
        prec_type = None
        if isinstance(linearReport, dict):
            linear_time = float(linearReport.get('SolverTime', 0.0))
            linear_res = float(linearReport.get('Residual', _np.nan))
            linear_iter = linearReport.get('Iterations', None)
            pre = linearReport.get('PreconditionerReport', None)
            if isinstance(pre, dict):
                prec_type = pre.get('Type', None)
        prefix = "   Newton pre" if int(newton) == 0 else f"   Newton {int(newton)}"
        line = (
            f"{prefix}: residual={py_max:.6g}, "
            f"converged={bool(_np.all(converged))}"
        )
        if worst_name:
            line += f", worst={worst_name}"
        if int(newton) > 0:
            line += f", linear={linear_time:.2f}s"
            if linear_iter is not None:
                line += f", liniter={linear_iter}"
            if _np.isfinite(linear_res):
                line += f", linres={linear_res:.3e}"
            if prec_type:
                line += f", prec={prec_type}"

        ref_step = self._reference_ministep()
        if ref_step is not None:
            ref_values = ref_step.get('residuals', [])
            if int(newton) < len(ref_values):
                mrst_values = _np.asarray(ref_values[int(newton)], dtype=float).ravel()
                n = min(values.size, mrst_values.size)
                mrst_max = self._max_abs(mrst_values)
                if n:
                    delta = values[:n] - mrst_values[:n]
                    worst = int(_np.argmax(_np.abs(delta)))
                    diff = float(_np.abs(delta[worst]))
                else:
                    worst = -1
                    diff = _np.nan
                line += f" | MRST residual={mrst_max:.6g}, diff={diff:.3e}"
                if worst >= 0 and diff > 1.0e-10:
                    line += (
                        f", worst[{worst}]="
                        f"{values[worst]:.6g}/{mrst_values[worst]:.6g}"
                    )
            ref_linear = ref_step.get('linear_time', [])
            if int(newton) > 0 and int(newton) - 1 < len(ref_linear):
                mrst_linear = float(ref_linear[int(newton) - 1])
                line += f", MRST linear={mrst_linear:.2f}s"
        print(line, flush=True)

    def printMinistepTrace(self, ministep, dt, step_report, final_ministep=False):
        """Print one ministep summary, optionally against MRST."""
        if not self.verbose:
            return
        iterations = int(step_report.get('Iterations', 0))
        converged = bool(step_report.get('Converged', False))
        line = (
            f"   Ministep {int(ministep)}: dt={float(dt):.12g}, "
            f"converged={converged}, iterations={iterations}"
        )
        ref_step = self._reference_ministep(ministep)
        if ref_step is not None:
            rdt = float(ref_step.get('dt', _np.nan))
            rits = int(ref_step.get('iterations', -1))
            rconv = bool(ref_step.get('converged', False))
            line += (
                f" | MRST dt={rdt:.12g}, iterations={rits}, "
                f"converged={rconv}, dt_diff={float(dt) - rdt:.3e}, "
                f"iter_diff={iterations - rits}"
            )
        if final_ministep:
            line += " final"
        print(line, flush=True)

    def beginMinistep(self):
        """Initialize MRST's relaxation bookkeeping for one mini-step.

        This is ``NonLinearSolver.solveMinistep`` lines 357--358.  The
        compact Python model executes the Newton loop in ``stepFunction``,
        so the reset belongs at the corresponding mini-step boundary.
        """
        self.relaxationParameter = self.maxRelaxation
        self.convergenceIssues = False
        self._residualHistory = []

    def stabilizeNewtonIncrements(self, dx):
        """Port ``NonLinearSolver.stabilizeNewtonIncrements`` for a flat dx."""
        increment = _np.asarray(dx, dtype=float).ravel().copy()
        w = float(self.relaxationParameter)
        report = {'relaxationParameter': w}
        if w < 1.0:
            mode = self.relaxationType.lower()
            if mode == 'dampen':
                increment *= w
            elif mode == 'sor':
                if self.previousIncrement is not None:
                    previous = _np.asarray(self.previousIncrement, dtype=float).ravel()
                    if previous.size == increment.size:
                        increment = increment * w + (1.0 - w) * previous
            elif mode != 'none':
                raise ValueError("Unknown relaxationType: valid options are 'dampen', 'none' or 'sor'")
        self.previousIncrement = increment.copy()
        return increment, report

    def applyLinesearch(self, model, state0, state, problem, dx, drivingForces, **kwargs):
        """Port ``NonLinearSolver.applyLinesearch`` (residual bisection).

        MRST's line search is *not* a Wolfe search: it repeatedly scales the
        Newton increment by ``linesearchReductionFactor`` (default 1/2) and
        re-assembles the residual-only equations until the normalized
        residual drops below the pre-update value, up to
        ``linesearchMaxIterations`` tries (NonLinearSolver.m lines 487--535).
        ``PhysicalModel.stepFunction`` only enters it when the solver is
        struggling (``convergenceIssues`` or ``alwaysUseStabilization``)
        *and* ``useLinesearch`` is on.
        """
        factor = float(self.linesearchDecreaseFactor)
        converged = False
        iteration = int(problem.get('iterationNo', 1))
        dt = float(problem['dt'])
        # getConvergenceValues returns raw values; normalize like MRST's
        # ``val0 = val0./tol``.  The line-search comparison is purely
        # relative, so the per-equation residual scaling can stay unity
        # (MRST's own double normalization is equivalent under it).
        val0, tol, names = model.getConvergenceValues(problem)
        tol = _np.asarray(tol, dtype=float).ravel()
        val0 = _np.asarray(val0, dtype=float).ravel() / _np.maximum(tol, 1e-300)
        if self.linesearchResidualScaling is None or \
                self.linesearchResidualScaling.size != val0.size:
            self.linesearchResidualScaling = _np.ones(val0.size, dtype=float)
        ok = val0 <= 1.0
        active = self._linesearch_active_names(names)
        v_best = self._linesearch_apply_update(val0, ok, active)

        state_next = state
        its = 0
        dx_scale = _np.asarray(dx, dtype=float).ravel().copy()
        for its in range(1, int(self.linesearchMaxIterations) + 1):
            state_next = model.updateState(state, problem, dx_scale, drivingForces)
            problem_next, _ = model.get_equations(
                state0, state_next, dt, drivingForces=drivingForces,
                ResOnly=True, iteration=iteration, **kwargs)
            val, _, _ = model.getConvergenceValues(problem_next)
            val = _np.asarray(val, dtype=float).ravel() / _np.maximum(tol, 1e-300)
            ok = val <= 1.0
            v = self._linesearch_apply_update(val, ok, active)
            # Success when everything converged or the residual actually
            # dropped: ``all(ok) || (any(v < vBest*factor) && sum(v)<=sum(vBest))``.
            if bool(_np.all(ok)) or (
                    bool(_np.any(v < v_best * factor)) and
                    float(_np.sum(v)) <= float(_np.sum(v_best))):
                converged = True
                break
            dx_scale = dx_scale * float(self.linesearchReductionFactor)
        line_report = {'Iterations': its, 'Converged': converged}
        return state_next, {}, line_report

    def _linesearch_apply_update(self, v, ok, active):
        """Port ``linesearchApplyUpdate``: scale, restrict to the active
        equations and (absent a custom reduction function) zero out the
        already-converged entries so they do not mask real reduction."""
        v = _np.asarray(v, dtype=float).ravel() / _np.asarray(
            self.linesearchResidualScaling, dtype=float).ravel()
        v = v[active]
        ok = ok[active]
        if self.linesearchReductionFn is not None:
            v = self.linesearchReductionFn(v)
        else:
            v = v * (~ok)
        return v

    def _linesearch_active_names(self, names):
        """Port ``getActiveNames``: which equations the line search tracks."""
        names = list(names or [])
        if not self.linesearchConvergenceNames:
            return _np.ones(len(names), dtype=bool)
        wanted = set(self.linesearchConvergenceNames)
        return _np.array([n in wanted for n in names], dtype=bool)

    def updateRelaxationFromResidual(self, values, converged):
        """Port MRST's residual-history relaxation decision (lines 402--437)."""
        if not (self.useRelaxation or self.useLinesearch or self.alwaysUseStabilization):
            return
        current = _np.asarray(values, dtype=float).ravel().copy()
        is_ok = _np.asarray(converged, dtype=bool).ravel()
        self._residualHistory.append(current)
        index = len(self._residualHistory)  # MATLAB's one-based ``i``.
        if index < 3:
            oscillating = _np.zeros(current.size, dtype=bool)
        else:
            old, mid, nxt = self._residualHistory[-3:]
            with _np.errstate(divide='ignore', invalid='ignore'):
                oscillating = ((nxt - mid) / (mid - old)) < 0.0
        if index < 2:
            stagnated = _np.zeros(current.size, dtype=bool)
        else:
            previous = self._residualHistory[-2]
            with _np.errstate(divide='ignore', invalid='ignore'):
                stagnated = _np.abs(current - previous) / previous < self.stagnateTol
        bad = oscillating | stagnated
        relax = (int(_np.sum(bad & ~is_ok)) >=
                 self.oscillationThreshold * int(_np.sum(~is_ok)) and
                 not bool(_np.all(is_ok)))
        if relax:
            self.convergenceIssues = True
            decrement = (self.relaxationIncrement if self.relaxationDecrement is None
                         else self.relaxationDecrement)
            self.relaxationParameter = max(self.relaxationParameter - decrement,
                                           self.minRelaxation)
        else:
            self.relaxationParameter = min(self.relaxationParameter + self.relaxationIncrement,
                                           self.maxRelaxation)

    def solveTimestep(self, state0, dT, model, *args, **kwargs):
        timer = _time.perf_counter()
        opts = {**kwargs}
        state = deepcopy(opts.pop('initialGuess', state0))
        drivingForces = opts.pop('drivingForces', {})
        control_id = opts.pop('controlId', 0)
        linear_solver = opts.pop('linsolver', self.LinearSolver)

        # ``NonLinearSolver.m``: prepare the report step before anything
        # else, and let the prepared state become the initial guess that
        # timestep cuts reuse.  This is where a facility model settles
        # whatever is fixed for the whole report step rather than for each
        # mini-step -- RESV wells convert their reservoir-volume target into
        # surface-rate conversion factors here.  The call was missing, so a
        # deck with RESV controls -- Norne, SPE10 model 2 -- reached the
        # first residual with no factors and stopped at "Unsupported MRST
        # well control type", even though everything needed to support them
        # was in place on both sides of this line.
        if hasattr(model, 'prepareReportstep'):
            prepared = model.prepareReportstep(state, state0, dT, drivingForces)
            if isinstance(prepared, tuple) and len(prepared) == 2:
                model, state = prepared
            else:
                state = prepared
            # MRST's ``opt.initialGuess = state`` right after: the prepared
            # state is what a cut mini-step restarts from, and ``state`` is
            # copied into ``initial_guess`` a few lines below.

        self.timeStepSelector.newControlStep(drivingForces, control_id)
        dt_min = dT / (2**self.maxTimestepCuts)
        t_local = 0.0
        acceptCount = 0
        reports = []
        ministates = []
        failure = False
        cutting_count = 0
        dt_prev = _np.nan
        state_prev = None
        state0_inner = deepcopy(state0)
        initial_guess = deepcopy(state)
        base_time = float(state0.get('time', 0.0))
        # MRST passes the previously attempted mini-step to the selector,
        # not the whole remaining report step.
        candidate_dt = float(dT)
        while True:
            if failure:
                dt_selector = self.timeStepSelector.cutTimestep(dt_prev, candidate_dt, model,
                                                                self, state_prev, state0_inner,
                                                                drivingForces)
            else:
                dt_selector = self.timeStepSelector.pickTimestep(dt_prev, candidate_dt, model,
                                                                 self, state_prev, state0_inner,
                                                                 drivingForces)
            # MRST also lets a physical model impose a maximum mini-step
            # (NonLinearSolver.m:193--200).  The fully implicit SPE1 model
            # returns ``inf``, but honoring the hook is necessary for the
            # other deck models.
            if hasattr(model, 'getMaximumTimestep'):
                dt_model = float(model.getMaximumTimestep(
                    state, state0_inner, dT - t_local, drivingForces
                ))
            else:
                dt_model = _np.inf
            dt_choice = float(min(dt_selector, dt_model))
            dt = dt_choice
            if t_local + dt >= dT:
                dt = dT - t_local
                final_ministep = True
            else:
                final_ministep = False
            if isinstance(state, dict):
                state['time'] = base_time + t_local + dt
            else:
                setattr(state, 'time', base_time + t_local + dt)

            # The current Python model performs the per-Newton loop inside
            # ``stepFunction``.  This is the exact mini-step boundary where
            # MRST's ``solveMinistep`` resets its relaxation state.
            if hasattr(self, 'beginMinistep'):
                self.beginMinistep()
            self._traceMinistep = acceptCount + 1
            self._traceDt = float(dt)

            # Exact NonLinearSolver.solveMinistep order: the facility model
            # updates connection pressure drops and control limits before the
            # first Newton equation evaluation of every accepted mini-step.
            if hasattr(model, 'prepareTimestep'):
                prepared = model.prepareTimestep(state, state0_inner, dt, drivingForces)
                if isinstance(prepared, tuple) and len(prepared) == 2:
                    model, state = prepared
                else:
                    state = prepared

            state, step_report = model.stepFunction(deepcopy(state), deepcopy(state0_inner), dt,
                                                   drivingForces=drivingForces,
                                                   linsolver=linear_solver,
                                                   nonlinsolver=self,
                                                   iteration=1)
            local_report = dict(step_report)
            local_report['Timestep'] = float(dt)
            local_report['ControlId'] = control_id
            local_report['Step'] = acceptCount + 1
            local_report['LocalTime'] = float(t_local + dt)
            linrep = local_report.get('LinearSolver', {}) if isinstance(local_report, dict) else {}
            nonlinear_entry = {
                'Converged': bool(local_report.get('Converged', False)),
                'Iterations': int(local_report.get('Iterations', 0)),
                'Residuals': local_report.get('Residuals', None),
                'ResidualsConverged': local_report.get('ResidualsConverged', None),
                'SolverTime': float(linrep.get('SolverTime', 0.0) if isinstance(linrep, dict) else 0.0),
                'LinearSolver': linrep,
            }
            solver_key = type(self.LinearSolver).__name__
            nonlinear_entry[solver_key] = linrep
            local_report['NonlinearReport'] = [nonlinear_entry]
            reports.append(local_report)
            ministates.append(deepcopy(state))
            # SimpleTimeStepSelector stores a history even for the final
            # mini-step when it was not merely clipped to hit report time.
            # MRST compares the report-clipped step against the selector's
            # *unclipped* proposal.  A short final remainder must not enter
            # the iteration-history controller (it would otherwise distort
            # the next report step's first mini-step).
            if (not final_ministep) or (dt_selector > 0 and dt / dt_selector > 0.9):
                self.timeStepSelector.storeTimestep(local_report)
            self.printMinistepTrace(acceptCount + 1, dt, step_report, final_ministep)
            if step_report['Converged']:
                # NonLinearSolver.m calls this on every converged ministep:
                #     [state, r] = model.updateAfterConvergence(state0, state, dt, forces)
                # It is where end-of-step state that is *not* a primary
                # variable gets advanced -- the polymer/surfactant maximum
                # concentrations (which drive irreversible adsorption,
                # PLYROCK item 4 == 2) and the aquifer pressure/volume.
                # Skipping it leaves those frozen at their initial values,
                # silently degrading irreversible adsorption to the fully
                # reversible model.
                update_hook = getattr(model, 'updateAfterConvergence', None)
                if callable(update_hook):
                    updated = update_hook(state0_inner, state, dt, drivingForces)
                    if isinstance(updated, tuple):
                        updated, final_update = updated[0], updated[1]
                        local_report['FinalUpdate'] = final_update
                    if updated is not None:
                        state = updated
                    # ``ministates`` already recorded the pre-hook state.
                    ministates[-1] = deepcopy(state)
                acceptCount += 1
                t_local += dt
                previous_inner = deepcopy(state0_inner)
                state0_inner = deepcopy(state)
                dt_prev = dt
                state_prev = previous_inner
                failure = False
                candidate_dt = dt
                if final_ministep or t_local >= dT - 1e-12:
                    break
                continue
            if dt <= dt_min:
                failure = True
                break
            if self.verbose:
                print(f"   Cutting timestep from {dt} to {dt/2}")
            state = deepcopy(initial_guess if acceptCount == 0 else state0_inner)
            dt_prev = dt
            state_prev = deepcopy(state0_inner)
            candidate_dt = dt
            failure = True
            cutting_count += 1
        if failure and self.errorOnFailure:
            raise RuntimeError('Nonlinear solver failed to converge')
        report = {
            'Iterations': sum(r['Iterations'] for r in reports),
            'Converged': not failure,
            'EarlyStop': False,
            'Time': float(state.get('time', 0.0)),
            'StepSize': float(dT),
            'Timestep': float(dT),
            'ControlId': control_id,
            'MinistepCuttingCount': int(cutting_count),
            'AcceptedMinisteps': int(acceptCount),
            'SimulationTime': float(_time.perf_counter() - timer),
            'StepReports': reports,
            'NonlinearReport': [
                {
                    'Converged': not failure,
                    'Iterations': int(sum(r.get('Iterations', 0) for r in reports)),
                    'SolverName': type(self).__name__,
                }
            ],
        }
        return state, report, ministates


class SimpleTimeStepSelector:
    """Direct port of MRST's SimpleTimeStepSelector control logic."""

    def __init__(self, maxTimestep=_np.inf, minTimestep=0.0,
                 maxHistoryLength=50, maxRelativeAdjustment=2.0,
                 minRelativeAdjustment=0.5, firstRampupStep=_np.inf,
                 firstRampupStepRelative=1.0, resetOnControlsChanged=False):
        self.maxTimestep = float(maxTimestep)
        self.minTimestep = float(minTimestep)
        self.maxHistoryLength = int(maxHistoryLength)
        self.maxRelativeAdjustment = float(maxRelativeAdjustment)
        self.minRelativeAdjustment = float(minRelativeAdjustment)
        self.firstRampupStep = float(firstRampupStep)
        self.firstRampupStepRelative = float(firstRampupStepRelative)
        self.resetOnControlsChanged = bool(resetOnControlsChanged)
        self.reset()

    def reset(self):
        self.history = []
        self.previousControl = None
        self.isStartOfCtrlStep = True
        self.isFirstStep = True
        self.controlsChanged = True
        self.stepLimitedByHardLimits = True

    def newControlStep(self, drivingForces, control_id=None):
        self.isStartOfCtrlStep = True
        # initEclipseProblemAD uses control id as the selector identity.
        identity = 0 if control_id is None or not _np.isfinite(control_id) else int(control_id)
        if self.previousControl is None or self.previousControl != identity:
            if self.resetOnControlsChanged:
                self.reset()
            self.controlsChanged = True
            self.previousControl = identity
        else:
            self.controlsChanged = False

    def storeTimestep(self, report):
        self.history.append(dict(report))
        if len(self.history) > self.maxHistoryLength:
            self.history = self.history[-self.maxHistoryLength:]

    def computeTimestep(self, dt, dt_prev, model, solver, state_prev, state_curr, forces):
        return float(dt)

    def pickTimestep(self, dt_prev, dt, model, solver, state_prev, state_curr, drivingForces):
        if self.controlsChanged and (self.resetOnControlsChanged or self.isFirstStep):
            dt = min(float(dt), self.firstRampupStepRelative * float(dt))
            dt = min(float(dt), self.firstRampupStep)
            self.stepLimitedByHardLimits = True
        dt0 = float(dt)
        dt_new = float(self.computeTimestep(dt, dt_prev, model, solver, state_prev, state_curr, drivingForces))
        change = dt_new / dt if dt != 0.0 else 1.0
        if not self.isStartOfCtrlStep:
            change = min(change, self.maxRelativeAdjustment)
            change = max(change, self.minRelativeAdjustment)
        dt = dt * change
        dt = min(self.maxTimestep, max(self.minTimestep, dt))
        self.stepLimitedByHardLimits = bool(dt != dt_new)
        self.isStartOfCtrlStep = False
        self.isFirstStep = False
        self.controlsChanged = False
        return float(dt)

    def cutTimestep(self, dt_prev, dt, model, solver, state_prev, state_curr, drivingForces):
        return float(dt) * self.minRelativeAdjustment


class IterationCountTimeStepSelector(SimpleTimeStepSelector):
    """MRST ``IterationCountTimeStepSelector`` (TimestepStrategy=iteration)."""

    def __init__(self, targetIterationCount=5, iterationOffset=5, **kwargs):
        super().__init__(**kwargs)
        self.targetIterationCount = float(targetIterationCount)
        self.iterationOffset = float(iterationOffset)

    def computeTimestep(self, dt, dt_prev, model, solver, state_prev, state_curr, forces):
        hist = self.history
        if not hist or not bool(hist[-1].get('Converged', False)) or \
                (self.controlsChanged and self.resetOnControlsChanged):
            return float(dt)
        iterations = float(hist[-1].get('Iterations', 0))
        if len(hist) > 1:
            restart = self.stepLimitedByHardLimits or not bool(hist[-2].get('Converged', False))
        else:
            restart = True
        maxits = float(solver.maxIterations) + self.iterationOffset
        offset = self.iterationOffset
        tol = (self.targetIterationCount + offset) / maxits
        le1 = (iterations + offset) / maxits
        dt1 = float(hist[-1]['Timestep'])
        if restart:
            return (tol / le1) * dt1
        le0 = (float(hist[-2].get('Iterations', 0)) + offset) / maxits
        dt0 = float(hist[-2]['Timestep'])
        return (dt1 / dt0) * (tol * le0 / (le1 * le1)) * dt1


class BackslashSolver:
    def solveLinearProblem(self, problem, model):
        J = _np.asarray(problem['Jacobian'], dtype=float)
        r = _np.asarray(problem['Residuals'], dtype=float)
        if J.ndim != 2 or r.ndim != 1:
            raise ValueError('Jacobian and residual shapes are invalid')
        if J.shape[0] != J.shape[1] or J.shape[0] != r.size:
            raise ValueError('Jacobian must be square and match residual length')
        dx = _np.linalg.solve(J, -r)
        report = {'Residual': _np.linalg.norm(r)}
        return dx, None, report
