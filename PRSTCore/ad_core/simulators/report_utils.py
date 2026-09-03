import numpy as np


def convert_report_to_schedule(report, schedule):
    """Create a schedule using converged ministeps from a schedule report."""
    timesteps = []
    controls = []
    control_reports = report.get('ControlstepReports', report if isinstance(report, list) else [])
    numsteps = np.full((len(control_reports),), np.nan, dtype=float)
    for i, cr in enumerate(control_reports):
        step_reports = [sr for sr in cr.get('StepReports', []) if sr.get('Converged', False)]
        timesteps.extend(float(sr.get('Timestep', 0.0)) for sr in step_reports)
        controls.extend([int(schedule['step']['control'][i])] * len(step_reports))
        numsteps[i] = len(step_reports)

    out = {
        'step': {
            'val': np.asarray(timesteps, dtype=float),
            'control': np.asarray(controls, dtype=int),
        },
        'control': schedule.get('control', []),
    }
    return out, np.asarray(timesteps, dtype=float), numsteps


def get_report_ministeps(report):
    """Get converged ministep sizes from a schedule report."""
    timesteps = []
    control_reports = report.get('ControlstepReports', report if isinstance(report, list) else [])
    for cr in control_reports:
        for sr in cr.get('StepReports', []):
            if sr.get('Converged', False):
                timesteps.append(float(sr.get('Timestep', 0.0)))
    return np.asarray(timesteps, dtype=float)


def get_report_output(reports, kind='nonlinearIterations', ministeps=False, solver=None):
    """Extract aggregate scalar outputs from schedule reports."""
    control_reports = reports.get('ControlstepReports', reports if isinstance(reports, list) else [])
    totals = []
    wasted = []
    cuts = []
    time = []

    for cr in control_reports:
        step_reports = cr.get('StepReports', [])
        iter_reports = step_reports if ministeps else [cr]
        for rep in iter_reports:
            nlr = rep.get('NonlinearReport', []) if isinstance(rep, dict) else []
            nli = nlr[0] if nlr else {}

            if kind == 'nonlinearIterations':
                val = float(nli.get('Iterations', rep.get('Iterations', 0)))
            elif kind == 'nonlinearSolverTime':
                val = float(rep.get('SimulationTime', nli.get('SolverTime', 0.0)))
            elif kind == 'linearIterations':
                lin = _extract_linear_report(rep, nli, solver)
                val = float(lin.get('Iterations', 0))
            elif kind == 'linearSolverTime':
                lin = _extract_linear_report(rep, nli, solver)
                val = float(lin.get('LinearSolutionTime', lin.get('SolverTime', 0.0)))
            else:
                raise ValueError('Unsupported kind: %s' % kind)

            converged = bool(rep.get('Converged', True))
            totals.append(val)
            wasted.append(0.0 if converged else val)
            cuts.append(0.0 if converged else 1.0)
            time.append(float(rep.get('Time', rep.get('LocalTime', 0.0))))

    return {
        'total': np.asarray(totals, dtype=float),
        'wasted': np.asarray(wasted, dtype=float),
        'cuts': np.asarray(cuts, dtype=float),
        'time': np.asarray(time, dtype=float),
    }


def _extract_linear_report(rep, nonlinear_entry, solver_name):
    if not isinstance(rep, dict):
        return {}
    if solver_name:
        if isinstance(nonlinear_entry, dict) and solver_name in nonlinear_entry:
            val = nonlinear_entry.get(solver_name, {})
            return val if isinstance(val, dict) else {}
        return {}

    lin = rep.get('LinearSolver', {})
    if isinstance(lin, dict) and lin:
        return lin
    if isinstance(nonlinear_entry, dict):
        lin = nonlinear_entry.get('LinearSolver', {})
        if isinstance(lin, dict):
            return lin
    return {}
