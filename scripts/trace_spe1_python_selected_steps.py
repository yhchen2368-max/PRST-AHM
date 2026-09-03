"""Record Python Newton states at the MRST-selected SPE1 report steps.

This is diagnostic-only.  It wraps the normal convergence call and leaves
the solver/model equations unchanged, so every saved state corresponds to
the same post-assembly point that MRST exposes in ``NonlinearReport``.
"""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad


DECK = ROOT / 'examples' / 'SPE1' / 'BENCH_SPE1.DATA'
OUTPUT = ROOT / 'spe1_python_selected_trace.npz'
SELECTED = {22, 58, 90}


def main():
    state, model, schedule, solver = init_eclipse_problem_ad(str(DECK))
    end_times = np.cumsum(np.asarray(schedule['step']['val'], dtype=float))
    targets = {float(end_times[i - 1]): i for i in SELECTED}
    records = []
    original_check = model.checkConvergence

    def record_check(problem):
        converged, values, names = original_check(problem)
        trial = problem.get('State', {}) if isinstance(problem, dict) else {}
        time = float(trial.get('time', np.nan)) if isinstance(trial, dict) else np.nan
        for target_time, step in targets.items():
            if np.isclose(time, target_time, rtol=0.0, atol=1e-7):
                wells = trial.get('wellSol', [])
                records.append({
                    'step': step,
                    'time': time,
                    'p': np.asarray(trial['pressure'], dtype=float).copy(),
                    'sw': np.asarray(trial['sW'], dtype=float).copy(),
                    'sg': np.asarray(trial['sG'], dtype=float).copy(),
                    'rs': np.asarray(trial['rs'], dtype=float).copy(),
                    'status': np.asarray(trial.get('status', []), dtype=int).copy(),
                    'bhp': np.asarray([w.get('bhp', np.nan) for w in wells], dtype=float),
                    'qgs': np.asarray([w.get('qGs', np.nan) for w in wells], dtype=float),
                    'residuals': np.asarray(values, dtype=float).copy(),
                    'converged': np.asarray(converged, dtype=bool).copy(),
                })
                break
        return converged, values, names

    model.checkConvergence = record_check
    solver.timeStepSelector.reset()
    previous_control = None
    for i, dt in enumerate(schedule['step']['val'], start=1):
        control_id = int(schedule['step']['control'][i - 1])
        forces = model.getDrivingForces(schedule['control'][control_id])
        if control_id != previous_control:
            model, state = model.updateForChangedControls(state, forces)
            previous_control = control_id
        old_state = deepcopy(state)
        state, report, _ = solver.solveTimestep(
            old_state, float(dt), model,
            drivingForces=forces, initialGuess=deepcopy(state), controlId=control_id,
        )
        if not report.get('Converged', False):
            raise RuntimeError(f'Python SPE1 did not converge at report step {i}')

    np.savez(
        OUTPUT,
        step=np.asarray([r['step'] for r in records], dtype=int),
        time=np.asarray([r['time'] for r in records], dtype=float),
        pressure=np.asarray([r['p'] for r in records], dtype=float),
        sw=np.asarray([r['sw'] for r in records], dtype=float),
        sg=np.asarray([r['sg'] for r in records], dtype=float),
        rs=np.asarray([r['rs'] for r in records], dtype=float),
        status=np.asarray([r['status'] for r in records], dtype=object),
        bhp=np.asarray([r['bhp'] for r in records], dtype=object),
        qgs=np.asarray([r['qgs'] for r in records], dtype=object),
        residuals=np.asarray([r['residuals'] for r in records], dtype=object),
        converged=np.asarray([r['converged'] for r in records], dtype=object),
    )
    print(f'Wrote {len(records)} Python nonlinear trace states to {OUTPUT}')
    for i, rec in enumerate(records, start=1):
        print(i, rec['step'], rec['residuals'], rec['converged'].astype(int))


if __name__ == '__main__':
    main()
