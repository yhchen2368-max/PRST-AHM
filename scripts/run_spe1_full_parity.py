"""Run all SPE1 report steps and compare every state against MRST 2026a.

The reference is produced only by ``export_mrst_spe1_full.m``.  This is a
diagnostic runner, not an alternate simulator: it stops on a non-converged
step and writes the last Python state plus per-step error metrics.
"""

from copy import deepcopy
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECK = ROOT / 'examples' / 'SPE1' / 'BENCH_SPE1.DATA'
REFERENCE = ROOT / 'spe1_mrst_full.mat'
CHECKPOINT = ROOT / 'spe1_python_full_parity_checkpoint.npz'
RESULT = ROOT / 'spe1_python_full_parity.npz'


def save_checkpoint(step, rows, state, finished=False, error=''):
    wellsol = state.get('wellSol', [])
    np.savez(
        CHECKPOINT if not finished else RESULT,
        step=np.asarray([step], dtype=int),
        rows=np.asarray(rows, dtype=float),
        pressure=np.asarray(state['pressure'], dtype=float),
        sw=np.asarray(state['sW'], dtype=float),
        sg=np.asarray(state['sG'], dtype=float),
        rs=np.asarray(state['rs'], dtype=float),
        status=np.asarray(state.get('status', []), dtype=int),
        bhp=np.asarray([w.get('bhp', np.nan) for w in wellsol], dtype=float),
        qws=np.asarray([w.get('qWs', np.nan) for w in wellsol], dtype=float),
        qos=np.asarray([w.get('qOs', np.nan) for w in wellsol], dtype=float),
        qgs=np.asarray([w.get('qGs', np.nan) for w in wellsol], dtype=float),
        finished=np.asarray([finished], dtype=bool),
        error=np.asarray([error]),
    )


def errors(state, reference, step):
    pairs = (
        ('pressure', state['pressure'], reference['pressure'][:, step]),
        ('sw', state['sW'], reference['sw'][:, step]),
        ('sg', state['sG'], reference['sg'][:, step]),
        ('rs', state['rs'], reference['rs'][:, step]),
    )
    values = []
    for _, actual, expected in pairs:
        delta = np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)
        values.extend((float(np.max(np.abs(delta))), float(np.linalg.norm(delta))))
    wells = state.get('wellSol', [])
    bhp = np.asarray([w.get('bhp', np.nan) for w in wells], dtype=float)
    qgs = np.asarray([w.get('qGs', np.nan) for w in wells], dtype=float)
    values.extend((float(np.max(np.abs(bhp - reference['bhp'][:, step]))),
                   float(np.max(np.abs(qgs - reference['qgs'][:, step])))))
    return values


def main():
    if not REFERENCE.is_file():
        raise FileNotFoundError(f'MRST reference not found: {REFERENCE}')
    state, model, schedule, solver = init_eclipse_problem_ad(str(DECK))
    reference = loadmat(REFERENCE, squeeze_me=True, struct_as_record=False)
    if len(schedule['step']['val']) != reference['pressure'].shape[1]:
        raise RuntimeError('Python and MRST SPE1 report-step counts differ')

    solver.timeStepSelector.reset()
    previous_control = None
    rows = []
    started = time.perf_counter()
    for step, dt in enumerate(schedule['step']['val']):
        control_id = int(schedule['step']['control'][step])
        forces = model.getDrivingForces(schedule['control'][control_id])
        if control_id != previous_control:
            model, state = model.updateForChangedControls(state, forces)
            previous_control = control_id
        old_state = deepcopy(state)
        try:
            state, report, _ = solver.solveTimestep(
                old_state, float(dt), model,
                drivingForces=forces, initialGuess=deepcopy(state),
                controlId=control_id,
            )
        except Exception as exc:
            save_checkpoint(step + 1, rows, state, error=repr(exc))
            raise
        if not report.get('Converged', False):
            message = f'Nonlinear solver did not converge at report step {step + 1}'
            save_checkpoint(step + 1, rows, state, error=message)
            raise RuntimeError(message)
        metric = errors(state, reference, step)
        row = [step + 1, *metric, int(report['Iterations']), int(report['AcceptedMinisteps'])]
        rows.append(row)
        save_checkpoint(step + 1, rows, state)
        print(
            'STEP={:03d} elapsed={:.2f}s iterations={} ministeps={} '
            'pmax={:.6e} swmax={:.6e} sgmax={:.6e} rsmax={:.6e} '
            'bhpmax={:.6e} qgsmax={:.6e}'.format(
                step + 1, time.perf_counter() - started,
                report['Iterations'], report['AcceptedMinisteps'],
                metric[0], metric[2], metric[4], metric[6], metric[8], metric[9],
            ),
            flush=True,
        )
    save_checkpoint(len(rows), rows, state, finished=True)
    matrix = np.asarray(rows, dtype=float)
    print('COMPLETE elapsed={:.2f}s'.format(time.perf_counter() - started), flush=True)
    print('GLOBAL_MAX=' + repr(matrix[:, 1:11].max(axis=0).tolist()), flush=True)
    print('ITERATIONS={} MINISTEPS={}'.format(int(matrix[:, -2].sum()), int(matrix[:, -1].sum())), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
