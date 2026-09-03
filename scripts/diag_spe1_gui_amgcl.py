"""Reproduce: SPE1 with AMGCL block CPR -- why does the GUI stop after one step?"""
import sys
import time

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = 'examples/SpE1/SPE1CASE1.DATA'

t0 = time.time()
state0, model, schedule, _ = init_eclipse_problem_ad(DECK)
print('init: %.2f s, cells=%d, steps=%d' % (
    time.time() - t0, len(state0['pressure']), len(schedule['step']['val'])))
print('phases water/oil/gas:', bool(model.water), bool(model.oil), bool(model.gas))

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                strategy='mrst', decoupling='trueIMPES')
lin_records = []
orig = linear.solveLinearProblem


def traced(problem, model_arg=None):
    dx, res, rep = orig(problem, model_arg)
    if isinstance(rep, dict):
        lin_records.append(dict(rep))
    return dx, res, rep


linear.solveLinearProblem = traced
nl = NonLinearSolver(maxIterations=15, minIterations=1, maxTimestepCuts=4,
                     linearSolver=linear, errorOnFailure=False, verbose=True)


def on_start(i, meta):
    print('REPORT STEP %d  dt=%.1f d' % (i + 1, meta['dt'] / 86400.0), flush=True)


def on_step(i, info):
    print('  step %d: conv=%s iters=%s wall=%.2fs' % (
        i + 1, info.get('converged'), info.get('iterations'),
        info.get('wall', 0.0)), flush=True)


t0 = time.time()
try:
    res = simulate_schedule(model, state0, schedule, nl, max_steps=8,
                            on_solve_start=on_start, on_step=on_step)
    print('simulate done in %.1f s; steps=%d'
          % (time.time() - t0, len(res.get('steps', []))), flush=True)
    for i, st in enumerate(res.get('steps', [])):
        print('  saved step %d: conv=%s iters=%s'
              % (i + 1, st.get('converged'), st.get('report', {}).get('Iterations')), flush=True)
except Exception as exc:
    print('EXCEPTION: %s: %s' % (type(exc).__name__, exc), flush=True)
    import traceback
    traceback.print_exc()

if lin_records:
    preps = [r.get('PreconditionerReport', {}) for r in lin_records]
    print('linear solves: %d, total iters=%d, total solver=%.2fs, '
          'total prep=%.2fs, kernel=%.2fs' % (
              len(lin_records),
              sum(int(r.get('Iterations', 0)) for r in lin_records),
              sum(float(r.get('SolverTime', 0)) for r in lin_records),
              sum(float(r.get('PreparationTime', 0)) for r in lin_records),
              sum(float(p.get('KernelTime', 0)) for p in preps)), flush=True)
    # first and last solve details
    for label, idx in (('first', 0), ('last', len(lin_records) - 1)):
        r = lin_records[idx]
        p = r.get('PreconditionerReport', {})
        print('%s solve: iters=%d res=%.3e fullres=%.3e prep=%.2fs '
              'kernel=%.2fs drs=%s mrst_w=%s trueimp=%s type=%s' % (
                  label, int(r.get('Iterations', 0)), float(r.get('Residual', 0)),
                  float(p.get('FullSystemResidual', 0)),
                  float(r.get('PreparationTime', 0)),
                  float(p.get('KernelTime', 0)),
                  p.get('AMGCLUseDRS'), p.get('MRSTRowWeighted'),
                  p.get('TrueIMPESWeighted'), p.get('Type')), flush=True)
