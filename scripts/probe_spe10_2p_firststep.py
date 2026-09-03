"""SPE10 Model 2 (2-phase deck) first report step with mrst_drs/mrst + trueIMPES.

The 3-phase deck declares GAS for the 3-phase 'Flo' simulator but has no gas
PVT, making the cell blocks near-singular (Newton diverged, |dp|~1e10).  The
2-phase deck fixes the conditioning; here we run the whole first report step.
"""
import sys
import threading
import time

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\spe10model2\SPE10_MODEL2_2P.DATA"
TIMEOUT = 1500.0

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
print('init ok: cells=%d steps=%d phases=(w=%s o=%s g=%s)'
      % (len(state0['pressure']), len(schedule['step']['val']),
         bool(model.water), bool(model.oil), bool(model.gas)), flush=True)
print('first report dt = %.6g d' % (schedule['step']['val'][0] / 86400.0), flush=True)

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=100, verbose=False,
                                strategy='mrst', decoupling='trueIMPES',
                                schurApproxType='full')
lin = {'n': 0, 'iters': 0, 't': 0.0, 'kernel': 0.0}
orig = linear.solveLinearProblem


def traced(problem, model_arg=None):
    t0 = time.perf_counter()
    dx, res, rep = orig(problem, model_arg)
    dt = time.perf_counter() - t0
    lin['n'] += 1
    if isinstance(rep, dict):
        lin['iters'] += int(rep.get('Iterations', 0))
        pre = rep.get('PreconditionerReport', {})
        if isinstance(pre, dict):
            lin['kernel'] += float(pre.get('KernelTime', 0))
    lin['t'] += dt
    return dx, res, rep


linear.solveLinearProblem = traced

nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=16,
                            linearSolver=linear, errorOnFailure=False, verbose=True)


def on_start(i, meta):
    print('--- REPORT STEP %d dt=%.6g d ---' % (i + 1, meta['dt'] / 86400.0), flush=True)


def on_step(i, info):
    print('    step %d: conv=%s iters=%s wall=%.1fs'
          % (i + 1, info.get('converged'), info.get('iterations'),
             info.get('wall', 0.0)), flush=True)


t0 = time.time()
done = threading.Event()
result = {}


def run():
    try:
        res = simulate_schedule(model, state0, schedule, nonlinear, max_steps=1,
                                on_solve_start=on_start, on_step=on_step)
        steps = res.get('steps', [])
        result['nsteps'] = len(steps)
        result['conv'] = bool(steps[0].get('converged')) if steps else None
        result['wall'] = float(steps[0].get('wall', 0.0)) if steps else None
    except Exception as exc:
        result['err'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        done.set()


th = threading.Thread(target=run, daemon=True)
th.start()
ok = done.wait(timeout=TIMEOUT)
print('=== wall=%.1fs timed_out=%s result=%s lin_solves=%d lin_iters=%d '
      'lin_time=%.1fs kernel=%.1fs ==='
      % (time.time() - t0, not ok, result, lin['n'], lin['iters'], lin['t'],
         lin['kernel']), flush=True)
