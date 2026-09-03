"""SPE10 Model 2 (1.12M cells) first-step comparison: mrst_drs vs amgcl_drs.

Both use the MRST 1:1 decoupling default (trueIMPES) and the exact Schur
('full'); each configuration runs in a guarded thread so a hang is reported
instead of blocking forever.
"""
import sys
import threading
import time

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\spe10model2\SPE10_MODEL2.DATA"
TIMEOUT = 900.0

CONFIGS = [
    ("mrst_drs", "trueIMPES"),
    ("amgcl_drs", "trueIMPES"),
]


def run_config(strategy, decoupling):
    linear = AMGCL_CPRSolverBlockAD(
        tolerance=1e-4,
        maxIterations=50,
        verbose=False,
        strategy=strategy,
        decoupling=decoupling,
        schurApproxType='full',
    )
    lin_records = []
    orig = linear.solveLinearProblem

    def traced(problem, model_arg=None):
        dx, res, rep = orig(problem, model_arg)
        if isinstance(rep, dict):
            lin_records.append(dict(rep))
        return dx, res, rep

    linear.solveLinearProblem = traced
    nonlinear = NonLinearSolver(maxIterations=12, minIterations=1,
                                maxTimestepCuts=4, linearSolver=linear,
                                verbose=False)
    done = threading.Event()
    result = {}
    t1 = time.time()

    def run():
        try:
            res = simulate_schedule(model, state0, schedule, nonlinear,
                                    max_steps=1)
            step = res["steps"][0] if res.get("steps") else {}
            result["conv"] = bool(step.get("converged", False))
            result["iters"] = step.get("report", {}).get("Iterations") \
                if isinstance(step.get("report"), dict) else None
            result["wall"] = float(step.get("wall", time.time() - t1))
        except Exception as exc:
            result["err"] = "%s: %s" % (type(exc).__name__, exc)
        finally:
            done.set()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    ok = done.wait(timeout=TIMEOUT)
    wall = time.time() - t1
    result["timed_out"] = not ok

    if lin_records:
        preps = [r.get('PreconditionerReport', {}) for r in lin_records]
        result["lin_count"] = len(lin_records)
        result["lin_iters_total"] = int(sum(
            float(r.get('Iterations', 0)) for r in lin_records))
        result["lin_solver_total"] = float(sum(
            float(r.get('SolverTime', 0)) for r in lin_records))
        result["kernel_total"] = float(sum(
            float(p.get('KernelTime', 0)) for p in preps))
        last = preps[-1] if preps else {}
        result["use_drs"] = last.get('AMGCLUseDRS')
        result["mrst_w"] = last.get('MRSTRowWeighted')
        result["trueimp"] = last.get('TrueIMPESWeighted')
        result["zero_w_rows"] = last.get('MRSTRowWeightZeroCount')
        result["prep_avg"] = float(np.mean([
            float(r.get('PreparationTime', 0)) for r in lin_records]))
    return result, wall


def main():
    print('SPE10 MODEL2 first step: mrst_drs vs amgcl_drs (trueIMPES, full Schur)',
          flush=True)
    t0 = time.time()
    global state0, model, schedule
    state0, model, schedule, _ = init_eclipse_problem_ad(DECK,
                                                         RemoveZeroPoreVolume=True)
    print('init: %.1f s, cells=%d, steps=%d'
          % (time.time() - t0, len(state0["pressure"]),
             len(schedule["step"]["val"])), flush=True)
    print('phases: water=%s oil=%s gas=%s'
          % (bool(getattr(model, 'water', False)),
             bool(getattr(model, 'oil', False)),
             bool(getattr(model, 'gas', False))), flush=True)

    for strategy, decoupling in CONFIGS:
        result, wall = run_config(strategy, decoupling)
        print(f'--- {strategy} + {decoupling} ---', flush=True)
        if result.get("timed_out"):
            print(f'  TIMEOUT after {TIMEOUT:.0f}s (no first-step result)',
                  flush=True)
            continue
        if "err" in result:
            print(f'  ERROR: {result["err"]}', flush=True)
            continue
        print(f'  conv={result.get("conv")} newton_iters={result.get("iters")} '
              f'wall={result.get("wall"):.1f}s', flush=True)
        print(f'  lin count={result.get("lin_count")} '
              f'iters_total={result.get("lin_iters_total")} '
              f'solver_total={result.get("lin_solver_total"):.1f}s '
              f'kernel_total={result.get("kernel_total"):.1f}s '
              f'prep_avg={result.get("prep_avg"):.2f}s', flush=True)
        print(f'  flags: use_drs={result.get("use_drs")} '
              f'mrst_w={result.get("mrst_w")} '
              f'trueimp={result.get("trueimp")} '
              f'zero_w_rows={result.get("zero_w_rows")}', flush=True)


if __name__ == '__main__':
    main()
