"""Probe: AMGCL block CPR on full SPE10 (Model 2, 1.12M cells), first step."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\spe10model2\SPE10_MODEL2.DATA"
t0 = time.time()
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
print("init: %.1f s, cells=%d, steps=%d" % (time.time() - t0, len(state0["pressure"]), len(schedule["step"]["val"])))

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                decoupling="trueIMPES")
nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=4,
                            linearSolver=linear, verbose=False)

import threading
done = threading.Event()
result = {}
t1 = time.time()


def run():
    try:
        res = simulate_schedule(model, state0, schedule, nonlinear, max_steps=1)
        result["steps"] = len(res["steps"])
        result["conv"] = res["steps"][0]["converged"] if res["steps"] else False
        result["iters"] = res["steps"][0]["report"].get("Iterations") if res["steps"] else None
        result["wall"] = res["steps"][0]["wall"] if res["steps"] else None
    except Exception as exc:
        result["err"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        done.set()


th = threading.Thread(target=run, daemon=True)
th.start()
if not done.wait(timeout=600):
    print("TIMEOUT after 600 s -- first SPE10 step did not finish")
else:
    print("first step in %.1f s: %s" % (time.time() - t1, result))
