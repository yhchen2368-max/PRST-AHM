"""Bounded: does the block AMGCL CPR kernel solve SPE9's first system?"""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD

DECK = r"examples\SPE9\SPE9.DATA"
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
print("cells=%d" % len(state0["pressure"]))

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=True,
                                decoupling="trueIMPES", blockSize=3)
nonlinear = NonLinearSolver(maxIterations=3, minIterations=1, maxTimestepCuts=2,
                            linearSolver=linear, verbose=False)

import threading
done = threading.Event()
result = {}
t0 = time.time()


def run():
    try:
        from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule
        res = simulate_schedule(model, state0, schedule, nonlinear, max_steps=1)
        result["steps"] = len(res["steps"])
        result["ok"] = True
    except Exception as exc:
        result["ok"] = False
        result["err"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        done.set()


th = threading.Thread(target=run, daemon=True)
th.start()
ok = done.wait(timeout=90)
if not ok:
    print("TIMEOUT after 90s -- block AMGCL CPR kernel did not finish the first linear solve")
else:
    print("finished in %.1f s: %s" % (time.time() - t0, result))
