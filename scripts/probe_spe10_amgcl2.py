"""SPE10 full: AMGCL block CPR first step with deep cuts + Newton trace."""
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
print("init: %.1f s, cells=%d, steps=%d, dt1=%.1f d"
      % (time.time() - t0, len(state0["pressure"]), len(schedule["step"]["val"]),
         schedule["step"]["val"][0] / 86400.0))

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                decoupling="trueIMPES")
nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=16,
                            linearSolver=linear, verbose=True)
print("blockSize auto:", linear.blockSize)

t1 = time.time()
try:
    res = simulate_schedule(model, state0, schedule, nonlinear, max_steps=1)
    s = res["steps"][0]
    print("step1: conv=%s iters=%s dt=%.2f d wall=%.1f s"
          % (s["converged"], s["report"].get("Iterations"), s["dt"] / 86400.0,
             s["wall"]))
except Exception as exc:
    print("FAILED after %.1f s: %s" % (time.time() - t1, exc))
