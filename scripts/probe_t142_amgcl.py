"""Probe: AMGCL block CPR on T142, first few steps, timed."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\T142\T142_E100.DATA"
nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 3

t0 = time.time()
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
print("init: %.1f s, cells=%d, steps=%d" % (time.time() - t0, len(state0["pressure"]), len(schedule["step"]["val"])))

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                decoupling="trueIMPES")
print("blockSize (auto, before solve):", linear.blockSize)
nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=4,
                            linearSolver=linear, verbose=False)

t1 = time.time()
res = simulate_schedule(model, state0, schedule, nonlinear, max_steps=nsteps)
wall = time.time() - t1
print("== done %d steps in %.1f s (%.1f s/step), auto blockSize=%d =="
      % (len(res["steps"]), wall, wall / max(1, len(res["steps"])), linear.blockSize))
for i, s in enumerate(res["steps"]):
    rep = s["report"]
    print("  step %d: conv=%s iters=%s wall=%.2f s dt=%.1f d"
          % (i + 1, s["converged"], rep.get("Iterations"), s["wall"], s["dt"] / 86400.0))
