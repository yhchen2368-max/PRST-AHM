"""Trace Newton + block CPR residual quality on SPE9 first step."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\SPE9\SPE9.DATA"
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                decoupling="trueIMPES")
reports = []
orig = linear.solveLinearProblem


def wrap(problem, model=None):
    dx, res, rep = orig(problem, model)
    reports.append(rep)
    return dx, res, rep


linear.solveLinearProblem = wrap

nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=4,
                            linearSolver=linear, verbose=True)
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule as ss
try:
    ss(model, state0, schedule, nonlinear, max_steps=1)
    print("CONVERGED")
except Exception as exc:
    print("FAILED: %s" % exc)

print("\nlinear-solve residual quality (first 6):")
for i, r in enumerate(reports[:6]):
    pre = r.get("PreconditionerReport", {})
    print("  solve %d: iters=%d  red=%.2e  full=%.2e  conv=%s"
          % (i, r.get("Iterations"), pre.get("ReducedSystemResidual"),
             pre.get("FullSystemResidual"), r.get("Converged")))
