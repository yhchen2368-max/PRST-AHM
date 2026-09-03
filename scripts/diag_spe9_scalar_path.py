"""Confirm what the scalar AMGCL_CPRSolverAD actually executes on SPE9."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverAD

DECK = r"examples\SPE9\SPE9.DATA"
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)

linear = AMGCL_CPRSolverAD(tolerance=1e-4, maxIterations=50, verbose=False,
                           decoupling="trueIMPES", extraReport=True)

# capture one linear-solve report from the first Newton step
reports = []


def wrap(problem, model=None):
    dx, res, rep = orig(problem, model)
    reports.append(rep)
    return dx, res, rep


orig = linear.solveLinearProblem
linear.solveLinearProblem = wrap

nonlinear = NonLinearSolver(maxIterations=10, minIterations=1, maxTimestepCuts=2,
                            linearSolver=linear, verbose=False)
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule
try:
    simulate_schedule(model, state0, schedule, nonlinear, max_steps=1)
except Exception as exc:
    print("(step may not have converged in the allowed iterations: %s)" % exc)

print("first linear-solve report:")
if reports:
    r = reports[0]
    pre = r.get("PreconditionerReport", {})
    print("  PreconditionerReport.Type =", pre.get("Type"))
    print("  SolverTime = %.3f s" % r.get("SolverTime", 0))
    print("  Iterations =", r.get("Iterations"))
    print("  residual   = %.3e" % r.get("Residual", 0))
