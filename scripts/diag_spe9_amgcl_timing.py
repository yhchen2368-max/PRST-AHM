"""Time each linear-solve phase inside the scalar AMGCL CPR solver on SPE9."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverAD
from PRSTCore.ad_core.solvers.linear_solver_ad import LinearSolverAD

DECK = r"examples\SPE9\SPE9.DATA"

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
print("cells=%d" % len(state0["pressure"]))

linear = AMGCL_CPRSolverAD(tolerance=1e-4, maxIterations=50, verbose=False,
                           decoupling="trueIMPES")

# --- wrap every linear-solve entry point with a timer --------------------
timings = []
_orig_solve = linear.solveLinearProblem
_orig_ell = linear.ellipticSolver.solveLinearProblem


def timed(method, label):
    def wrapper(problem, model=None):
        t0 = time.perf_counter()
        out = method(problem, model)
        dt = time.perf_counter() - t0
        timings.append((label, dt, getattr(problem, "iterationNo", None)))
        return out
    return wrapper

linear.solveLinearProblem = timed(_orig_solve, "CPR total")
linear.ellipticSolver.solveLinearProblem = timed(_orig_ell, "  elliptic(AMGCL)")

nonlinear = NonLinearSolver(maxIterations=10, minIterations=1, maxTimestepCuts=4,
                            linearSolver=linear, verbose=False)

from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule
res = simulate_schedule(model, state0, schedule, nonlinear, max_steps=1)
steps = res["steps"]
print("step1 converged=%s iters=%s" % (steps[0]["converged"], steps[0]["report"].get("Iterations")))

# aggregate
from collections import defaultdict
agg = defaultdict(lambda: [0.0, 0])
for label, dt, _ in timings:
    agg[label][0] += dt
    agg[label][1] += 1
print("\n--- linear-solve timings over the report step ---")
total = 0.0
for label, (dt, n) in agg.items():
    print("%-22s n=%-3d total=%.2f s  avg=%.3f s" % (label, n, dt, dt / max(n, 1)))
    total += dt
print("summed: %.2f s" % total)
