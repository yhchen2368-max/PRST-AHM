"""Full first SPE9 step with fixed auto-blockSize AMGCL block CPR."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\SPE9\SPE9.DATA"
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
print("cells=%d nunknowns=3*nc+wells" % len(state0["pressure"]))

# defaults now: blockSize=0 -> auto, schurApproxType='full'
linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                decoupling="trueIMPES")
nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=4,
                            linearSolver=linear, verbose=False)

t0 = time.time()
try:
    res = simulate_schedule(model, state0, schedule, nonlinear, max_steps=1)
    wall = time.time() - t0
    steps = res["steps"]
    s = steps[0]
    rep = s["report"]
    print("step1: converged=%s iters=%s dt=%.1f d  wall=%.1f s"
          % (s["converged"], rep.get("Iterations"), s["dt"] / 86400.0, wall))
    print("auto blockSize ->", linear.blockSize)
    pre = (rep.get("LinearSolver") or {}).get("PreconditionerReport", {})
    print("preconditioner:", pre.get("Type"))
    for k in ("KernelTime", "BlockConversionTime", "ReducedSystemResidual",
              "FullSystemResidual"):
        if k in pre:
            print("  %s = %s" % (k, pre[k]))
except Exception as exc:
    print("FAILED: %s: %s" % (type(exc).__name__, exc))
