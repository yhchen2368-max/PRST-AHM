"""Full SPE9 (90 report steps) with the fixed block AMGCL CPR."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\SPE9\SPE9.DATA"
out = open(r"results\spe9_amgcl_block_full.log", "w")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.write(line + "\n")
    out.flush()


t0 = time.time()
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
log("init: %.1f s, cells=%d, steps=%d" % (time.time() - t0, len(state0["pressure"]), len(schedule["step"]["val"])))

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                decoupling="trueIMPES")
nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=4,
                            linearSolver=linear, verbose=False)
log("auto blockSize = %d" % linear.blockSize)

lin_times = []
conv = 0


def on_step(index, info):
    rep = info.get("report", {})
    lin = rep.get("LinearSolver", {})
    if isinstance(lin, dict):
        lin_times.append(lin.get("SolverTime", 0))
    if info.get("converged"):
        global conv
        conv += 1
    log("  step %d/%d: conv=%s iters=%s wall=%.2f s"
        % (index + 1, len(schedule["step"]["val"]), info.get("converged"),
           rep.get("Iterations"), info.get("wall")))


t1 = time.time()
res = simulate_schedule(model, state0, schedule, nonlinear,
                        on_step=on_step, max_steps=None)
wall = time.time() - t1
log("=== done: %d steps in %.1f s (%.2f s/step) ===" % (res["nsteps"], wall, wall / max(1, res["nsteps"])))
log("converged steps: %d/%d" % (conv, res["nsteps"]))
if lin_times:
    lt = np.asarray(lin_times, dtype=float)
    lt = lt[np.isfinite(lt)]
    log("linear solve times: n=%d mean=%.3f s max=%.3f s" % (len(lt), lt.mean(), lt.max()))
out.close()
