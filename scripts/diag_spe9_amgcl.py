"""Diagnose AMGCL CPR on SPE9: scalar vs block, first report step, timed."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverAD, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = r"examples\SPE9\SPE9.DATA"


def run_first_step(solver_name, make_linear, max_steps=1):
    print("=" * 70)
    print("SPE9 first step with %s" % solver_name)
    t0 = time.time()
    state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
    print("  deck init: %.1f s, cells=%d, wells=%d"
          % (time.time() - t0, len(state0["pressure"]), len(schedule["control"][0]["W"])))

    linear = make_linear()
    nonlinear = NonLinearSolver(
        maxIterations=10, minIterations=1, maxTimestepCuts=4,
        linearSolver=linear, verbose=False, useRelaxation=True)

    times = {"linear": [], "assembly": []}
    t_wall = time.time()

    def on_step(index, info):
        rep = info.get("report", {})
        lin = rep.get("LinearSolver", {})
        if isinstance(lin, dict):
            times["linear"].append(lin.get("SolverTime", None))
        times["assembly"].append(rep.get("AssemblyTime", None))

    try:
        result = simulate_schedule(
            model, state0, schedule, nonlinear, max_steps=max_steps,
            start=None, on_step=on_step)
    except Exception as exc:
        print("  FAILED: %s: %s" % (type(exc).__name__, exc))
        return None

    wall = time.time() - t_wall
    steps = result.get("steps", [])
    print("  steps done: %d  wall=%.1f s" % (len(steps), wall))
    for i, s in enumerate(steps):
        rep = s.get("report", {})
        iters = rep.get("Iterations", "?")
        conv = s.get("converged")
        print("   step %d: converged=%s iters=%s dt=%.1f d"
              % (i + 1, conv, iters, s.get("dt", 0) / 86400.0))
        lin = rep.get("LinearSolver", {})
        if isinstance(lin, dict):
            for k in ("SolverTime", "LinearSolutionTime", "PreparationTime",
                      "PostProcessTime", "Iterations"):
                if k in lin:
                    print("     linear: %s=%s" % (k, lin[k]))
    print("  linear solve times: %s" % [round(t, 2) if t else t for t in times["linear"]])
    return steps


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "scalar"
    if which == "scalar":
        run_first_step("AMGCL_CPRSolverAD (scalar CPR)",
                       lambda: AMGCL_CPRSolverAD(tolerance=1e-4, maxIterations=50,
                                                 verbose=True, decoupling="trueIMPES"))
    elif which == "block":
        run_first_step("AMGCL_CPRSolverBlockAD (block CPR)",
                       lambda: AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50,
                                                      verbose=True, decoupling="trueIMPES"))
    else:
        print("usage: diag_spe9_amgcl.py [scalar|block]")
