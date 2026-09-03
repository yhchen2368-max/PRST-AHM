"""Headless core-only repro: load SPE1, run AMGCL CPR simulate_schedule for
up to 20 steps with NO Qt/GUI involved.  If the process dies after ~1-3 steps
with no Python traceback, the crash is in the numerical core (scipy/numpy/BLAS
mix), not in the GUI.  Writes a log line per step to stdout (flushed)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import PRSTCore  # noqa: F401  -- DLL path bootstrap

print("python:", sys.version.split()[0], flush=True)
print("numpy :", np.__version__, "from", os.path.dirname(np.__file__), flush=True)
import scipy
print("scipy :", scipy.__version__, "from", os.path.dirname(scipy.__file__), flush=True)

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
    init_eclipse_problem_ad)
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "SpE1", "SPE1CASE1.DATA")

print("loading deck ...", flush=True)
state0, model, schedule, solver = init_eclipse_problem_ad(
    DECK, RemoveZeroPoreVolume=True)
print("deck loaded, steps =", len(schedule["step"]["val"]), flush=True)

from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD
solver.LinearSolver = AMGCL_CPRSolverBlockAD(
    tolerance=1e-4, maxIterations=50,
    strategy="mrst", decoupling="trueIMPES")

# Match the GUI defaults exactly (the checkboxes are on by default).
solver.useLinesearch = True
solver.enforceResidualDecrease = True
solver.acceptanceFactor = 2.0
print("useLinesearch=%s enforceResidualDecrease=%s acceptanceFactor=%s"
      % (solver.useLinesearch, solver.enforceResidualDecrease,
         solver.acceptanceFactor), flush=True)

print("=== running core simulate_schedule (max 20 steps) ===", flush=True)
t0 = time.time()


def on_solve_start(index, meta):
    print("REPORT STEP %d  TIME=%.1f days  DT=%.1f d"
          % (index + 1, meta["time_days"], meta["dt"] / 86400.0), flush=True)


def on_step(index, info):
    print("   %s in %d iterations, wall=%.2f s"
          % ("converged" if info["converged"] else "FAILED",
             info["iterations"], info["wall"]), flush=True)


try:
    result = simulate_schedule(
        model, state0, schedule, solver, max_steps=20,
        on_solve_start=on_solve_start, on_step=on_step)
    print("=== DONE: nsteps=%d wall=%.1f s ==="
          % (result["nsteps"], result["wall"]), flush=True)
except Exception as exc:
    import traceback
    print("=== EXCEPTION ===", flush=True)
    traceback.print_exc()
    sys.exit(2)

print("CORE-RUN-OK", flush=True)
