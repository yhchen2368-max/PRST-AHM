"""One-step diagnostic for spe3: which solver, where does it hang."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import (
    init_eclipse_problem_ad)

deck = os.path.join(ROOT, "examples", "spe3", "SPE3CASE1.DATA")
t0 = time.time()
state0, model, schedule, solver = init_eclipse_problem_ad(
    deck, RemoveZeroPoreVolume=True)
print("load %.1fs  nc=%d  phases(o/w/g)=%s/%s/%s"
      % (time.time() - t0, len(state0["pressure"]), model.oil, model.water,
         model.gas), flush=True)
lin = getattr(solver, "LinearSolver", None)
print("nonlinear solver: %s (maxIts=%s, cuts=%s)"
      % (type(solver).__name__, solver.maxIterations, solver.maxTimestepCuts),
      flush=True)
print("linear solver: %s" % type(lin).__name__, flush=True)
for attr in ("pressure_precond", "second_stage", "strategy", "tolerance",
             "solver"):
    print("   %s = %r" % (attr, getattr(lin, attr, "<n/a>")), flush=True)

# one report step with a watchdog: run in a thread, abort after 120 s
import threading

result = {"done": False, "report": None, "exc": None}

def work():
    try:
        control = int(schedule["step"]["control"][0])
        forces = model.getDrivingForces(schedule["control"][control])
        m2, st = model.updateForChangedControls(state0, forces)
        dt = float(schedule["step"]["val"][0])
        st, rep, _ = solver.solveTimestep(
            st, dt, m2, drivingForces=forces,
            initialGuess=st, controlId=control)
        result["done"] = True
        result["report"] = rep
    except Exception as exc:
        result["exc"] = "%s: %s" % (type(exc).__name__, exc)

th = threading.Thread(target=work, daemon=True)
th.start()
th.join(timeout=120)
if th.is_alive():
    print("HANG: solveTimestep did not return in 120 s", flush=True)
else:
    print("returned; done=%s exc=%s" % (result["done"], result["exc"]),
          flush=True)
    if result["report"]:
        rep = result["report"]
        print("Converged=%s Iterations=%s Cutting=%s StepSize=%s"
              % (rep.get("Converged"), rep.get("Iterations"),
                 rep.get("MinistepCuttingCount"), rep.get("StepSize")),
              flush=True)
