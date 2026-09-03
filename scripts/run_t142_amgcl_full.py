"""Full T142 (388 steps) with the fixed AMGCL block CPR.

Writes the same well_rates.csv / timing.csv format as run_t142_full.py so the
results line up with the previous PETSc baseline (well_rates_baseline.csv /
timing_baseline.csv).
"""
import sys
import time
import csv
import os
from datetime import date, timedelta
sys.path.insert(0, ".")
import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECK = os.path.join(ROOT, "examples", "T142", "T142_E100.DATA")
OUT = os.path.join(ROOT, "results", "T142_full")
os.makedirs(OUT, exist_ok=True)
log_path = os.path.join(OUT, "full_run_amgcl_log.txt")
csv_path = os.path.join(OUT, "well_rates_amgcl.csv")
timing_path = os.path.join(OUT, "timing_amgcl.csv")
start = date(1999, 9, 1)

out = open(log_path, "w")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    out.write(line + "\n")
    out.flush()


t0 = time.time()
state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
log("init: %.1f s, cells=%d, steps=%d" % (time.time() - t0, len(state0["pressure"]), len(schedule["step"]["val"])))

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50, verbose=False,
                                decoupling="trueIMPES")
nonlinear = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=4,
                            linearSolver=linear, verbose=False)

nsteps = len(schedule["step"]["val"])
elapsed_days = 0.0
conv = 0
with open(csv_path, "w", newline="") as wcsv, open(timing_path, "w", newline="") as wtim:
    wr = csv.writer(wcsv)
    wr.writerow(["step", "time_days", "date", "dt_days", "converged",
                 "nonlinear_iters", "wall_s", "well", "status",
                 "qO_sm3d", "qW_sm3d", "qG_sm3d", "bhp_bar"])
    wt = csv.writer(wtim)
    wt.writerow(["step", "wall_s", "newton_iters", "assembly_s", "assembly_calls",
                 "linear_s", "linear_calls", "krylov_iters", "other_s",
                 "pc_setup_s", "pc_apply_s", "ilu_factor_s", "ilu_solve_s",
                 "matvec_s", "ksp_total_s"])

    def on_step(index, info):
        global conv
        rep = info.get("report", {})
        it = int(rep.get("Iterations", 0))
        if info.get("converged"):
            conv += 1
        wall = float(info.get("wall", 0))
        dt_days = info.get("dt", 0) / 86400.0
        when = (start + timedelta(days=info["time_days"])).isoformat()
        log("step %d/%d: conv=%s iters=%d wall=%.2f s"
            % (index + 1, nsteps, info.get("converged"), it, wall))
        # timing row (PETSc event columns left blank: not a PETSc run)
        wt.writerow([index + 1, "%.3f" % wall, it, "0", "0",
                     "0", "0", "0", "0",
                     "0", "0", "0", "0", "0", "0"])
        # well rates
        for well in info.get("wellSol", []):
            def value(key):
                raw = well.get(key)
                if raw is None:
                    return 0.0
                arr = np.atleast_1d(np.asarray(raw, dtype=float))
                return float(arr[0]) if arr.size else 0.0
            wr.writerow([index + 1, "%.3f" % info["time_days"], when, "%.3f" % dt_days,
                         int(bool(info.get("converged"))), it, "%.2f" % wall,
                         str(well.get("name", "?")), int(bool(well.get("status"))),
                         "%.6f" % (value("qOs") * 86400.0),
                         "%.6f" % (value("qWs") * 86400.0),
                         "%.6f" % (value("qGs") * 86400.0),
                         "%.6f" % (value("bhp") / 1e5)])

    t1 = time.time()
    res = simulate_schedule(model, state0, schedule, nonlinear,
                            on_step=on_step, max_steps=None)
    wall = time.time() - t1
log("=== done: %d steps in %.1f s (%.2f s/step), auto blockSize=%d ==="
    % (res["nsteps"], wall, wall / max(1, res["nsteps"]), linear.blockSize))
log("converged: %d/%d" % (conv, res["nsteps"]))
out.close()
