"""Profile one T142 report step to locate the performance bottleneck."""
import cProfile
import io
import pstats
import sys
import time
from copy import deepcopy

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

STEP_TO_PROFILE = 5  # a step with several wells already open

t0 = time.time()
s0, model, schedule, solver = init_eclipse_problem_ad(
    'examples/T142/T142_E100.DATA',
    RemoveZeroPoreVolume=True,
)
print(f'deck+init: {time.time()-t0:.1f}s', flush=True)
solver.errorOnFailure = False
solver.verbose = False

# advance to the step we want to profile
state = s0
for step_idx in range(STEP_TO_PROFILE):
    ctrl_idx = int(schedule['step']['control'][step_idx])
    forces = model.getDrivingForces(schedule['control'][ctrl_idx])
    model, state = model.updateForChangedControls(state, forces)
    dt = float(schedule['step']['val'][step_idx])
    state, report, _ = solver.solveTimestep(
        deepcopy(state), dt, model,
        drivingForces=forces, initialGuess=deepcopy(state), controlId=ctrl_idx,
    )
    print(f'warmup step {step_idx}: converged={report.get("Converged")} '
          f'iters={report.get("Iterations")} '
          f'SimTime={report.get("SimulationTime"):.1f}s', flush=True)

# profile the next step
ctrl_idx = int(schedule['step']['control'][STEP_TO_PROFILE])
forces = model.getDrivingForces(schedule['control'][ctrl_idx])
model, state = model.updateForChangedControls(state, forces)
dt = float(schedule['step']['val'][STEP_TO_PROFILE])

pr = cProfile.Profile()
pr.enable()
t1 = time.time()
state, report, _ = solver.solveTimestep(
    deepcopy(state), dt, model,
    drivingForces=forces, initialGuess=deepcopy(state), controlId=ctrl_idx,
)
wall = time.time() - t1
pr.disable()

print(f'\nprofiled step {STEP_TO_PROFILE}: wall={wall:.1f}s '
      f'converged={report.get("Converged")} iters={report.get("Iterations")} '
      f'solver_time={report.get("SimulationTime"):.1f}s', flush=True)

# aggregate linear solver time from ministeps
total_lin = 0.0
nlin = 0
for sr in report.get('StepReports', []):
    nlr = sr.get('NonlinearReport', [])
    if nlr:
        e = nlr[0]
        ls = e.get('LinearSolver', {}) if isinstance(e, dict) else {}
        if isinstance(ls, dict) and ls:
            total_lin += float(ls.get('SolverTime', 0.0))
            nlin += 1
print(f'linear solves: {nlin} total_lin_time={total_lin:.1f}s '
      f'({100*total_lin/wall:.0f}% of wall)', flush=True)

print('\n=== cProfile top 40 ===')
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())
