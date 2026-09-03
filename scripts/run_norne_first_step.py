"""Run Norne's first report step with the deck-selected solver."""
from copy import deepcopy
import time
import sys

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

PATH = 'examples/Norne/Norne_simplified/NORNE_ATW2013.DATA'

state0, model, schedule, solver = init_eclipse_problem_ad(PATH)
solver.verbose = True
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
dt = float(schedule['step']['val'][0])
print(f'cells={len(state0["pressure"])} dt={dt:g} wells={len(forces.get("W", []))}', flush=True)
start = time.perf_counter()
state, report, _ = solver.solveTimestep(
    deepcopy(state0), dt, model, drivingForces=forces,
    initialGuess=deepcopy(state0), controlId=control_id,
)
print(
    f'converged={report["Converged"]} iterations={report["Iterations"]} '
    f'ministeps={report["AcceptedMinisteps"]} elapsed={time.perf_counter() - start:.1f}s',
    flush=True,
)
print(f'p_range=[{state["pressure"].min():.9g}, {state["pressure"].max():.9g}]', flush=True)
for step in report['StepReports']:
    print(f'dt={step["Timestep"]:.9g} iterations={step["Iterations"]} converged={step["Converged"]}', flush=True)
