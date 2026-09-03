"""Run complete SPE9 simulation with AMGCL CPR solver."""
import sys
import time
sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverAD

print('--- SPE9 init ---')
t0 = time.time()
s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
print(f'cells={len(s0["pressure"])} steps={len(schedule["step"]["val"])} gas={model.gas} time={time.time()-t0:.1f}s')

# Force AMGCL CPR solver
linear_solver = AMGCL_CPRSolverAD(tolerance=1e-3, maxIterations=100, verbose=True)
nonlinear_solver = NonLinearSolver(linearSolver=linear_solver, verbose=True, maxIterations=10)

print('--- SPE9 simulate ---')
try:
    t0 = time.time()
    wsol, states, report = simulate_schedule_ad(
        s0, model, schedule,
        NonLinearSolver=nonlinear_solver,
        return_report=True,
    )
    print(
        f'OK nstates={len(states)} '
        f'nsteps={report.get("NumControlSteps", len(states))} '
        f'converged={report.get("Converged", True)} '
        f'time={time.time()-t0:.1f}s'
    )
    import numpy as np
    p_vals = [float(s['pressure'][0]) for s in states if s is not None]
    print(f'  p_min={min(p_vals):.2e} p_max={max(p_vals):.2e}')
except Exception as e:
    print(f'FAIL {type(e).__name__}: {str(e)[:200]}')
    import traceback
    traceback.print_exc()
