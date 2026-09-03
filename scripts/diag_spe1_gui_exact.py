"""Reproduce the GUI's exact SPE1 + AMGCL CPR configuration (errorOnFailure=True)."""
import sys
import time
import traceback

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = 'examples/SpE1/SPE1CASE1.DATA'

t0 = time.time()
state0, model, schedule, solver = init_eclipse_problem_ad(DECK,
                                                          RemoveZeroPoreVolume=True)
print('init: %.2f s, cells=%d, steps=%d' % (
    time.time() - t0, len(state0['pressure']), len(schedule['step']['val'])))
print('init solver type:', type(solver).__name__)
print('  errorOnFailure=%s maxIterations=%s maxTimestepCuts=%s minIterations=%s'
      % (getattr(solver, 'errorOnFailure', '?'),
         getattr(solver, 'maxIterations', '?'),
         getattr(solver, 'maxTimestepCuts', '?'),
         getattr(solver, 'minIterations', '?')))
print('  useLinesearch=%s enforceResidualDecrease=%s acceptanceFactor=%s'
      % (getattr(solver, 'useLinesearch', '?'),
         getattr(solver, 'enforceResidualDecrease', '?'),
         getattr(solver, 'acceptanceFactor', '?')))
print('  original LinearSolver:', type(getattr(solver, 'LinearSolver', None)).__name__)

# --- exactly what _SimulationWorker._apply_params does for AMGCL CPR ---
solver.useLinesearch = True
solver.enforceResidualDecrease = True
solver.acceptanceFactor = 2.0
solver.LinearSolver = AMGCL_CPRSolverBlockAD(
    tolerance=1e-4,
    maxIterations=50,
    strategy='mrst',
    decoupling='trueIMPES',
)
print('swapped LinearSolver:', type(solver.LinearSolver).__name__)

t0 = time.time()
try:
    res = simulate_schedule(model, state0, schedule, solver, max_steps=6)
    print('simulate OK in %.1f s; steps=%d' % (time.time() - t0, len(res.get('steps', []))))
    for i, st in enumerate(res.get('steps', [])):
        print('  step %d: conv=%s iters=%s' % (
            i + 1, st.get('converged'), st.get('report', {}).get('Iterations')))
except Exception as exc:
    print('EXCEPTION after %.1f s: %s: %s' % (time.time() - t0, type(exc).__name__, exc))
    traceback.print_exc()
