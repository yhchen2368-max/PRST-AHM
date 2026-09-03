"""SPE1 + AMGCL CPR (GUI exact config, errorOnFailure=True), run 30 steps.

Finds the step at which the run would raise and stop the GUI.
"""
import sys
import time
import traceback

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECK = 'examples/SpE1/SPE1CASE1.DATA'
NSTEPS = 30

state0, model, schedule, solver = init_eclipse_problem_ad(DECK,
                                                          RemoveZeroPoreVolume=True)
solver.useLinesearch = True
solver.enforceResidualDecrease = True
solver.acceptanceFactor = 2.0
solver.LinearSolver = AMGCL_CPRSolverBlockAD(
    tolerance=1e-4, maxIterations=50, strategy='mrst', decoupling='trueIMPES')

t0 = time.time()
try:
    res = simulate_schedule(model, state0, schedule, solver, max_steps=NSTEPS)
    print('simulate OK in %.1f s; steps=%d' % (time.time() - t0, len(res.get('steps', []))))
    for i, st in enumerate(res.get('steps', [])):
        print('  step %d: conv=%s iters=%s' % (
            i + 1, st.get('converged'), st.get('report', {}).get('Iterations')))
except Exception as exc:
    print('EXCEPTION at t=%.1f s: %s: %s' % (time.time() - t0, type(exc).__name__, exc))
    traceback.print_exc()
