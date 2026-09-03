"""Test SPE9 simulation end-to-end."""
import time, sys, traceback
sys.path.insert(0, '.')
sys.setrecursionlimit(10000)

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import AMGCL_CPRSolverAD
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad

print('--- SPE9 init ---')
t0 = time.time()
s, m, sch, nl = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
nl.LinearSolver = AMGCL_CPRSolverAD(
    tolerance=1e-4,
    maxIterations=50,
    verbose=False,
    extraReport=True,
)
nl.verbose = True
print(f'cells={len(s["pressure"])} steps={len(sch["step"]["val"])} gas={m.gas} time={time.time()-t0:.1f}s')
print(
    f'nls={type(nl).__name__} linear={type(nl.LinearSolver).__name__} '
    f'tol={nl.LinearSolver.tolerance:g} maxit={nl.LinearSolver.maxIterations}'
)

ops = getattr(m, 'operators', None)
if ops:
    print(f'operators N={ops.get("N",[]).shape} T={ops.get("T",[]).shape}')
else:
    print('operators=None')

print('--- SPE9 simulate ---')
try:
    t0 = time.time()
    wsol, states, report = simulate_schedule_ad(
        s, m, sch, NonLinearSolver=nl, return_report=True
    )
    elapsed = time.time() - t0
    print(
        f'OK nstates={len(states)} time={elapsed:.1f}s '
        f'converged={report.get("Converged")}'
    )
    valid_states = [x for x in states if x is not None]
    if valid_states:
        pfirst = valid_states[0]['pressure'][0]
        plast = valid_states[-1]['pressure'][0]
        print(f'  p[0] first={pfirst:.2e} last={plast:.2e}')
except Exception as e:
    traceback.print_exc()
    print(f'FAIL {type(e).__name__}: {e}')
