"""SPE10 2-phase first Newton system: is the AMGCL block-CPR solution accurate?

Solve the identical system at tol=1e-4 and tol=1e-10 and compare the
solutions.  If they differ by orders of magnitude, the relative-residual
convergence is misleading on this 11-orders-of-magnitude-contrast system and
the Newton direction is garbage despite a "converged" linear solve.
"""
import sys
import time

sys.path.insert(0, '.')

import numpy as np
import scipy.sparse as sp

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD

DECK = r"examples\spe10model2\SPE10_MODEL2_2P.DATA"
DT = 86400.0  # 1 day (same first report step as the earlier diagnostic)

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)
model, state0 = model.prepareReportstep(state0, state0, 1.0, forces)
nc = len(state0['pressure'])
problem, _ = model.get_equations(state0, state0, DT, forces)
A = problem['Jacobian'].tocsr().astype(float)
b = -np.asarray(problem['Residuals'], dtype=float).ravel()
n = A.shape[0]
nb = np.linalg.norm(b)
print('system n=%d nnz=%.2fM ||b||=%.3e' % (n, A.nnz / 1e6, nb), flush=True)

solutions = {}
for tol in (1e-4, 1e-10):
    linear = AMGCL_CPRSolverBlockAD(tolerance=tol, maxIterations=1000, verbose=False,
                                    strategy='mrst', decoupling='trueIMPES',
                                    schurApproxType='full')
    t0 = time.time()
    try:
        dx, res, rep = linear.solveLinearProblem(problem, model)
        wall = time.time() - t0
        dx = np.asarray(dx, dtype=float).ravel()
        pre = rep.get('PreconditionerReport', {}) if isinstance(rep, dict) else {}
        # absolute / relative full residual of the ORIGINAL system
        rel = float(np.linalg.norm(A.dot(dx) - b) / max(nb, 1e-300))
        # solution against the residual: how well does dx solve A dx = b
        sol = float(np.linalg.norm(A.dot(dx) - b))
        print('tol=%.0e: iters=%d kernel=%.1fs wall=%.1fs rel|A dx-b|/|b|=%.2e '
              'abs|A dx-b|=%.2e | max|dp|=%.2e |dsW|=%.2e'
              % (tol, int(rep.get('Iterations', 0)) if isinstance(rep, dict) else -1,
                 float(pre.get('KernelTime', 0)), wall, rel, sol,
                 np.max(np.abs(dx[:nc])), np.max(np.abs(dx[nc:2 * nc]))), flush=True)
        solutions[tol] = dx
    except Exception as exc:
        print('tol=%.0e ERROR %s: %s' % (tol, type(exc).__name__, exc), flush=True)

if 1e-4 in solutions and 1e-10 in solutions:
    d = solutions[1e-4] - solutions[1e-10]
    print('max|dx(1e-4) - dx(1e-10)| = %.3e' % np.max(np.abs(d)), flush=True)
    print('|dx(1e-10)| max = %.3e (pressure) / %.3e (sW)'
          % (np.max(np.abs(solutions[1e-10][:nc])),
             np.max(np.abs(solutions[1e-10][nc:2 * nc]))), flush=True)
