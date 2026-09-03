"""SPE10 Model 2: isolate the FIRST Newton linear solve across configs.

The full Newton loop oscillates and never converges; here we solve just the
first Newton system A x = b once per configuration and report the solve's
*reduced* vs *full* residual plus the update magnitudes, so we can see whether
the linear solve itself is accurate (or whether the Schur/scaling loses
accuracy on this 3-phase-declared-but-gasless 1.12M-cell system).
"""
import sys
import time

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD

DECK = r"examples\spe10model2\SPE10_MODEL2.DATA"

CONFIGS = [
    ('mrst', 'trueIMPES'),
    ('mrst', 'quasiIMPES'),
    ('amgcl', 'none'),
    ('amgcl', 'trueIMPES'),
    ('amgcl_drs', 'trueIMPES'),
]

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)
dt = float(schedule['step']['val'][0])
nc = len(state0['pressure'])
print('cells=%d dt=%.6g d phases(w/o/g)=%s/%s/%s'
      % (nc, dt / 86400.0, bool(model.water), bool(model.oil), bool(model.gas)),
      flush=True)

# First Newton system: Jacobian + residual at the initial guess.  RESV wells
# need the report-step density factors before the residual is assembled.
model, state0 = model.prepareReportstep(state0, state0, dt, forces)
problem, _ = model.get_equations(state0, state0, dt, forces)
print('problem keys:', sorted(problem.keys()), flush=True)
A0 = problem['Jacobian']
n = A0.shape[0]
print('system size n=%d (3*nc=%d, wells=%d) nnz=%.2fM'
      % (n, 3 * nc, n - 3 * nc, A0.nnz / 1e6), flush=True)
gas_rows = A0[2 * nc:3 * nc, :].tocsr()
gas_max = float(np.max(np.abs(gas_rows.data))) if gas_rows.nnz else 0.0
print('gas rows max |A| (first nc rows): %.3e  (near-zero => degenerate gas phase)'
      % gas_max, flush=True)

for strategy, decoupling in CONFIGS:
    linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=100, verbose=False,
                                    strategy=strategy, decoupling=decoupling,
                                    schurApproxType='full')
    t0 = time.time()
    try:
        dx, res, rep = linear.solveLinearProblem(problem, model)
        wall = time.time() - t0
        pre = rep.get('PreconditionerReport', {}) if isinstance(rep, dict) else {}
        if not isinstance(dx, np.ndarray) or dx.size == 0:
            print('%-16s %-10s FAILED (empty solution)' % (strategy, decoupling), flush=True)
            continue
        dx = np.asarray(dx, dtype=float).ravel()
        gp = np.max(np.abs(dx[:nc])) if dx.size >= nc else float('nan')
        gs = np.max(np.abs(dx[nc:2 * nc])) if dx.size >= 2 * nc else float('nan')
        gg = np.max(np.abs(dx[2 * nc:3 * nc])) if dx.size >= 3 * nc else float('nan')
        gw = np.max(np.abs(dx[3 * nc:])) if dx.size > 3 * nc else 0.0
        print('%-16s %-10s iters=%d red=%.2e full=%.2e kernel=%.2fs prep=%.2fs '
              'wall=%.1fs | max|dp|=%.2e |dsW|=%.2e |dsG|=%.2e |dw|=%.2e'
              % (strategy, decoupling,
                 int(rep.get('Iterations', 0)) if isinstance(rep, dict) else -1,
                 float(pre.get('ReducedSystemResidual', float('nan'))),
                 float(pre.get('FullSystemResidual', float('nan'))),
                 float(pre.get('KernelTime', 0)), float(pre.get('PreparationTime', 0)),
                 wall, gp, gs, gg, gw), flush=True)
    except Exception as exc:
        print('%-16s %-10s ERROR %s: %s' % (strategy, decoupling, type(exc).__name__, exc),
              flush=True)
