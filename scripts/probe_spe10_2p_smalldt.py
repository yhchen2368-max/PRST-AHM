"""SPE10 Model 2 (2-phase deck): does Newton converge on a small first step?

The 1-day report step diverges (Newton residual oscillates ~1e4-1e6, worst
CNV_W, relax stays 1).  Here we drive solveTimestep directly at small dt with
residual-decrease damping enabled, to find the step size where Newton works.
"""
import sys
import time
from copy import deepcopy

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD

DECK = r"examples\spe10model2\SPE10_MODEL2_2P.DATA"
DT_CANDIDATES = [8640.0, 3600.0, 360.0]  # 0.1 d, 1 h, 6 min

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)
model, state0 = model.prepareReportstep(state0, state0, 1.0, forces)
nc = len(state0['pressure'])
print('cells=%d phases(w/o/g)=%s/%s/%s'
      % (nc, bool(model.water), bool(model.oil), bool(model.gas)), flush=True)

for dt in DT_CANDIDATES:
    linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=100, verbose=False,
                                    strategy='mrst', decoupling='trueIMPES',
                                    schurApproxType='full')
    nl = NonLinearSolver(maxIterations=8, minIterations=1, maxTimestepCuts=1,
                         linearSolver=linear, errorOnFailure=False,
                         useRelaxation=True, enforceResidualDecrease=True,
                         verbose=True)
    t0 = time.time()
    try:
        state, report, _ = nl.solveTimestep(
            deepcopy(state0), dt, model, drivingForces=forces,
            initialGuess=deepcopy(state0), controlId=control_id)
        wall = time.time() - t0
        conv = bool(report.get('Converged'))
        iters = report.get('Iterations')
        print('dt=%.4g d (%7.0fs) conv=%s iters=%s wall=%.1fs'
              % (dt / 86400.0, dt, conv, iters, wall), flush=True)
        if conv:
            print('   p range: %.6g .. %.6g Pa, sW range: %.6g .. %.6g'
                  % (float(state['pressure'].min()), float(state['pressure'].max()),
                     float(state['sW'].min()), float(state['sW'].max())), flush=True)
            break
    except Exception as exc:
        print('dt=%.4g d ERROR %s: %s' % (dt / 86400.0, type(exc).__name__, exc),
              flush=True)
