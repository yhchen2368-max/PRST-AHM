"""SPE10 Model 2 (2-phase): does forced line-search stabilization make Newton
converge on a 0.1-day step?

Default damping only fires when *all* equations oscillate/worsen; on SPE10
only CNV_W blows up, so relax stays 1 and the huge saturation update destroys
the state.  alwaysUseStabilization forces the residual-bisection line search
on every Newton iteration instead.
"""
import sys
import time
from copy import deepcopy

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD

DECK = r"examples\spe10model2\SPE10_MODEL2_2P.DATA"
DT = 8640.0  # 0.1 day

state0, model, schedule, _ = init_eclipse_problem_ad(DECK, RemoveZeroPoreVolume=True)
control_id = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][control_id])
model, state0 = model.updateForChangedControls(state0, forces)
model, state0 = model.prepareReportstep(state0, state0, 1.0, forces)
nc = len(state0['pressure'])
print('cells=%d dt=%.4g d phases(w/o/g)=%s/%s/%s'
      % (nc, DT / 86400.0, bool(model.water), bool(model.oil), bool(model.gas)),
      flush=True)

linear = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=100, verbose=False,
                                strategy='mrst', decoupling='trueIMPES',
                                schurApproxType='full')
nl = NonLinearSolver(maxIterations=12, minIterations=1, maxTimestepCuts=4,
                     linearSolver=linear, errorOnFailure=False,
                     useRelaxation=True, enforceResidualDecrease=True,
                     alwaysUseStabilization=True, useLinesearch=True,
                     acceptanceFactor=2.0, verbose=True)
t0 = time.time()
try:
    state, report, _ = nl.solveTimestep(
        deepcopy(state0), DT, model, drivingForces=forces,
        initialGuess=deepcopy(state0), controlId=control_id)
    wall = time.time() - t0
    print('RESULT conv=%s iters=%s wall=%.1fs'
          % (report.get('Converged'), report.get('Iterations'), wall), flush=True)
    if report.get('Converged'):
        print('   p range: %.6g .. %.6g Pa, sW range: %.6g .. %.6g'
              % (float(state['pressure'].min()), float(state['pressure'].max()),
                 float(state['sW'].min()), float(state['sW'].max())), flush=True)
except Exception as exc:
    print('ERROR %s: %s' % (type(exc).__name__, exc), flush=True)
