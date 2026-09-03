"""Cross-strategy SPE9 first-step result parity check.

Every MRST AMGCL CPR strategy + decoupling should converge to the *same*
final state (scaling preserves the solution); this checks that to machine
precision against an amgcl/none baseline.
"""
import sys
from copy import deepcopy

sys.path.insert(0, '.')

import numpy as np

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import NonLinearSolver, AMGCL_CPRSolverBlockAD


def run(strategy, decoupling):
    s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
    control_id = int(schedule['step']['control'][0])
    forces = model.getDrivingForces(schedule['control'][control_id])
    model, s0 = model.updateForChangedControls(s0, forces)
    dt = float(schedule['step']['val'][0])
    ls = AMGCL_CPRSolverBlockAD(blockSize=3, tolerance=1e-4, maxIterations=100,
                                strategy=strategy, decoupling=decoupling,
                                schurApproxType='full')
    nl = NonLinearSolver(linearSolver=ls, maxIterations=15, errorOnFailure=False)
    state, report, _ = nl.solveTimestep(deepcopy(s0), dt, model,
                                        drivingForces=forces,
                                        initialGuess=deepcopy(s0),
                                        controlId=control_id)
    return state, report


def main():
    base, rep0 = run('amgcl', 'none')
    print('baseline amgcl/none converged:', bool(rep0.get('Converged')))
    cases = [
        ('mrst', 'trueIMPES'), ('mrst', 'quasiIMPES'), ('mrst', 'none'),
        ('mrst_drs', 'trueIMPES'), ('mrst_drs', 'quasiIMPES'),
        ('amgcl', 'trueIMPES'), ('amgcl_drs', 'trueIMPES'),
        ('amgcl_drs', 'quasiIMPES'),
    ]
    worst = 0.0
    for strat, dec in cases:
        st, rep = run(strat, dec)
        dp = float(np.max(np.abs(np.asarray(st['pressure']) - np.asarray(base['pressure']))))
        dsw = float(np.max(np.abs(np.asarray(st['sW']) - np.asarray(base['sW']))))
        dsg = float(np.max(np.abs(np.asarray(st['sG']) - np.asarray(base['sG']))))
        worst = max(worst, dp, dsw, dsg)
        print(f'{strat:9s} {dec:10s} conv={bool(rep.get("Converged"))} '
              f'maxdiff p={dp:.3e} sW={dsw:.3e} sG={dsg:.3e}')
    print(f'\nworst abs diff across all strategies: {worst:.3e}')


if __name__ == '__main__':
    main()
