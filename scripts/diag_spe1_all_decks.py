"""Run every SPE1-family deck with the GUI-exact AMGCL CPR config (errorOnFailure=True)
for 6 steps, to find which deck/step stops the GUI."""
import sys
import time
import traceback

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule

DECKS = [
    'examples/SpE1/SPE1CASE1.DATA',
    'examples/SpE1/SPE1CASE1_INF.DATA',
    'examples/SpE1/SPE1CASE1_MID.DATA',
    'examples/SpE1/SPE1CASE2.DATA',
    'examples/SpE1/SPE1CASE2_2P.DATA',
    'examples/SpE1/SPE1CASE2_OILGAS.DATA',
    'examples/SpE1/SPE1CASE2_SLGOF.DATA',
    'examples/SpE1/SPE1CASE2_NOWELLS.DATA',
    'examples/SpE1/BENCH_SPE1.DATA',
]


def run_deck(deck):
    print('==== %s ====' % deck, flush=True)
    try:
        state0, model, schedule, solver = init_eclipse_problem_ad(
            deck, RemoveZeroPoreVolume=True)
    except Exception as exc:
        print('  init ERROR: %s: %s' % (type(exc).__name__, exc), flush=True)
        return
    print('  cells=%d steps=%d phases=(w=%s o=%s g=%s) first_dt=%.4g d'
          % (len(state0['pressure']), len(schedule['step']['val']),
             bool(model.water), bool(model.oil), bool(model.gas),
             schedule['step']['val'][0] / 86400.0), flush=True)
    solver.useLinesearch = True
    solver.enforceResidualDecrease = True
    solver.acceptanceFactor = 2.0
    solver.LinearSolver = AMGCL_CPRSolverBlockAD(
        tolerance=1e-4, maxIterations=50, strategy='mrst',
        decoupling='trueIMPES')
    t0 = time.time()
    try:
        res = simulate_schedule(model, state0, schedule, solver, max_steps=6)
        print('  simulate OK in %.1f s; steps=%d'
              % (time.time() - t0, len(res.get('steps', []))), flush=True)
        for i, st in enumerate(res.get('steps', [])):
            print('    step %d: conv=%s iters=%s'
                  % (i + 1, st.get('converged'),
                     st.get('report', {}).get('Iterations')), flush=True)
    except Exception as exc:
        print('  EXCEPTION at %.1f s: %s: %s'
              % (time.time() - t0, type(exc).__name__, exc), flush=True)
        traceback.print_exc()


for deck in DECKS:
    run_deck(deck)
