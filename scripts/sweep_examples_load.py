"""Try loading every top-level (non-INCLUDE) DATA deck under examples/ via
init_eclipse_problem_ad, and report which ones parse cleanly vs fail, plus
basic size info so we can judge which ones are feasible to actually simulate.
"""
import sys
import time
import traceback

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

DECKS = [
    'examples/EGG/Egg_Model_ECL.DATA',
    'examples/Norne/NORNE_ATW2013.DATA',
    'examples/Norne/Norne_simplified/NORNE_ATW2013.DATA',
    'examples/SPE9/SPE9.DATA',
    'examples/SPE9/SPE9_CP.DATA',
    'examples/SPE9/SPE9_CP_GROUP.DATA',
    'examples/SPE9/SPE9_CP_SHORT.DATA',
    'examples/SPE9/SPE9_CP_SHORT_RESTART.DATA',
    'examples/SpE1/BENCH_SPE1.DATA',
    'examples/SpE1/SPE1CASE1.DATA',
    'examples/SpE1/SPE1CASE1_INF.DATA',
    'examples/SpE1/SPE1CASE1_MID.DATA',
    'examples/SpE1/SPE1CASE2.DATA',
    'examples/SpE1/SPE1CASE2_2P.DATA',
    'examples/SpE1/SPE1CASE2_ACTNUM.DATA',
    'examples/SpE1/SPE1CASE2_ACTNUM_RESTART.DATA',
    'examples/SpE1/SPE1CASE2_FAMII.DATA',
    'examples/SpE1/SPE1CASE2_NOWELLS.DATA',
    'examples/SpE1/SPE1CASE2_OILGAS.DATA',
    'examples/SpE1/SPE1CASE2_RESTART.DATA',
    'examples/SpE1/SPE1CASE2_SLGOF.DATA',
    'examples/SpE1/SPE1CASE2_THERMAL.DATA',
    'examples/sleipner/SLEIPNER_ORG.DATA',
    'examples/spe10model1/SPE10_MODEL1.DATA',
    'examples/spe10model1/SPE10_MODEL1_CP.DATA',
    'examples/spe10model2/SPE10_MODEL2.DATA',
    'examples/spe3/SPE3CASE1.DATA',
    'examples/spe3/SPE3CASE2.DATA',
    'examples/spe5/SPE5CASE1.DATA',
]


def main():
    results = []
    for deck in DECKS:
        t0 = time.time()
        try:
            s0, model, schedule, extra = init_eclipse_problem_ad(deck)
            nc = len(s0['pressure'])
            nsteps = len(schedule['step']['val'])
            phases = ''.join(p for p, on in (('O', getattr(model, 'oil', False)),
                                              ('W', getattr(model, 'water', False)),
                                              ('G', getattr(model, 'gas', False))) if on)
            dt = time.time() - t0
            results.append((deck, 'OK', nc, nsteps, phases, dt, None))
            print(f'OK   {deck:<55} cells={nc:<8} steps={nsteps:<5} phases={phases:<4} load={dt:.2f}s')
        except Exception as e:
            dt = time.time() - t0
            err = f'{type(e).__name__}: {str(e)[:180]}'
            results.append((deck, 'FAIL', None, None, None, dt, err))
            print(f'FAIL {deck:<55} load={dt:.2f}s  {err}')
            print('  ' + traceback.format_exc().replace('\n', '\n  ').rstrip())

    print('\n=== Summary ===')
    ok = [r for r in results if r[1] == 'OK']
    fail = [r for r in results if r[1] == 'FAIL']
    print(f'{len(ok)}/{len(results)} decks loaded OK, {len(fail)} failed')
    if fail:
        print('Failed decks:')
        for deck, _, _, _, _, _, err in fail:
            print(f'  {deck}: {err}')


if __name__ == '__main__':
    main()
