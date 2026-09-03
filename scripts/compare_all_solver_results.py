"""Aggregate every scripts/spe9_solver_result_{mumps,amgcl_cpr}[_deck].pkl
pair into one MUMPS vs AMGCL-CPR comparison table."""
import glob
import os
import pickle

import numpy as np

PAIRS = [
    ('SPE9_CP', 'mumps_cp', 'amgcl_cpr'),
    ('SPE9 (block-centred TOPS)', 'mumps_spe9', 'amgcl_cpr_spe9'),
    ('BENCH_SPE1', 'mumps_bench_spe1', 'amgcl_cpr_bench_spe1'),
    ('SPE1CASE1', 'mumps_spe1case1', 'amgcl_cpr_spe1case1'),
    ('SPE1CASE1_INF', 'mumps_spe1case1_inf', 'amgcl_cpr_spe1case1_inf'),
    ('SPE1CASE1_MID', 'mumps_spe1case1_mid', 'amgcl_cpr_spe1case1_mid'),
    ('SPE1CASE2', 'mumps_spe1case2', 'amgcl_cpr_spe1case2'),
    ('SPE1CASE2_2P', 'mumps_spe1case2_2p', 'amgcl_cpr_spe1case2_2p'),
    ('SPE1CASE2_ACTNUM', 'mumps_spe1case2_actnum', 'amgcl_cpr_spe1case2_actnum'),
    ('SPE1CASE2_FAMII', 'mumps_spe1case2_famii', 'amgcl_cpr_spe1case2_famii'),
    ('SPE1CASE2_OILGAS', 'mumps_spe1case2_oilgas', 'amgcl_cpr_spe1case2_oilgas'),
    ('SPE1CASE2_SLGOF', 'mumps_spe1case2_slgof', 'amgcl_cpr_spe1case2_slgof'),
    ('SPE1CASE2_THERMAL', 'mumps_spe1case2_thermal', 'amgcl_cpr_spe1case2_thermal'),
    ('SPE3CASE1', 'mumps_spe3case1', 'amgcl_cpr_spe3case1'),
    ('SPE3CASE2', 'mumps_spe3case2', 'amgcl_cpr_spe3case2'),
    ('SPE5CASE1', 'mumps_spe5case1', 'amgcl_cpr_spe5case1'),
    ('EGG', 'mumps_egg_model_ecl', 'amgcl_cpr_egg_model_ecl'),
    ('SPE10_MODEL1', 'mumps_spe10_model1', 'amgcl_cpr_spe10_model1'),
    ('SPE10_MODEL1_CP', 'mumps_spe10_model1_cp', 'amgcl_cpr_spe10_model1_cp'),
]


def load(name):
    path = f'scripts/spe9_solver_result_{name}.pkl'
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def fmt_status(r):
    if r is None:
        return '(missing)'
    if r.get('ok'):
        return 'OK'
    return 'FAIL: ' + str(r.get('error', '?'))[:60]


def main():
    header = (f'{"Deck":<26}{"MUMPS":<10}{"AMGCL-CPR":<10}'
              f'{"t_mumps[s]":<12}{"t_amgcl[s]":<12}{"speedup":<10}{"max_rel_dp":<12}')
    print(header)
    print('-' * len(header))
    for label, mkey, akey in PAIRS:
        rm = load(mkey)
        ra = load(akey)
        sm = 'OK' if rm and rm.get('ok') else ('FAIL' if rm else '-')
        sa = 'OK' if ra and ra.get('ok') else ('FAIL' if ra else '-')
        tm = f'{rm["sim_time"]:.1f}' if rm else '-'
        ta = f'{ra["sim_time"]:.1f}' if ra else '-'
        speedup = '-'
        maxreldp = '-'
        if rm and ra and rm.get('ok') and ra.get('ok'):
            if rm['sim_time'] > 0:
                speedup = f'{ra["sim_time"] / rm["sim_time"]:.2f}x'
            n = min(len(rm['pressures']), len(ra['pressures']))
            worst = 0.0
            for i in range(n):
                pm = rm['pressures'][i]
                pa = ra['pressures'][i]
                if pm.shape != pa.shape:
                    continue
                denom = max(float(np.abs(pm).max()), 1e-30)
                worst = max(worst, float(np.max(np.abs(pm - pa)) / denom))
            maxreldp = f'{worst:.2e}'
        print(f'{label:<26}{sm:<10}{sa:<10}{tm:<12}{ta:<12}{speedup:<10}{maxreldp:<12}')

    print('\n=== Failure details ===')
    for label, mkey, akey in PAIRS:
        for tag, key in (('mumps', mkey), ('amgcl_cpr', akey)):
            r = load(key)
            if r is not None and not r.get('ok'):
                print(f'{label} [{tag}]: {r.get("error")}')


if __name__ == '__main__':
    main()
