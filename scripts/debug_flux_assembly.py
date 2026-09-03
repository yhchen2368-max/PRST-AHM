"""Debug flux divergence assembly to find why Jacobian is zero."""
import sys
sys.path.insert(0, '.')
import numpy as np
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

print('Loading SPE1...')
state0, model, schedule, solver = init_eclipse_problem_ad('examples/SpE1/BENCH_SPE1.DATA')

dt = schedule['step']['val'][0]
forces = model.getDrivingForces(schedule['control'][0])

# Get pressures
p = state0['pressure']
print(f'\n=== Pressure Information ===')
print(f'All pressures equal? {np.all(p == p[0])}')
print(f'Min pressure: {p.min():.6e}')
print(f'Max pressure: {p.max():.6e}')
print(f'Std pressure: {np.std(p):.6e}')
print(f'First 5 pressures: {p[:5]}')

# Get mobilies
pvt = model._phase_pvt(p)
lamW, lamO, lamG, _ = model._three_phase_mobility(
    p, state0['sW'], state0['sG'],
    rs_override=state0.get('rs'),
    rv_override=state0.get('rv'),
)

print(f'\n=== Mobility Information ===')
print(f'lamW: min={lamW.min():.6e}, max={lamW.max():.6e}, sum={lamW.sum():.6e}')
print(f'lamO: min={lamO.min():.6e}, max={lamO.max():.6e}, sum={lamO.sum():.6e}')
print(f'lamG: min={lamG.min():.6e}, max={lamG.max():.6e}, sum={lamG.sum():.6e}')
print(f'First 5 lamW: {lamW[:5]}')

# Debug _assemble_flux_divergence manually
ops = model.operators or {}
N = np.asarray(ops.get('N', np.zeros((0, 2), dtype=int)), dtype=int)
T = np.asarray(ops.get('T', np.zeros((0,), dtype=float)), dtype=float).ravel()

print(f'\n=== Operator Information ===')
print(f'N shape: {N.shape}, rows={N.shape[0]}')
print(f'T shape: {T.shape}, length={len(T)}')
print(f'First 5 N pairs: {N[:5]}')
print(f'First 5 T values: {T[:5]}')

# Manually compute flux divergence
nc = model._num_cells()
div_w_debug = np.zeros((nc,), dtype=float)
Lw_debug_rows, Lw_debug_cols, Lw_debug_vals = [], [], []

print(f'\n=== Flux Assembly Debug ===')
print(f'Processing {min(N.shape[0], T.size)} faces...')

nonzero_count = 0
zero_count = 0
skipped_count = 0

for f in range(min(N.shape[0], T.size)):
    c1_1based = int(N[f, 0])
    c2_1based = int(N[f, 1])
    c1 = c1_1based - 1
    c2 = c2_1based - 1
    
    if c1 < 0 or c2 < 0 or c1 >= nc or c2 >= nc:
        skipped_count += 1
        continue
    
    dp = float(p[c1] - p[c2])
    up = c1 if dp >= 0 else c2
    tf = float(T[f])
    
    bw_up = float(pvt['bw'][up])
    gw = tf * float(lamW[up]) * bw_up
    fw = gw * dp
    
    if gw != 0.0:
        nonzero_count += 1
        div_w_debug[c1] += fw
        div_w_debug[c2] -= fw
        Lw_debug_rows.extend([c1, c1, c2, c2])
        Lw_debug_cols.extend([c1, c2, c1, c2])
        Lw_debug_vals.extend([gw, -gw, -gw, gw])
    else:
        zero_count += 1
    
    if f < 5:
        print(f'Face {f}: c1={c1}, c2={c2}, dp={dp:.6e}, tf={tf:.6e}, lamW[up]={lamW[up]:.6e}, bw={bw_up:.6e}, gw={gw:.6e}')

print(f'\nProcessing summary:')
print(f'  Non-zero gw: {nonzero_count}')
print(f'  Zero gw: {zero_count}')
print(f'  Skipped (invalid indices): {skipped_count}')
print(f'  Total Jacobian entries: {len(Lw_debug_vals)}')
print(f'\nDiv_w non-zeros: {np.count_nonzero(div_w_debug)}')
print(f'First 5 div_w: {div_w_debug[:5]}')

# Check if problem is with all pressures being equal
if np.all(p == p[0]):
    print(f'\n*** ALL PRESSURES EQUAL: This explains zero pressure gradients dp! ***')
    print(f'When all p are equal, dp=0 for all faces → gw=0 (no contribution)')
    print(f'Jacobian will only have facility coupling, no reservoir-to-reservoir coupling!')
