"""T142 (FROSIT black-oil, 141x147x28) trial run.

T142 has 433104 active cells of which 242115 have zero pore volume
(PORV/PORO/NTG/MULTPV), which makes the Jacobian structurally singular.
RemoveZeroPoreVolume=True drops those so a linear solver can start.
"""
import sys
import time
from copy import deepcopy

sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

t0 = time.time()
s0, model, schedule, solver = init_eclipse_problem_ad(
    'examples/T142/T142_E100.DATA',
    RemoveZeroPoreVolume=True,
)
print(f'deck+init: {time.time()-t0:.1f}s', flush=True)

nc = len(s0['pressure'])
print(f'active cells (after zero-PV removal): {nc}', flush=True)
print(f'gas={model.gas} water={model.water} oil={model.oil}', flush=True)
print(f'pressure range: {float(s0["pressure"].min()):.2f} .. {float(s0["pressure"].max()):.2f}', flush=True)
if 'sW' in s0:
    print(f'sW range: {float(s0["sW"].min()):.4f} .. {float(s0["sW"].max()):.4f}', flush=True)
if 'sG' in s0:
    print(f'sG range: {float(s0["sG"].min()):.4f} .. {float(s0["sG"].max()):.4f}', flush=True)

# --- run a few report steps ---
nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
print(f'\n=== running {nsteps} report steps ===', flush=True)
t_start = time.time()
state = s0
for step_idx in range(min(nsteps, len(schedule['step']['val']))):
    control_id = int(schedule['step']['control'][step_idx])
    forces = model.getDrivingForces(schedule['control'][control_id])
    model, state = model.updateForChangedControls(state, forces)
    dt = float(schedule['step']['val'][step_idx])

    t1 = time.time()
    solver.errorOnFailure = False
    state, report, _ = solver.solveTimestep(
        deepcopy(state), dt, model,
        drivingForces=forces,
        initialGuess=deepcopy(state),
        controlId=control_id,
    )
    dt_wall = time.time() - t1
    conv = report.get('Converged')
    iters = len(report.get('NonlinearReport', []))
    print(f'step {step_idx}: dt={dt:.1f}d converged={conv} '
          f'nonlinear_iters={iters} wall={dt_wall:.1f}s '
          f'p={float(state["pressure"].min()):.1f}..{float(state["pressure"].max()):.1f} '
          f'sW={float(state["sW"].min()):.4f}..{float(state["sW"].max()):.4f}',
          flush=True)

    # well production data (filled after solve); SI units: m3/s -> m3/day
    import numpy as _np
    ws = state.get('wellSol', [])
    if ws:
        if step_idx == 0:
            print('wellSol keys:', list(ws[0].keys()), flush=True)
        line = []
        for w in ws:
            nm = str(w.get('name', '?'))
            def _v(key):
                v = w.get(key)
                if v is None:
                    return 0.0
                a = _np.atleast_1d(_np.asarray(v, dtype=float))
                return float(a[0]) if a.size else 0.0
            qO = _v('qOs') * 86400.0   # m3/day (negative = production)
            qW = _v('qWs') * 86400.0
            bhp = _v('bhp') / 1e5      # bar
            st = w.get('status')
            line.append(f'{nm}[{"ON" if st else "OFF"}]:qO={qO:8.2f} qW={qW:7.2f} bhp={bhp:7.1f}')
        print('  ' + ' | '.join(line), flush=True)
    state = state

print(f'\ntotal wall: {time.time()-t_start:.1f}s', flush=True)
print('T142 TRIAL DONE', flush=True)
