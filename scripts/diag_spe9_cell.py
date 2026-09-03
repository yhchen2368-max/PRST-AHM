"""Trace per-cell water residual for SPE9."""
import numpy as np
from copy import deepcopy
import sys; sys.path.insert(0,'.')
sys.setrecursionlimit(10000)

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

s0, m, sch, nl = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
ctrl = sch['control'][0]; forces = m.getDrivingForces(ctrl)
dt = float(sch['step']['val'][0])

s = deepcopy(s0); s['time'] = 10.0
s = m.validateState(s); s0_v = m.validateState(s0)

nc = 9000
p = s['pressure']; sW = s['sW']; sG = s['sG']
p0 = s0_v['pressure']; sW0 = s0_v['sW']; sG0 = s0_v['sG']
pv = m._average_porevolume()
pvt = m._phase_pvt(p); pvt0 = m._phase_pvt(p0)
lamW, lamO, lamG, _ = m._three_phase_mobility(p, sW, sG)
div_w, div_o, div_g, _, _, _ = m._assemble_flux_divergence(p, lamW, lamO, lamG)

# Manually compute src
src_w = np.zeros(nc); src_o = np.zeros(nc); src_g = np.zeros(nc)
cells_set = set()
for w in forces['W']:
    cells = m._well_cells(w); cells_set.update(cells)
    nperf = len(cells) if cells else 1
    sign = float(w.get('sign', -1)); val = float(w.get('val', 0))
    typ = w.get('type', 'rate'); phase = w.get('phase', 'OIL').upper()
    for c in cells:
        if typ == 'rate' and sign > 0:
            qin = val / nperf
            if phase == 'WATER': src_w[c] += qin * pvt['bw'][c]
            elif phase == 'GAS': src_g[c] += qin * pvt['bg'][c]
            else: src_o[c] += qin * pvt['bo'][c]
        elif typ == 'rate':
            lt = lamW[c] + lamO[c] + lamG[c]
            fw = lamW[c] / lt if lt > 0 else 0
            fo = lamO[c] / lt if lt > 0 else 1
            fg = lamG[c] / lt if lt > 0 else 0
            qout = abs(val) / nperf
            src_w[c] += -fw * qout * pvt['bw'][c]
            src_o[c] += -fo * qout * pvt['bo'][c]
            src_g[c] += -fg * qout * pvt['bg'][c]

inv_dt = 1.0 / max(dt, 1e-30)
acc_w = (pv * pvt['bw'] * sW - pv * pvt0['bw'] * sW0) * inv_dt
res_w = acc_w + div_w - src_w

max_idx = int(np.argmax(np.abs(res_w)))
print(f'max residual idx={max_idx}')
print(f'  res_w = {res_w[max_idx]:.3e}')
print(f'  acc_w = {acc_w[max_idx]:.3e}')
print(f'  div_w = {div_w[max_idx]:.3e}')
print(f'  src_w = {src_w[max_idx]:.3e}')
print(f'  pv    = {pv}')
print(f'  sW0   = {sW0[max_idx]:.6f}')
print(f'  sW    = {sW[max_idx]:.6f}')
print(f'  bw    = {pvt["bw"][max_idx]:.6f}')
print(f'  bw0   = {pvt0["bw"][max_idx]:.6f}')
print(f'  dt    = {dt}')
print(f'  inv_dt = {inv_dt:.6e}')
print(f'  is_well_cell: {max_idx in cells_set}')

for w in forces['W']:
    cells = m._well_cells(w)
    if max_idx in cells:
        print(f'  well: {w.get("name")} type={w.get("type")} val={w.get("val")} sign={w.get("sign")} nperf={len(cells)}')
        break
