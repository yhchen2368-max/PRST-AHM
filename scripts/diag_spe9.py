"""Diagnose SPE9 first Newton step issues."""
import time, sys, numpy as np
sys.path.insert(0, '.')
sys.setrecursionlimit(10000)

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from copy import deepcopy

print('--- init ---')
s0, m, sch, nl = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
print(f'cells={len(s0["pressure"])} gas={m.gas}')

# First control
ctrl_idx = int(sch['step']['control'][0])
ctrl = sch['control'][ctrl_idx]
forces = m.getDrivingForces(ctrl)
dt = float(sch['step']['val'][0])

print(f'dt={dt} control_idx={ctrl_idx} nwells={len(forces.get("W",[]))}')

# Show well info
for w in forces['W'][:5]:
    print(f'  well {w.get("name","?")} type={w.get("type")} val={w.get("val")} sign={w.get("sign")} phase={w.get("phase")}')

# Evaluate first equation
s = deepcopy(s0)
s['time'] = float(s0.get('time', 0.0)) + dt
s = m.validateState(s)
s0_v = m.validateState(s0)
prob, s_out = m.get_equations(s0_v, s, dt, forces)

res = prob['Residuals']
J = prob['Jacobian']
ws = prob['wellSol']

nc = len(s0['pressure'])
print(f'\nresidual shape={res.shape} J shape={J.shape} nnz={J.nnz}')
print(f'res_w  max={np.max(np.abs(res[:nc])):.3e} mean={np.mean(np.abs(res[:nc])):.3e}')
print(f'res_o  max={np.max(np.abs(res[nc:2*nc])):.3e} mean={np.mean(np.abs(res[nc:2*nc])):.3e}')
print(f'res_g  max={np.max(np.abs(res[2*nc:])):.3e} mean={np.mean(np.abs(res[2*nc:])):.3e}')

# Check wellSol
print(f'\nnwells_output={len(ws)}')
for w in ws[:5]:
    print(f'  well {w.get("name","?")} qWs={w.get("qWs",0):.3e} qOs={w.get("qOs",0):.3e} qGs={w.get("qGs",0):.3e} bhp={w.get("bhp",0):.1f}')

# Check diagonal of Jacobian (sparse-safe)
Jdiag = J.diagonal()
print(f'\nJ diag min={np.min(np.abs(Jdiag)):.3e} max={np.max(np.abs(Jdiag)):.3e} zeros={np.sum(np.abs(Jdiag)<1e-30)}')

# Check water block diagonal  
Jwp_diag = J[:nc, :nc].diagonal()
Jws_diag = J[nc:2*nc, nc:2*nc].diagonal()
Jgsg_diag = J[2*nc:, 2*nc:].diagonal()
print(f'Jwp diag min={np.min(np.abs(Jwp_diag)):.3e} avg={np.mean(np.abs(Jwp_diag)):.3e}')
print(f'Jws diag min={np.min(np.abs(Jws_diag)):.3e} avg={np.mean(np.abs(Jws_diag)):.3e}')
print(f'Jgsg diag min={np.min(np.abs(Jgsg_diag)):.3e} avg={np.mean(np.abs(Jgsg_diag)):.3e}')

# Try linear solve
import scipy.sparse.linalg as spla
try:
    dx = spla.spsolve(J, -res)
    print(f'\ndx norm={np.linalg.norm(dx):.3e} p_max={np.max(np.abs(dx[:nc])):.3e} sW_max={np.max(np.abs(dx[nc:2*nc])):.3e} sG_max={np.max(np.abs(dx[2*nc:])):.3e}')
except Exception as e:
    print(f'\nspsolve FAIL: {e}')
