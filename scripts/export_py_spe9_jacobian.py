"""Export Python SPE9 Jacobian structure for MRST comparison."""
import sys
import numpy as np
sys.path.insert(0, '.')

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from scipy.sparse import find

s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
ctrl = int(schedule['step']['control'][0])
forces = model.getDrivingForces(schedule['control'][ctrl])
model, s0 = model.updateForChangedControls(s0, forces)
dt = float(schedule['step']['val'][0])

problem, state = model.get_equations(s0, s0, dt, forces, iteration=1)

J = problem['Jacobian']
r = problem['Residuals']
nc = len(s0['pressure'])

print(f'Python: nc={nc} J=({J.shape[0]},{J.shape[1]}) nnz={J.nnz} res_max={np.max(np.abs(r)):.3e}')
print(f'Primary vars: pressure({nc}) + sW({nc}) + x({nc}) + facility({len(state.get("facility_primary_variables",[]))})')
print(f'Total unknowns: {J.shape[1]}')

# Analyze block structure
nres = 3 * nc
nfac = J.shape[0] - nres
print(f'Residual blocks: water({nc}) + oil({nc}) + gas({nc}) + facility({nfac})')

# Diagonal statistics per block
for name, sl in [('water', slice(0, nc)), ('oil', slice(nc, 2*nc)), ('gas', slice(2*nc, 3*nc))]:
    diag = J[sl, sl].diagonal()
    print(f'  {name}: diag min={np.min(np.abs(diag)):.3e} max={np.max(np.abs(diag)):.3e} zeros={np.sum(np.abs(diag) < 1e-30)}')

# Cross blocks
J_rp = J[:nc, :nc]          # water-pressure
J_rs = J[:nc, nc:2*nc]      # water-sW
J_rx = J[:nc, 2*nc:3*nc]    # water-x
J_rf = J[:nc, nres:]        # water-facility
print(f'  J_wp nnz={J_rp.nnz} J_ws nnz={J_rs.nnz} J_wx nnz={J_rx.nnz} J_wf nnz={J_rf.nnz}')

J_op = J[nc:2*nc, :nc]
J_os = J[nc:2*nc, nc:2*nc]
J_ox = J[nc:2*nc, 2*nc:3*nc]
J_of = J[nc:2*nc, nres:]
print(f'  J_op nnz={J_op.nnz} J_os nnz={J_os.nnz} J_ox nnz={J_ox.nnz} J_of nnz={J_of.nnz}')

J_gp = J[2*nc:3*nc, :nc]
J_gs = J[2*nc:3*nc, nc:2*nc]
J_gx = J[2*nc:3*nc, 2*nc:3*nc]
J_gf = J[2*nc:3*nc, nres:]
print(f'  J_gp nnz={J_gp.nnz} J_gs nnz={J_gs.nnz} J_gx nnz={J_gx.nnz} J_gf nnz={J_gf.nnz}')

# Facility block
J_ff = J[nres:, nres:]
print(f'  J_ff nnz={J_ff.nnz} shape=({J_ff.shape[0]},{J_ff.shape[1]})')

# Residual statistics
print(f'\nResiduals:')
print(f'  water: max={np.max(np.abs(r[:nc])):.3e}')
print(f'  oil:   max={np.max(np.abs(r[nc:2*nc])):.3e}')
print(f'  gas:   max={np.max(np.abs(r[2*nc:3*nc])):.3e}')
if nfac > 0:
    print(f'  facility: max={np.max(np.abs(r[nres:])):.3e}')

# Save for comparison
rows, cols, vals = find(J)
np.savetxt('spe9_py_jacobian.txt', np.column_stack([rows, cols, vals]), fmt='%d %d %.15e')
np.savetxt('spe9_py_residual.txt', r, fmt='%.15e')
print('\nSaved to spe9_py_jacobian.txt and spe9_py_residual.txt')
print('Note: Python Residuals use the opposite sign of MRST problem.b.')
