import sys, copy, time
sys.path.insert(0, '.')
from Cgnet.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from Cgnet.ad_core.solvers import AMGCL_CPRSolverBlockAD
print('start', flush=True)
s0, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
print('init done', flush=True)
ctrl = schedule['control'][int(schedule['step']['control'][0])]
forces = model.getDrivingForces(ctrl)
model, state = model.updateForChangedControls(model.validateState(s0), forces)
dt = float(schedule['step']['val'][0])
print('assemble', flush=True)
problem, state = model.get_equations(state, copy.deepcopy(state), dt, forces, iteration=1)
print('assembled', problem['Jacobian'].shape, problem['Jacobian'].nnz, flush=True)
solver = AMGCL_CPRSolverBlockAD(blockSize=3, tolerance=1e-3, maxIterations=1, verbose=True, s_relaxation='spai0')
print('solve', flush=True)
t0=time.time(); dx,res,rep=solver.solveLinearProblem(problem, model); print('done', time.time()-t0, res, rep, flush=True)
