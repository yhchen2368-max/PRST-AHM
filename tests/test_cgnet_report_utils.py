import numpy as np

from PRSTCore.ad_core.simulators import (
    convert_report_to_schedule,
    get_report_ministeps,
    get_report_output,
    simulate_schedule_ad,
)
from PRSTCore.ad_core.solvers import AGMGSolverAD, AMGCL_CPRSolverAD


def _run_case():
    schedule = {
        'step': {'val': np.array([1.0, 1.0]), 'control': np.array([0, 0], dtype=int)},
        'control': [{'W': [{'type': 'rate', 'val': 1.0, 'sign': -1, 'status': True}]}],
    }
    state0 = {'time': 0.0}
    model = {'porevolume': np.array([1.0])}
    _, _, report = simulate_schedule_ad(state0, model, schedule, return_report=True)
    return schedule, report


def test_report_utils_basic_flow():
    schedule, report = _run_case()
    new_schedule, timesteps, numsteps = convert_report_to_schedule(report, schedule)
    assert 'step' in new_schedule
    assert timesteps.size >= 2
    assert numsteps.size == 2

    minis = get_report_ministeps(report)
    assert minis.size >= 2

    out = get_report_output(report, kind='nonlinearIterations', ministeps=True)
    assert out['total'].size >= 2
    assert out['wasted'].size == out['total'].size


def test_report_output_nonlinear_report_hierarchy():
    schedule, report = _run_case()
    cr = report['ControlstepReports'][0]
    assert 'NonlinearReport' in cr
    assert isinstance(cr['NonlinearReport'], list)
    assert len(cr['StepReports']) >= 1
    assert 'NonlinearReport' in cr['StepReports'][0]

    lout = get_report_output(report, kind='linearSolverTime', ministeps=True)
    assert lout['total'].size >= 1


def test_solver_specific_report_extraction():
    schedule = {
        'step': {'val': np.array([1.0]), 'control': np.array([0], dtype=int)},
        'control': [{'W': [{'type': 'rate', 'val': 1.0, 'sign': -1, 'status': True}]}],
    }
    state0 = {'time': 0.0}
    model = {'porevolume': np.array([1.0])}
    solver = {'linearSolver': AGMGSolverAD(extraReport=True)}
    _, _, report = simulate_schedule_ad(state0, model, schedule, NonLinearSolver=solver, return_report=True)
    out = get_report_output(report, kind='linearIterations', ministeps=True, solver='AGMGSolverAD')
    assert out['total'].size >= 1


def test_amgcl_cpr_report_contains_preconditioner():
    A = np.eye(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    solver = AMGCL_CPRSolverAD()
    dx, rel, rep = solver.solveLinearProblem({'Jacobian': A, 'Residuals': -b, 'State': {'pressure': np.array([1.0, 1.0])}})
    assert dx.shape == (4,)
    assert 'PreconditionerReport' in rep
    assert 'AMGCLCPR' in rep


def test_amgcl_cpr_reuse_policy_matches_mrst_switches():
    A = np.eye(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    solver = AMGCL_CPRSolverAD()

    # Default: no update_sprecond/update_ptransfer => reuseMode = 1
    _, _, rep1 = solver.solveLinearProblem({'Jacobian': A, 'Residuals': -b, 'State': {'pressure': np.array([1.0, 1.0])}, 'iterationNo': 1})
    assert rep1['AMGCLCPR']['ReuseMode'] == 1

    # MRST-aligned switch: update_sprecond => reuseMode = 2
    solver.amgcl_setup['update_sprecond'] = True
    _, _, rep2 = solver.solveLinearProblem({'Jacobian': A, 'Residuals': -b, 'State': {'pressure': np.array([1.0, 1.0])}, 'iterationNo': 1})
    assert rep2['AMGCLCPR']['ReuseMode'] == 2

    # Keep reuse and ensure cache action metadata is present from elliptic solver.
    _, _, rep3 = solver.solveLinearProblem({'Jacobian': A, 'Residuals': -b, 'State': {'pressure': np.array([1.0, 1.0])}, 'iterationNo': 2})
    pstage = rep3['PreconditionerReport']['PressureSolve']['Report']
    assert 'CacheAction' in pstage


def test_amgcl_cpr_strategy_mrst_and_drs_params():
    A = np.eye(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])

    s_mrst = AMGCL_CPRSolverAD(strategy='mrst', decoupling='trueIMPES')
    _, _, r1 = s_mrst.solveLinearProblem({'Jacobian': A, 'Residuals': -b, 'State': {'pressure': np.array([1.0, 1.0])}})
    c1 = r1['AMGCLCPR']
    assert c1['UseDRS'] is True
    assert c1['DRSEpsPS'] == -1e8
    assert c1['DRSEpsDD'] == -1e8
    assert c1['DRSRowWeightsCount'] == 4

    s_adrs = AMGCL_CPRSolverAD(strategy='amgcl_drs', decoupling='quasiimpes', diagonalTol=0.3, couplingTol=0.1)
    _, _, r2 = s_adrs.solveLinearProblem({'Jacobian': A, 'Residuals': -b, 'State': {'pressure': np.array([1.0, 1.0])}})
    c2 = r2['AMGCLCPR']
    assert c2['UseDRS'] is True
    assert abs(c2['DRSEpsPS'] - 0.1) < 1e-15
    assert abs(c2['DRSEpsDD'] - 0.3) < 1e-15
    assert c2['DRSRowWeightsCount'] == 4

    s_amgcl = AMGCL_CPRSolverAD(strategy='amgcl', decoupling='none')
    _, _, r3 = s_amgcl.solveLinearProblem({'Jacobian': A, 'Residuals': -b, 'State': {'pressure': np.array([1.0, 1.0])}})
    c3 = r3['AMGCLCPR']
    assert c3['UseDRS'] is False
    assert c3['DRSRowWeightsCount'] == 0


def test_amgcl_cpr_scaling_and_adjoint_scaling_paths():
    A = np.eye(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    solver = AMGCL_CPRSolverAD(strategy='mrst', decoupling='quasiimpes')
    solver.pressureScaling = 2.0

    As, bs, scaling, _ = solver.applyScaling(A, b)
    assert As.shape == A.shape
    assert bs.shape == b.shape
    assert 'M' in scaling
    assert 'D' in scaling

    Aa, ba, scaling_a = solver.applyScalingAdjoint(A, b)
    assert Aa.shape == A.shape
    assert ba.shape == b.shape
    v = np.ones((4,), dtype=float)
    vu = solver.undoScalingAdjoint(v, scaling_a)
    assert vu.shape == v.shape
