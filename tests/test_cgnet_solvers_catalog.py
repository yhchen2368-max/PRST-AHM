import numpy as np

from PRSTCore.ad_core.solvers import (
    LinearizedProblem,
    LinearSolverAD,
    NonLinearSolver,
    BackslashSolverAD,
    CPRSolverAD,
    AGMGSolverAD,
    AMGCLSolverAD,
    AMGCLSolverBlockAD,
    AMGCL_CPRSolverAD,
    AMGCL_CPRSolverBlockAD,
    GMRES_ILUSolverAD,
    HandleLinearSolverAD,
    NoOpSolverAD,
    getNonLinearSolver,
)


def test_pyamgcl_backend_if_available():
    A = np.eye(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    s = AMGCLSolverAD(tolerance=1e-10, maxIterations=50, usePyAMGCL=True, extraReport=True)
    dx, rel, rep = s.solveLinearProblem({'Jacobian': A, 'Residuals': -b})
    assert dx.shape == (4,)
    assert 'Backend' in rep
    # Either pyamgcl backend is active, or a clear fallback path is reported.
    assert rep['Backend'] in ('pyamgcl', 'scipy/dense-fallback')


def test_solver_catalog_import_and_basic_solve():
    problem = LinearizedProblem(A=np.eye(2), b=np.array([1.0, 2.0]))

    solvers = [
        LinearSolverAD(),
        BackslashSolverAD(),
        CPRSolverAD(),
        AGMGSolverAD(),
        AMGCLSolverAD(),
        AMGCLSolverBlockAD(),
        AMGCL_CPRSolverAD(),
        AMGCL_CPRSolverBlockAD(),
        GMRES_ILUSolverAD(),
        HandleLinearSolverAD(lambda A, b: np.linalg.solve(A, b)),
        NoOpSolverAD(),
    ]
    for solver in solvers:
        dx, _, _ = solver.solveLinearProblem(problem)
        assert dx.shape == (2,)


def test_get_non_linear_solver_factory():
    nls = getNonLinearSolver(MaxIterations=7, Verbose=False)
    assert isinstance(nls, NonLinearSolver)
    assert nls.maxIterations == 7
