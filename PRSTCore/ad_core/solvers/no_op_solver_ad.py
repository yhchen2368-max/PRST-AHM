import numpy as _np

from .linear_solver_ad import LinearSolverAD


class NoOpSolverAD(LinearSolverAD):
    """No-op solver that returns zero updates, used for diagnostics/testing."""

    def __init__(self, verbose=False):
        super().__init__(solver=None, verbose=verbose)

    def solveLinearProblem(self, problem, model=None):
        if hasattr(problem, 'getLinearSystem'):
            A, b = problem.getLinearSystem()
        else:
            A = _np.asarray(problem['Jacobian'], dtype=float)
            b = -_np.asarray(problem['Residuals'], dtype=float)
        n = int(A.shape[1])
        dx = _np.zeros((n,), dtype=float)
        residual = float(_np.linalg.norm(A.dot(dx) - b))
        return dx, residual, {'iterations': 0}
