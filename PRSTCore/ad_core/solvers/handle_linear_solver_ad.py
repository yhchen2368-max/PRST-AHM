import numpy as _np

from .linear_solver_ad import LinearSolverAD


class HandleLinearSolverAD(LinearSolverAD):
    """Wrap a user-provided callable as a LinearSolverAD instance."""

    def __init__(self, solver_handle, verbose=False):
        if not callable(solver_handle):
            raise TypeError('solver_handle must be callable')
        super().__init__(solver=solver_handle, verbose=verbose)

    def solveLinearProblem(self, problem, model=None):
        if hasattr(problem, 'getLinearSystem'):
            A, b = problem.getLinearSystem()
        else:
            A = _np.asarray(problem['Jacobian'], dtype=float)
            b = -_np.asarray(problem['Residuals'], dtype=float)
        dx = self.solver(A, b)
        residual = float(_np.linalg.norm(A.dot(dx) - b))
        return _np.asarray(dx, dtype=float).ravel(), residual, {'iterations': 1}
