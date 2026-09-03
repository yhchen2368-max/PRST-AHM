import numpy as _np

from .linear_solver_ad import LinearSolverAD


class BackslashSolverAD(LinearSolverAD):
    """MRST-style direct linear solver using dense solve."""

    def __init__(self, verbose=False):
        super().__init__(solver=_np.linalg.solve, verbose=verbose)
