from .amgcl_solver_ad import AMGCLSolverAD


class AMGCLSolverBlockAD(AMGCLSolverAD):
    """Block variant of AMGCL solver scaffold."""

    def __init__(self, blockSize=2, tolerance=1e-8, maxIterations=100, verbose=False):
        super().__init__(tolerance=tolerance, maxIterations=maxIterations, verbose=verbose)
        self.blockSize = int(blockSize)
