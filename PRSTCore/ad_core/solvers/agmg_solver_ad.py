import numpy as _np
import time as _time
try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except Exception:
    _sp = None
    _spla = None
try:
    import pyamg as _pyamg
except Exception:
    _pyamg = None

from .linear_solver_ad import LinearSolverAD


class AGMGSolverAD(LinearSolverAD):
    """Algebraic-multigrid solver with MRST-compatible parameters and
    reporting.

    Backed by ``pyamg``'s Ruge-Stuben AMG hierarchy (the same role MRST's
    AGMG plays as the CPR elliptic/pressure sub-solver) when ``pyamg`` is
    importable. If ``pyamg`` is unavailable, the hierarchy cannot be built
    for the given matrix, or the AMG-preconditioned Krylov iteration fails
    to reach ``tolerance`` within ``maxIterations``, this falls back to an
    exact sparse direct solve -- callers always get a solution at least as
    accurate as a direct solve would give, with AMG only ever making a
    given call *faster*, never less correct.
    """

    def __init__(self, tolerance=1e-8, maxIterations=100, verbose=False,
                 reuseSetup=False, singleApply=False, extraReport=False,
                 accel='bicgstab'):
        super().__init__(verbose=verbose, tolerance=tolerance,
                         maxIterations=maxIterations, extraReport=extraReport,
                         reduceToCell=True)
        self.setupDone = False
        self.reuseSetup = bool(reuseSetup)
        self.singleApply = bool(singleApply)
        self.accel = str(accel)
        self._ml = None
        self._ml_shape = None

    def setupSolver(self, A, b=None, **kwargs):
        if self.reuseSetup:
            self.setupDone = True
        return self

    def cleanupSolver(self, A=None, b=None, **kwargs):
        if self.reuseSetup and self.setupDone:
            self.setupDone = False
        return self

    def _amg_solve(self, A, b):
        """Attempt a pyamg Ruge-Stuben AMG-preconditioned Krylov solve.

        Returns ``(x, relres, iterations)`` on success, or ``None`` if
        ``pyamg`` is unavailable, the hierarchy could not be built for this
        matrix, or the result is non-finite -- any of which sends the
        caller to the direct-solve fallback.
        """
        if _pyamg is None or _sp is None or not _sp.issparse(A):
            return None
        try:
            Acsr = A.tocsr()
            reuse = self.reuseSetup and self._ml is not None and self._ml_shape == Acsr.shape
            ml = self._ml if reuse else _pyamg.ruge_stuben_solver(Acsr)
            if self.reuseSetup:
                self._ml, self._ml_shape = ml, Acsr.shape
            residual_history = []
            x = ml.solve(b, tol=self.tolerance, maxiter=self.maxIterations,
                         accel=self.accel, residuals=residual_history)
            x = _np.asarray(x, dtype=float).ravel()
            if not _np.all(_np.isfinite(x)):
                return None
            relres = float(_np.linalg.norm(b - Acsr.dot(x)) / max(_np.linalg.norm(b), 1e-30))
            iterations = max(1, len(residual_history) - 1) if residual_history else 1
            return x, relres, iterations
        except Exception:
            return None

    def solveLinearProblem(self, problem, model=None):
        t0 = _time.perf_counter()
        A, b = self._get_system(problem)
        clean_after = False
        if not self.setupDone:
            self.setupSolver(A, b)
            clean_after = True

        residual_history = []
        amg_used = False
        try:
            if self.singleApply:
                # Single preconditioner application approximation.
                if _sp is not None and _sp.issparse(A):
                    d = _np.abs(A.diagonal()).copy()
                else:
                    d = _np.abs(_np.diag(A)).copy()
                d[d == 0] = 1.0
                x = b / d
                it = 1
                relres = float(_np.linalg.norm(A.dot(x) - b) / max(_np.linalg.norm(b), 1e-30))
                residual_history.append(relres)
            else:
                amg_result = self._amg_solve(A, b)
                if amg_result is not None and amg_result[1] > self.tolerance:
                    amg_result = None
                if amg_result is not None:
                    x, relres, it = amg_result
                    amg_used = True
                else:
                    if _sp is not None and _sp.issparse(A) and _spla is not None:
                        x = _spla.spsolve(A, b)
                    else:
                        x = _np.linalg.solve(A, b)
                    it = 1
                    relres = float(_np.linalg.norm(A.dot(x) - b) / max(_np.linalg.norm(b), 1e-30))
                residual_history.append(relres)
        finally:
            if clean_after:
                self.cleanupSolver(A, b)

        converged = relres <= self.tolerance
        elapsed = float(_time.perf_counter() - t0)
        report = self.getSolveReport(
            Iterations=it,
            Residual=relres,
            SolverTime=elapsed,
            LinearSolutionTime=elapsed,
            Converged=converged,
            ReuseSetup=bool(self.reuseSetup),
            SetupDone=bool(self.setupDone),
            SingleApply=bool(self.singleApply),
            PreconditionerReport={'Type': 'pyamg-ruge-stuben' if amg_used else 'sparse-direct-fallback'},
        )
        if self.extraReport:
            report['ResidualHistory'] = residual_history
        return _np.asarray(x, dtype=float).ravel(), relres, report
