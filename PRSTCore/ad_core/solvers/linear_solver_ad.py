import numpy as _np
import time as _time
try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except Exception:
    _sp = None
    _spla = None


class LinearSolverAD:
    def __init__(self, solver=None, verbose=False, tolerance=1e-8, maxIterations=25,
                 extraReport=False, reduceToCell=False, id=''):
        self.solver = solver or _np.linalg.solve
        self.verbose = bool(verbose)
        self.tolerance = float(tolerance)
        self.maxIterations = int(maxIterations)
        self.extraReport = bool(extraReport)
        self.reduceToCell = bool(reduceToCell)
        self.id = str(id)
        self.iterations = 0
        self.lastResidual = None

    def solveAdjointProblem(self, problemPrev, problemCurr, nextLambdaVec,
                            objective, model=None, **kwargs):
        """Port of ``LinearSolverAD.solveAdjointProblem``.

        One backward step of

            J_n^T lambda_n = -(dg_n/dx_n)^T - B_{n+1}^T lambda_{n+1}

        with ``problemCurr`` carrying ``J_n = dR_n/dx_n`` and
        ``problemPrev`` carrying ``B_{n+1} = dR_{n+1}/dx_n``.

        It lives on the base class because MRST puts it there: every
        solver in the family inherits this and overrides only the linear
        solve underneath. ``AMGCL_CPRSolverAD`` already carried its own
        copy here, which densified the Jacobian -- tolerable on SPE1's
        908 unknowns, 200 GB on a field model's 162276 -- so this one
        stays sparse throughout.

        MRST substitutes ``problemCurr`` for an absent ``problemPrev``,
        which is what lets the final step, where nothing feeds back, run
        the same code as every other.
        """
        A = self._sparse(problemCurr['Jacobian'])
        n = A.shape[0]

        b = _np.zeros(n)
        row = self._objective_row(objective, n)
        if row is not None:
            b -= row

        if nextLambdaVec is not None:
            previous = problemPrev if problemPrev is not None else problemCurr
            B = self._sparse(previous['Jacobian'])
            lam = _np.asarray(nextLambdaVec, dtype=float).ravel()
            if B.shape[0] == lam.size:
                fed_back = B.T @ lam
                b -= (fed_back[:n] if fed_back.size >= n
                      else _np.pad(fed_back, (0, n - fed_back.size)))

        lam = self.solveLinearSystem(A.T, b)
        lam = _np.asarray(lam[0] if isinstance(lam, tuple) else lam,
                          dtype=float).ravel()
        return lam, lam, self.getSolveReport()

    def solveLinearSystem(self, A, b):
        """A sparse direct solve -- MATLAB's backslash by another name."""
        b = _np.asarray(b, dtype=float).ravel()
        if _sp is not None and _sp.issparse(A):
            return _spla.spsolve(A.tocsc(), b)
        return _np.linalg.solve(_np.asarray(A, dtype=float), b)

    @staticmethod
    def _sparse(matrix):
        if _sp is not None and _sp.issparse(matrix):
            return matrix.tocsr()
        if _sp is None:
            return _np.asarray(matrix, dtype=float)
        return _sp.csr_matrix(_np.asarray(matrix, dtype=float))

    @staticmethod
    def _objective_row(objective, n):
        """``-objective.jac{1}'`` as a dense column.

        An AD scalar's single Jacobian row *is* ``dg/dx``. A step the
        objective does not see contributes nothing, and MRST leaves the
        right-hand side empty for it rather than reading the absence as
        a zero that happens to be correct.
        """
        if objective is None:
            return None
        jac = getattr(objective, 'jac', None)
        if jac is not None:
            row = (_np.asarray(jac.todense()).ravel()
                   if hasattr(jac, 'todense') else _np.asarray(jac).ravel())
        else:
            row = _np.atleast_1d(_np.asarray(objective, dtype=float)).ravel()
        if row.size == 0:
            return None
        if row.size == n:
            return row
        out = _np.zeros(n)
        out[:min(row.size, n)] = row[:n]
        return out

    def getSolveReport(self, **kwargs):
        report = {
            'Iterations': 0,
            'Residual': 0.0,
            'SolverTime': 0.0,
            'LinearSolutionTime': 0.0,
            'PreparationTime': 0.0,
            'PostProcessTime': 0.0,
            'Converged': True,
        }
        report.update(kwargs)
        return report

    def _get_system(self, problem):
        if hasattr(problem, 'getLinearSystem'):
            A, b = problem.getLinearSystem()
        elif isinstance(problem, dict) and 'Jacobian' in problem and 'Residuals' in problem:
            A = problem['Jacobian']
            b = -_np.asarray(problem['Residuals'], dtype=float)
        else:
            raise ValueError('LinearSolverAD requires a LinearizedProblem or dict with Jacobian/Residuals.')
        if _sp is not None and _sp.issparse(A):
            Amat = A.tocsr().astype(float)
        else:
            Amat = _np.asarray(A, dtype=float)
        return Amat, _np.asarray(b, dtype=float).ravel()

    def solveLinearProblem(self, problem, model=None):
        t0 = _time.perf_counter()
        A, b = self._get_system(problem)

        if self.verbose:
            print('LinearSolverAD: solving A x = b, shape', A.shape)
        try:
            if _sp is not None and _sp.issparse(A):
                if self.solver is _np.linalg.solve and _spla is not None:
                    dx = _spla.spsolve(A, b)
                else:
                    dx = self.solver(A, b)
            else:
                dx = self.solver(A, b)
        except Exception as exc:
            raise RuntimeError('LinearSolverAD failed: %s' % exc) from exc
        self.iterations += 1
        dx = _np.asarray(dx, dtype=float).ravel()
        self.lastResidual = float(_np.linalg.norm(A.dot(dx) - b))
        converged = self.lastResidual <= self.tolerance
        elapsed = float(_time.perf_counter() - t0)
        report = self.getSolveReport(
            Iterations=1,
            Residual=self.lastResidual,
            SolverTime=elapsed,
            LinearSolutionTime=elapsed,
            Converged=converged,
        )
        if self.extraReport:
            report['ResidualHistory'] = [self.lastResidual]
        return dx, self.lastResidual, report

    def asSolver(self, solver):
        self.solver = solver
        return self
