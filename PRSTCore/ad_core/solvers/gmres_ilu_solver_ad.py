"""SciPy port of MRST's ``GMRES_ILUSolverAD``.

MRST selects this solver for large deck cases when AMGCL/AGMG are not
available.  It uses GMRES preconditioned by a no-fill ILU and deliberately
does *not* replace a non-converged Krylov solve with a dense direct solve.
"""

import time as _time
import numpy as _np

from .linear_solver_ad import LinearSolverAD


class GMRES_ILUSolverAD(LinearSolverAD):
    """GMRES with the direct SciPy equivalent of MRST's ILU(0) path."""

    def __init__(self, tolerance=1.0e-4, maxIterations=100, restart=10,
                 verbose=False, ilutype='nofill', dropTolerance=0.0,
                 modifiedIncompleteILU='off', udiagReplacement=True,
                 pivotThreshold=1.0, reorderEquations=True, **kwargs):
        super().__init__(verbose=verbose, tolerance=tolerance,
                         maxIterations=maxIterations,
                         extraReport=kwargs.get('extraReport', False))
        self.restart = int(restart)
        self.ilutype = str(ilutype).lower()
        self.dropTolerance = float(dropTolerance)
        self.modifiedIncompleteILU = str(modifiedIncompleteILU).lower()
        self.udiagReplacement = bool(udiagReplacement)
        self.pivotThreshold = float(pivotThreshold)
        self.reorderEquations = bool(reorderEquations)

    @staticmethod
    def _reorder_for_ilu(A, b):
        """Direct sparse port of MRST ``reorderForILU(A, b)``.

        Only equation rows are swapped.  The column ordering, and therefore
        the variable vector returned by GMRES, is unchanged.
        """
        diagonal = A.diagonal()
        bad = _np.flatnonzero(diagonal == 0.0)
        if bad.size == 0:
            return A, b
        coo = A.tocoo()
        row_to_cols = {}
        col_to_rows = {}
        for row, col in zip(coo.row, coo.col):
            row_to_cols.setdefault(int(row), []).append(int(col))
            col_to_rows.setdefault(int(col), []).append(int(row))
        new_indices = _np.zeros(bad.size, dtype=int)
        used_rows = set()
        for k, entry in enumerate(bad):
            possible = set(row_to_cols.get(int(entry), ()))
            candidates = set(col_to_rows.get(int(entry), ()))
            choices = sorted((possible & candidates) - used_rows)
            if choices:
                new_indices[k] = choices[0]
                used_rows.add(choices[0])
        permutation = _np.arange(A.shape[0], dtype=int)
        good = new_indices != 0
        if _np.any(good):
            # MATLAB's simultaneous indexed assignments use the original
            # vector on both right-hand sides.
            old = permutation.copy()
            permutation[new_indices[good]] = old[bad[good]]
            permutation[bad[good]] = old[new_indices[good]]
        return A[permutation, :].tocsr(), b[permutation]

    def solveLinearProblem(self, problem, model=None):
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _spla

        if hasattr(problem, 'getLinearSystem'):
            A, b = problem.getLinearSystem()
        else:
            A = problem['Jacobian']
            b = -_np.asarray(problem['Residuals'], dtype=float).ravel()
        A = _sp.csr_matrix(A, dtype=float)
        b = _np.asarray(b, dtype=float).ravel()
        if A.shape[0] != A.shape[1] or A.shape[0] != b.size:
            raise ValueError('GMRES_ILUSolverAD expects a square compatible system')
        if self.reorderEquations:
            A, b = self._reorder_for_ilu(A, b)

        timer = _time.perf_counter()
        # MATLAB ``ilu(A, struct('type', 'nofill', 'droptol', 0,
        # 'milu', 'off', 'udiag', true, 'thresh', 1))``.  SuperLU's
        # NATURAL permutation keeps its factorization on the input pattern;
        # fill_factor=1 and drop_tol=0 select the corresponding ILU(0)
        # branch.  A LinearOperator applies U\(L\x), matching GMRES's
        # M1/M2 preconditioner pair in MRST.
        if self.ilutype != 'nofill':
            raise NotImplementedError('Only MRST GMRES_ILUSolverAD nofill ILU is supported')
        ilu = _spla.spilu(
            A.tocsc(), drop_tol=self.dropTolerance, fill_factor=1.0,
            diag_pivot_thresh=self.pivotThreshold, permc_spec='NATURAL',
        )
        preconditioner = _spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=float)
        iteration_counter = [0]

        def count_iteration(_):
            iteration_counter[0] += 1

        dx, flag = _spla.gmres(
            A, b, M=preconditioner, restart=max(1, self.restart),
            rtol=self.tolerance, atol=0.0,
            maxiter=min(self.maxIterations, A.shape[0]),
            callback=count_iteration, callback_type='legacy',
        )
        residual = float(_np.linalg.norm(A.dot(dx) - b) /
                         max(_np.linalg.norm(b), 1.0e-30))
        elapsed = float(_time.perf_counter() - timer)
        self.iterations += int(iteration_counter[0])
        self.lastResidual = residual
        report = self.getSolveReport(
            Residual=residual,
            Iterations=int(iteration_counter[0]),
            Converged=(int(flag) == 0),
            SolverTime=elapsed,
            LinearSolutionTime=elapsed,
        )
        report['GMRESFlag'] = int(flag)
        return _np.asarray(dx, dtype=float).ravel(), residual, report
