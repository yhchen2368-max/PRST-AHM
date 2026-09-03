import numpy as _np
import time as _time

from .linear_solver_ad import LinearSolverAD
from .agmg_solver_ad import AGMGSolverAD


class CPRSolverAD(LinearSolverAD):
    """CPR-like two-stage solver with MRST-style bookkeeping reports."""

    def __init__(self, pressureWeight=1.0, verbose=False, tolerance=1e-8,
                 maxIterations=100, relativeTolerance=1e-4, diagonalTol=1e-2,
                 trueIMPES=False, ellipticVarName='pressure', ellipticSign=-1,
                 ellipticSolver=None, extraReport=False,
                 directSolveThreshold=50000, iluDropTol=1.0e-4,
                 iluFillFactor=10):
        super().__init__(verbose=verbose, tolerance=tolerance,
                         maxIterations=maxIterations, extraReport=extraReport,
                         reduceToCell=True)
        self.pressureWeight = float(pressureWeight)
        self.relativeTolerance = float(relativeTolerance)
        self.diagonalTol = float(diagonalTol)
        self.trueIMPES = bool(trueIMPES)
        self.ellipticVarName = str(ellipticVarName)
        self.ellipticSign = float(ellipticSign)
        self.ellipticSolver = ellipticSolver or AGMGSolverAD(tolerance=tolerance,
                                                              maxIterations=maxIterations,
                                                              verbose=verbose)
        self.directSolveThreshold = int(directSolveThreshold)
        self.iluDropTol = float(iluDropTol)
        self.iluFillFactor = float(iluFillFactor)

    def _pressure_index(self, n, problem=None, model=None):
        if isinstance(problem, dict) and isinstance(problem.get('State', None), dict):
            p = problem['State'].get('pressure', None)
            if p is not None:
                try:
                    nc = int(_np.asarray(p).size)
                    if 0 < nc <= n:
                        return _np.arange(nc, dtype=int)
                except Exception:
                    pass
        if model is not None:
            grid = getattr(model, 'G', None)
            if isinstance(grid, dict):
                cells = grid.get('cells', {})
                if isinstance(cells, dict):
                    nc = int(cells.get('num', 0))
                    if 0 < nc <= n:
                        return _np.arange(nc, dtype=int)
        return _np.arange(0, n, 2, dtype=int)

    def solveLinearProblem(self, problem, model=None):
        t0 = _time.perf_counter()
        A, b = self._get_system(problem)
        n = A.shape[0]

        # Deck-derived GenericBlackOilModel systems are stored in grouped
        # primary-variable order (all cell pressures, then saturations,
        # then well variables), whereas this lightweight CPR scaffold used
        # an interleaved ``0, 2, 4, ...`` pressure index.  Besides being a
        # wrong pressure block, the following dense solve of the remaining
        # 18--25k variables dominated each Newton iteration and caused the
        # accepted timestep sequence to diverge.  For the moderate systems
        # covered by the bundled first-step parity cases, use the exact
        # sparse solve that is mathematically equivalent to completing the
        # CPR correction.  Large systems retain the iterative CPR path
        # below until its block preconditioner is fully ported.
        try:
            import scipy.sparse as _sp
            import scipy.sparse.linalg as _spla
            is_sparse = _sp.issparse(A)
        except Exception:
            is_sparse = False
        if is_sparse and self.directSolveThreshold > 0 and n <= self.directSolveThreshold:
            try:
                dx = _np.asarray(_spla.spsolve(A.tocsc(), b), dtype=float).ravel()
                rfin = b - A.dot(dx)
                relres = float(_np.linalg.norm(rfin) / max(_np.linalg.norm(b), 1e-30))
                elapsed = float(_time.perf_counter() - t0)
                report = self.getSolveReport(
                    Iterations=1,
                    Residual=relres,
                    SolverTime=elapsed,
                    LinearSolutionTime=elapsed,
                    PreparationTime=0.0,
                    PostProcessTime=0.0,
                    Converged=bool(_np.all(_np.isfinite(dx)) and relres <= self.tolerance),
                    FlagGMRES=0 if relres <= self.tolerance else 1,
                    PreconditionerReport={'Type': 'sparse-direct-parity'},
                )
                return dx, relres, report
            except Exception:
                # Fall through to the legacy CPR path for a diagnostic
                # report instead of concealing an unsupported factorization.
                pass

        if is_sparse:
            # Norne's first system is too large for a sparse LU factorization
            # on the available workstation, while the legacy CPR scaffold
            # below picks an interleaved pressure block from a grouped ADI
            # system and then densifies a 67k-by-67k remainder.  A robust
            # ILUT-preconditioned GMRES solve preserves sparsity and reaches
            # the same Newton correction to the requested linear tolerance.
            # Tiny diagonal regularization is restricted to structurally
            # zero pivots introduced by rate-controlled facility equations;
            # the residual is always evaluated against the original matrix.
            try:
                from scipy.sparse import diags as _sp_diags
                from scipy.sparse.linalg import (
                    LinearOperator as _LinearOperator,
                    gmres as _gmres,
                    spilu as _spilu,
                )

                Acsr = A.tocsr()
                diagonal = Acsr.diagonal()
                zero_pivot = _np.abs(diagonal) < 1.0e-12
                Awork = Acsr
                if _np.any(zero_pivot):
                    row_scale = _np.maximum(_np.asarray(_np.abs(Acsr).sum(axis=1)).ravel(), 1.0)
                    Awork = (Acsr + _sp_diags(_np.where(zero_pivot, row_scale * 1.0e-12, 0.0))).tocsc()
                else:
                    Awork = Acsr.tocsc()
                ilu = _spilu(
                    Awork,
                    drop_tol=self.iluDropTol,
                    fill_factor=self.iluFillFactor,
                    permc_spec='COLAMD',
                    diag_pivot_thresh=0.1,
                )
                preconditioner = _LinearOperator(Awork.shape, ilu.solve)
                history = []
                restart = max(1, min(50, int(self.maxIterations)))
                maxiter_cycles = max(1, int(_np.ceil(float(self.maxIterations) / float(restart))))
                dx, flag = _gmres(
                    Awork, b, M=preconditioner,
                    rtol=self.tolerance, atol=0.0,
                    restart=restart, maxiter=maxiter_cycles,
                    callback=lambda residual: history.append(float(residual)),
                    callback_type='pr_norm',
                )
                dx = _np.asarray(dx, dtype=float).ravel()
                rfin = b - A.dot(dx)
                relres = float(_np.linalg.norm(rfin) / max(_np.linalg.norm(b), 1e-30))
                converged = bool(flag == 0 and _np.all(_np.isfinite(dx)) and relres <= self.tolerance)
                fallback_report = None
                if not converged and n <= 50000:
                    tfb = _time.perf_counter()
                    dx_direct = _np.asarray(_spla.spsolve(A.tocsc(), b), dtype=float).ravel()
                    rfin_direct = b - A.dot(dx_direct)
                    relres_direct = float(_np.linalg.norm(rfin_direct) / max(_np.linalg.norm(b), 1e-30))
                    fallback_report = {
                        'Type': 'sparse-direct-fallback',
                        'TriggerResidual': relres,
                        'TriggerFlagGMRES': int(flag),
                        'TriggerIterations': len(history),
                        'FallbackTime': float(_time.perf_counter() - tfb),
                    }
                    dx = dx_direct
                    relres = relres_direct
                    flag = 0 if relres <= self.tolerance else 1
                    converged = bool(_np.all(_np.isfinite(dx)) and relres <= self.tolerance)
                elapsed = float(_time.perf_counter() - t0)
                pre_report = {
                    'Type': 'sparse-ilut-gmres-parity',
                    'DropTolerance': self.iluDropTol,
                    'FillFactor': self.iluFillFactor,
                    'ZeroPivotRegularization': int(_np.count_nonzero(zero_pivot)),
                }
                if fallback_report is not None:
                    pre_report = {
                        'Type': 'sparse-ilut-gmres-with-direct-fallback',
                        'ILUTGMRES': pre_report,
                        'Fallback': fallback_report,
                    }
                report = self.getSolveReport(
                    Iterations=len(history),
                    Residual=relres,
                    SolverTime=elapsed,
                    LinearSolutionTime=elapsed,
                    PreparationTime=0.0,
                    PostProcessTime=0.0,
                    Converged=converged,
                    FlagGMRES=int(flag),
                    PreconditionerReport=pre_report,
                )
                if self.extraReport:
                    report['ResidualHistory'] = history
                return dx, relres, report
            except Exception:
                # Keep the historical implementation available for unusual
                # matrices that cannot be factorized by SuperLU's ILUT.
                pass

        pidx = self._pressure_index(n, problem=problem, model=model)
        other = _np.array([i for i in range(n) if i not in set(pidx.tolist())], dtype=int)

        Ap = self.ellipticSign * A[_np.ix_(pidx, pidx)]
        bp = self.ellipticSign * b[pidx]
        tprep = float(_time.perf_counter() - t0)

        tp0 = _time.perf_counter()
        xp, rp, preport = self.ellipticSolver.solveLinearProblem({'Jacobian': Ap, 'Residuals': -bp}, model)
        tpressure = float(_time.perf_counter() - tp0)

        x0 = _np.zeros_like(b)
        x0[pidx] = xp
        r1 = b - A.dot(x0)

        ts0 = _time.perf_counter()
        if other.size > 0:
            A2 = A[_np.ix_(other, other)]
            b2 = r1[other]
            try:
                xs = _np.linalg.solve(A2, b2)
            except Exception:
                xs = _np.zeros_like(b2)
            x0[other] += xs
        tsmooth = float(_time.perf_counter() - ts0)

        rfin = b - A.dot(x0)
        relres = float(_np.linalg.norm(rfin) / max(_np.linalg.norm(b), 1e-30))
        converged = relres <= self.tolerance
        elapsed = float(_time.perf_counter() - t0)

        report = self.getSolveReport(
            Iterations=2,
            Residual=relres,
            SolverTime=elapsed,
            LinearSolutionTime=tpressure + tsmooth,
            PreparationTime=tprep,
            PostProcessTime=max(0.0, elapsed - tprep - tpressure - tsmooth),
            Converged=converged,
            FlagGMRES=0 if converged else 1,
            PreconditionerReport={
                'Type': 'CPR',
                'PressureSolve': {
                    'Indices': pidx.tolist(),
                    'Report': preport,
                    'Time': tpressure,
                },
                'Smoother': {
                    'Type': 'direct',
                    'Time': tsmooth,
                },
            },
        )
        if self.extraReport:
            report['ResidualHistory'] = [rp, relres]
        return _np.asarray(x0, dtype=float).ravel(), relres, report
