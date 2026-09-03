from .cpr_solver_ad import CPRSolverAD
from .amgcl_solver_ad import AMGCLSolverAD
import numpy as _np


class AMGCL_CPRSolverAD(CPRSolverAD):
    """AMGCL-backed CPR solver scaffold."""

    def __init__(self, pressureWeight=1.0, tolerance=1e-6, maxIterations=100,
                 verbose=False, diagonalTol=0.2, couplingTol=0.02,
                 decoupling='trueIMPES', strategy='mrst', doApplyScalingCPR=True,
                 useSYMRCMOrdering=False, extraReport=False,
                 blockSize=2, directSolveThreshold=0,
                 iluDropTol=1.0e-5, iluFillFactor=50):
        super().__init__(pressureWeight=pressureWeight, verbose=verbose,
                         tolerance=tolerance, maxIterations=maxIterations,
                         diagonalTol=diagonalTol, trueIMPES=(str(decoupling).lower() == 'trueimpes'),
                         extraReport=extraReport,
                         directSolveThreshold=directSolveThreshold,
                         iluDropTol=iluDropTol,
                         iluFillFactor=iluFillFactor)
        self.couplingTol = float(couplingTol)
        self.decoupling = str(decoupling)
        self.strategy = str(strategy)
        self.doApplyScalingCPR = bool(doApplyScalingCPR)
        self.useSYMRCMOrdering = bool(useSYMRCMOrdering)
        self.pressureScaling = 1.0
        # Report scratch shared with AMGCL_CPRSolverBlockAD, which also
        # writes the block-reduction fields into it.
        self._last_block_prep = {}
        self.amgcl_setup = {
            'solver_id': 2,
            'use_drs': self.strategy.lower().endswith('drs'),
            'drs_eps_ps': self.couplingTol,
            'drs_eps_dd': self.diagonalTol,
            'decoupling': self.decoupling,
            'strategy': self.strategy,
            'update_sprecond': False,
            'update_ptransfer': False,
            'block_size': int(blockSize),
        }
        self.ellipticSolver = AMGCLSolverAD(
            tolerance=tolerance,
            maxIterations=maxIterations,
            verbose=verbose,
            extraReport=extraReport,
            usePyAMGCL=True,
            solver='bicgstab',
            preconditioner='amg',
            relaxation='spai0',
            coarsening='smoothed_aggregation',
            block_size=1,
        )

    def prepareProblemCPR(self, problem, model=None):
        setup = self.amgcl_setup
        update_sp = bool(setup.get('update_sprecond', False))
        update_pt = bool(setup.get('update_ptransfer', False))
        if update_sp or update_pt:
            self.ellipticSolver.reuseMode = 2
            iteration_no = None
            if hasattr(problem, 'get'):
                iteration_no = problem.get('iterationNo', None)
            if iteration_no is None:
                iteration_no = getattr(problem, 'iterationNo', None)
            if iteration_no == 1:
                self.ellipticSolver.resetAMGCL()
        else:
            self.ellipticSolver.reuseMode = 1

        if isinstance(problem, dict) and isinstance(problem.get('State', None), dict):
            p = problem['State'].get('pressure', None)
            if p is not None:
                self.pressureScaling = float(max(1e-30, _abs_mean(p)))

        if isinstance(problem, dict) and 'Jacobian' in problem and 'Residuals' in problem:
            problem = self._configure_cpr_strategy(problem, model)
        return problem

    def _configure_cpr_strategy(self, problem, model=None):
        # Preserve CSR matrices so the configured pyamgcl extension can handle
        # SPE9/EGG/NORNE systems without dense materialization.
        import scipy.sparse as _sp
        A0 = problem['Jacobian']
        A = A0.tocsr().astype(float) if _sp.issparse(A0) else _np.asarray(A0, dtype=float)
        r = _np.asarray(problem['Residuals'], dtype=float).ravel()
        b = -r

        n = A.shape[0]
        bz = int(self.amgcl_setup.get('block_size', 0) or 0)
        if bz <= 0:
            bz = 2
            self.amgcl_setup['block_size'] = bz
        nc = int(self.amgcl_setup.get('cell_size', 0) or 0)
        if nc <= 0:
            nc = max(1, n // bz)
            self.amgcl_setup['cell_size'] = nc
        ncv = min(n, bz * nc)
        self.amgcl_setup['cell_num'] = int(nc)

        strategy = str(self.strategy).lower()
        decoupling = str(self.decoupling).lower()
        if strategy in ('mrst', 'mrst_drs'):
            self.amgcl_setup['use_drs'] = True
            self.amgcl_setup['drs_eps_ps'] = -1e8
            self.amgcl_setup['drs_eps_dd'] = -1e8
        elif strategy == 'amgcl':
            self.amgcl_setup['use_drs'] = False
            self.amgcl_setup['drs_row_weights'] = []
        elif strategy == 'amgcl_drs':
            self.amgcl_setup['use_drs'] = True
            self.amgcl_setup['drs_eps_ps'] = self.couplingTol if _np.isfinite(self.couplingTol) else -1e8
            self.amgcl_setup['drs_eps_dd'] = self.diagonalTol if _np.isfinite(self.diagonalTol) else -1e8
        else:
            raise ValueError('Unknown CPR strategy %s' % self.strategy)

        # MRST AMGCL_CPRSolverAD.prepareProblemCPR weights each *cell
        # equation* by getScalingFactorsCPR before the system is assembled,
        # and getScalingInternalCPR then returns unit weights for
        # trueIMPES so that D takes the plain sum of the already-weighted
        # rows.  Skipping the first half leaves 'trueIMPES' summing
        # unweighted mass balances -- MRST's 'none' under another name.
        A, b = self._apply_true_impes_scaling(A, b, nc, bz, problem, model)

        A, b, scaling, _ = self.applyScaling(A, b)

        out = dict(problem)
        out['Jacobian'] = A
        out['Residuals'] = -b
        out['CPRRowWeights'] = _np.asarray(self.amgcl_setup.get('drs_row_weights', []), dtype=float)
        out['Scaling'] = scaling
        return out

    def _apply_true_impes_scaling(self, A, b, nc, bz, problem, model):
        """MRST ``getScalingFactorsCPR``: weight each cell equation.

        For the 'trueimpes' decoupling MRST multiplies every cell equation
        by a state-dependent scale factor before the well reduction.  When
        the fluid state is unavailable it silently falls back to no scaling
        (matching MRST's try/catch around getScalingFactorsCPR).

        ``A`` may be the stored Jacobian by reference; the scaled system is
        rebuilt through ``vstack`` (a sparse row-slice assignment on a
        27k-row CSR costs seconds in scipy) so the caller's matrix is left
        untouched.
        """
        import numpy as _np
        import scipy.sparse as _sp
        if not (self.doApplyScalingCPR
                and str(self.decoupling).lower() == 'trueimpes'):
            return A, b
        state = None
        if isinstance(problem, dict) and isinstance(problem.get('State'), dict):
            state = problem['State']
        if model is None or state is None:
            return A, b
        try:
            from PRSTCore.ad_core import cpr_decoupling
            W = cpr_decoupling.decoupling_weights('trueimpes', A, nc, bz,
                                                  model=model, state=state)
            W = _np.asarray(W, dtype=float)
            if W.shape != (nc, bz):
                return A, b
            ncv = int(nc * bz)
            # A[:ncv] is variable-grouped (unknown v of cell i at v*nc + i)
            # while W is cell-major (i, v); transpose+ravel maps one to one.
            wflat = W.T.ravel()
            if wflat.size != ncv:
                return A, b
            if _sp.issparse(A):
                A = A.tocsr()
                n = A.shape[0]
                top = A[:ncv, :].copy()
                idx = _np.repeat(_np.arange(ncv, dtype=int),
                                 _np.diff(top.indptr))
                top.data = top.data * wflat[idx]
                if n > ncv:
                    A = _sp.vstack([top, A[ncv:, :]]).tocsr()
                else:
                    A = top
            else:
                A = A.copy()
                A[:ncv, :] = A[:ncv, :] * wflat[:, None]
            b = b.copy()
            b[:ncv] = b[:ncv] * wflat
            self._last_block_prep['TrueIMPESWeighted'] = True
        except Exception:
            pass
        return A, b

    def applyScaling(self, A, b, x0=None):
        import scipy.sparse as _sp
        is_sparse = _sp.issparse(A)
        A = A.tocsr().astype(float) if is_sparse else _np.asarray(A, dtype=float)
        b = _np.asarray(b, dtype=float).ravel()
        x = None if x0 is None else _np.asarray(x0, dtype=float).ravel().copy()

        n = int(A.shape[0])
        bz = int(self.amgcl_setup.get('block_size', 2) or 2)
        nc = int(self.amgcl_setup.get('cell_size', max(1, n // bz)) or max(1, n // bz))
        ncv = min(n, bz * nc)
        psub = _np.arange(0, ncv, bz, dtype=int)
        scaling = {}

        # MRST: right scaling by pressureScaling on pressure-unknown columns.
        if not _np.isclose(self.pressureScaling, 1.0):
            d = _np.ones((n,), dtype=float)
            d[psub] = float(self.pressureScaling)
            A = A.multiply(d[None, :]).tocsr() if _sp.issparse(A) else A * d[None, :]
            if x is not None:
                x[psub] = x[psub] / d[psub]
            scaling['M'] = d

        # MRST's getScalingInternalCPR and the D scaling act on the
        # cell-major reordered system (getCellMajorReordering); the PRSTCore
        # systems are variable-grouped, so reorder for the weights/scaling
        # and map the result back to the original rows.
        strategy = str(self.strategy).lower()
        decoupling = str(self.decoupling).lower()
        perm = None
        w_cm = None
        if ncv > 0 and nc > 0 and ncv <= n:
            perm = _np.arange(ncv, dtype=int).reshape(bz, nc).T.ravel()
            A_cm = A[perm, :][:, perm].tocsr() if _sp.issparse(A) \
                else A[_np.ix_(perm, perm)]
            w_cm = _get_scaling_internal_cpr(
                A_cm, b, bz, nc, decoupling, strategy,
                diagonal_tol=self.diagonalTol,
                coupling_tol=self.couplingTol,
            )
        else:
            w_cm = _get_scaling_internal_cpr(
                A, b, bz, nc, decoupling, strategy,
                diagonal_tol=self.diagonalTol,
                coupling_tol=self.couplingTol,
            )
        w_cm = _np.asarray(w_cm, dtype=float).ravel()

        if strategy in ('mrst', 'mrst_drs'):
            # MRST applyScaling: D replaces each cell block's first
            # (pressure) row by the weighted sum of the block's rows and
            # keeps the remaining rows as identity rows (D*A, D*b).
            if w_cm.size < ncv:
                w_cm = _np.concatenate(
                    [w_cm, _np.ones((ncv - w_cm.size,), dtype=float)])
            D_cm = _build_mrst_scaling_D(w_cm, ndof=ncv, ncv=ncv, bz=bz)
            if _sp.issparse(A) and perm is not None:
                A = A.tocsr()
                PA = A[perm[:ncv], :]                    # ncv x n cell-major rows
                A[perm[:ncv], :] = (D_cm @ PA).tocsr()
                b[perm[:ncv]] = D_cm @ b[perm[:ncv]]
            elif _sp.issparse(A):
                A = (D_cm @ A[:ncv, :]).tocsr()
                b[:ncv] = D_cm @ b[:ncv]
            else:
                A = A.copy()
                if perm is not None:
                    A[perm[:ncv], :] = D_cm @ A[perm[:ncv], :]
                    b[perm[:ncv]] = D_cm @ b[perm[:ncv]]
                else:
                    A[:ncv, :] = D_cm @ A[:ncv, :]
                    b[:ncv] = D_cm @ b[:ncv]
            scaling['D'] = D_cm

            w_override = _np.zeros((ncv,), dtype=float)
            w_override[psub] = 1.0
            self.amgcl_setup['drs_row_weights'] = w_override.tolist()
        elif strategy == 'amgcl_drs':
            self.amgcl_setup['drs_row_weights'] = w_cm[:ncv].tolist()
        elif strategy == 'amgcl':
            self.amgcl_setup['drs_row_weights'] = []

        return A, b, scaling, x

    def undoScaling(self, x, scaling):
        out = _np.asarray(x, dtype=float).ravel().copy()
        d = scaling.get('M', None)
        if d is not None:
            out = _np.asarray(d, dtype=float).ravel() * out
        return out

    def applyScalingAdjoint(self, A, b):
        import scipy.sparse as _sp
        A_s, b_s, scaling, _ = self.applyScaling(A, b)
        D = scaling.get('D', None)
        if D is not None:
            if _sp.issparse(D):
                b_s = _sp.linalg.spsolve(D.tocsc(), b_s)
            else:
                Dv = _np.asarray(D, dtype=float).ravel()
                mask = _np.abs(Dv) > 1e-30
                tmp = b_s.copy()
                tmp[mask] = tmp[mask] / Dv[mask]
                tmp[~mask] = 0.0
                b_s = tmp
        M = scaling.get('M', None)
        if M is not None:
            b_s = _np.asarray(M, dtype=float).ravel() * b_s
        return A_s, b_s, scaling

    def undoScalingAdjoint(self, x, scaling):
        import scipy.sparse as _sp
        out = _np.asarray(x, dtype=float).ravel().copy()
        D = scaling.get('D', None)
        if D is not None:
            if _sp.issparse(D):
                out = D.T.dot(out)
            else:
                out = _np.asarray(D, dtype=float).ravel() * out
        return out

    def solveAdjointProblem(self, problemPrev, problemCurr, nextLambdaVec, objective, model=None, **kwargs):
        A = _np.asarray(problemCurr['Jacobian'], dtype=float)
        b = _np.zeros((A.shape[0],), dtype=float)
        if objective is not None:
            obj = _np.asarray(objective, dtype=float).ravel()
            b[:min(b.size, obj.size)] -= obj[:min(b.size, obj.size)]
        if nextLambdaVec is not None and problemPrev is not None and 'Jacobian' in problemPrev:
            Ap = _np.asarray(problemPrev['Jacobian'], dtype=float)
            nv = _np.asarray(nextLambdaVec, dtype=float).ravel()
            b -= Ap.T.dot(nv)

        A_s, b_s, scaling = self.applyScalingAdjoint(A, b)
        lam, _, rep = self.ellipticSolver.solveLinearProblem({'Jacobian': A_s.T, 'Residuals': -b_s}, model)
        lam = self.undoScalingAdjoint(lam, scaling)
        return lam, lam, rep

    def _solve_two_stage_cpr(self, problem, model=None):
        """Genuine two-stage CPR: the pressure block is preconditioned by
        one compiled-AMGCL V-cycle (``pyamgcl.amgcl.__call__``, a single
        preconditioner application -- not an inner converged solve) and
        the full correction by ILU0, combined into one CPR preconditioner
        for GMRES on the full system. This is what ``CPRSolverAD``'s own
        ``solveLinearProblem`` cannot give: its sparse-matrix branch runs
        ILU0+GMRES directly on the full system with no pressure-specific
        acceleration at all, so ``self.ellipticSolver`` (the compiled AMGCL
        elliptic solver this class configures) was never actually invoked
        for any sparse deck-derived model. Returns ``(dx, relres, report)``
        on a converged solve, or ``None`` to fall back to the parent's
        ILU0+GMRES-on-full-system path (matrix too small/dense for a
        meaningful pressure/other split, pyamgcl unavailable, AMG hierarchy
        build failure, or the preconditioned GMRES itself not converging).
        """
        import time as _time
        import scipy.sparse as _sp
        from scipy.sparse.linalg import LinearOperator, spilu, gmres

        t0 = _time.perf_counter()
        A, b = self._get_system(problem)
        if not _sp.issparse(A):
            return None
        Acsr = A.tocsr().astype(float)
        b = _np.asarray(b, dtype=float).ravel()
        n = Acsr.shape[0]

        pidx = _np.asarray(self._pressure_index(n, problem=problem, model=model), dtype=int)
        if pidx.size == 0 or pidx.size >= n:
            return None

        pyamgcl = self.ellipticSolver._load_pyamgcl()
        if pyamgcl is None:
            return None

        try:
            Ap = (self.ellipticSign * Acsr[pidx, :][:, pidx]).tocsr()
            P = pyamgcl.amgcl(Ap, prm={})
        except Exception:
            return None

        try:
            diagonal = Acsr.diagonal()
            zero_pivot = _np.abs(diagonal) < 1.0e-12
            if _np.any(zero_pivot):
                row_scale = _np.maximum(_np.asarray(_np.abs(Acsr).sum(axis=1)).ravel(), 1.0)
                Awork = (Acsr + _sp.diags(_np.where(zero_pivot, row_scale * 1.0e-12, 0.0))).tocsc()
            else:
                Awork = Acsr.tocsc()
            ilu = spilu(Awork, drop_tol=self.iluDropTol, fill_factor=self.iluFillFactor,
                       permc_spec='COLAMD', diag_pivot_thresh=0.1)
        except Exception:
            return None

        def apply_cpr(r):
            r = _np.asarray(r, dtype=float).ravel()
            rp = self.ellipticSign * r[pidx]
            dp = _np.zeros(n)
            dp[pidx] = _np.asarray(P(rp), dtype=float).ravel()
            r2 = r - Acsr.dot(dp)
            return dp + ilu.solve(r2)

        M = LinearOperator(Acsr.shape, matvec=apply_cpr)
        history = []
        restart = max(1, min(50, int(self.maxIterations)))
        maxiter_cycles = max(1, int(_np.ceil(float(self.maxIterations) / float(restart))))
        try:
            dx, flag = gmres(Acsr, b, M=M, rtol=self.tolerance, atol=0.0,
                             restart=restart, maxiter=maxiter_cycles,
                             callback=lambda residual: history.append(float(residual)),
                             callback_type='pr_norm')
        except Exception:
            return None

        dx = _np.asarray(dx, dtype=float).ravel()
        if not _np.all(_np.isfinite(dx)):
            return None
        rfin = b - Acsr.dot(dx)
        relres = float(_np.linalg.norm(rfin) / max(_np.linalg.norm(b), 1e-30))
        if not (flag == 0 and relres <= self.tolerance):
            return None

        elapsed = float(_time.perf_counter() - t0)
        report = self.getSolveReport(
            Iterations=len(history),
            Residual=relres,
            SolverTime=elapsed,
            LinearSolutionTime=elapsed,
            PreparationTime=0.0,
            PostProcessTime=0.0,
            Converged=True,
            FlagGMRES=int(flag),
            PreconditionerReport={
                'Type': 'AMGCL-CPR-two-stage',
                'PressureBlockSize': int(pidx.size),
            },
        )
        if self.extraReport:
            report['ResidualHistory'] = history
        return dx, relres, report

    def solveLinearProblem(self, problem, model=None):
        problem = self.prepareProblemCPR(problem, model)
        if isinstance(problem, dict) and isinstance(problem.get('State', None), dict):
            p = problem['State'].get('pressure', None)
            if p is not None:
                self.pressureScaling = float(max(1e-30, _abs_mean(p)))

        result = None
        try:
            result = self._solve_two_stage_cpr(problem, model)
        except Exception:
            result = None
        if result is not None:
            dx, relres, report = result
        else:
            dx, relres, report = super().solveLinearProblem(problem, model)
        report['AMGCLCPR'] = {
            'Setup': dict(self.amgcl_setup),
            'PressureScaling': float(self.pressureScaling),
            'UseSYMRCMOrdering': bool(self.useSYMRCMOrdering),
            'ApplyScalingCPR': bool(self.doApplyScalingCPR),
            'ReuseMode': int(self.ellipticSolver.reuseMode),
            'Strategy': str(self.strategy),
            'Decoupling': str(self.decoupling),
            'DRSEpsPS': float(self.amgcl_setup.get('drs_eps_ps', 0.0)),
            'DRSEpsDD': float(self.amgcl_setup.get('drs_eps_dd', 0.0)),
            'UseDRS': bool(self.amgcl_setup.get('use_drs', False)),
            'DRSRowWeightsCount': int(len(self.amgcl_setup.get('drs_row_weights', []))),
        }
        return dx, relres, report


def _abs_mean(values):
    arr = _np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 1.0
    return _np.mean(_np.abs(arr))


def _get_scaling_internal_cpr(A, b, bz, nc, decoupling, strategy, diagonal_tol, coupling_tol):
    """1:1 port of MRST ``AMGCL_CPRSolverAD.getScalingInternalCPR``.

    Returns the per-row weights ``w`` (length ``A.shape[0]``) that build the
    CPR scaling matrix ``D``: the quasi-IMPES weights on the cell unknowns
    (sparse solve of the *transposed* block diagonal, ``Ap' * w = q`` with
    ``q = 1`` on the pressure unknowns) and ones elsewhere; the ``mrst_drs``
    dynamic-row-sum filter (Gries et al, SPE-163608-PA) then zeroes rows that
    fail the diagonal-dominance / weak-coupling checks.  ``A`` must be given
    in cell-major block form (each cell's ``bz`` unknowns contiguous) --
    MRST feeds it the ``getCellMajorReordering``-permuted system.
    """
    import scipy.sparse as _sp

    n = int(A.shape[0])
    ndof = min(n, int(bz * nc))
    p_idx = _np.arange(0, ndof, bz, dtype=int)
    w = _np.ones((n,), dtype=float)
    if not _sp.issparse(A):
        A = _sp.csr_matrix(A)
    A = A.tocsr()

    key = str(decoupling).lower().replace('-', '').replace('_', '')
    if key == 'quasiimpes':
        iis, jjs, vvs = _sp.find(A)
        block_i = iis // bz
        block_j = jjs // bz
        keep = (block_j >= block_i) & (block_j < block_i + 1) \
            & (iis < ndof) & (jjs < ndof)
        # MRST: Ap = sparse(jj, ii, vv) -- the transposed block diagonal --
        # and w = Ap \ q with q = 1 on the pressure unknowns.
        Ap = _sp.csr_matrix((vvs[keep], (jjs[keep], iis[keep])),
                            shape=(ndof, ndof))
        q = _np.zeros((ndof,), dtype=float)
        q[p_idx] = 1.0
        try:
            wp = _np.asarray(_sp.linalg.spsolve(Ap.tocsc(), q),
                             dtype=float).ravel()
        except Exception:
            wp, *_ = _np.linalg.lstsq(Ap.toarray(), q, rcond=None)
        if wp.size == ndof:
            w[:ndof] = wp

    if str(strategy).lower() == 'mrst_drs':
        A = A.tocsr()
        iis, jjs, vvs = _sp.find(A[:ndof, :ndof])
        isdp = (jjs % bz) == 0                    # pressure-column entries
        same_block = (iis // bz) == (jjs // bz)
        isdiag = same_block & isdp
        is_offdiag_p = isdp & ~isdiag             # pressure cols, other blocks

        pd = _np.zeros((ndof,), dtype=float)
        pd[iis[isdiag]] = vvs[isdiag]

        if _np.isfinite(diagonal_tol):
            sum_offdiag = _np.zeros((ndof,), dtype=float)
            _np.add.at(sum_offdiag, iis[is_offdiag_p], _np.abs(vvs[is_offdiag_p]))
            ok_dd = pd >= sum_offdiag * float(diagonal_tol)
        else:
            ok_dd = _np.ones((ndof,), dtype=bool)

        if _np.isfinite(coupling_tol):
            # Same set as is_offdiag_p: pressure columns in other blocks.
            sum_other = _np.zeros((ndof,), dtype=float)
            _np.add.at(sum_other, iis[is_offdiag_p], _np.abs(vvs[is_offdiag_p]))
            ok_ps = sum_other >= pd * float(coupling_tol)
        else:
            ok_ps = _np.ones((ndof,), dtype=bool)

        ok = ok_dd & ok_ps
        cell_ids = _np.arange(ndof) // bz
        ok_count = _np.zeros((int(_np.ceil(ndof / bz)),), dtype=int)
        _np.add.at(ok_count, cell_ids, ok)
        bad = ok_count == 0
        if _np.any(bad):
            # all rows of a block failed: keep the first (MRST keeps the
            # pressure unknown of that cell).
            first = _np.where(bad)[0] * bz
            ok[first] = True
        ok[p_idx] = True
        w[:ndof][~ok] = 0.0
    return w


def _build_mrst_scaling_D(w, ndof, ncv, bz):
    """Build MRST's CPR scaling matrix ``D`` (1:1 with ``applyScaling``).

    ``w`` are the per-row weights (length >= ``ncv``).  In the cell-major
    reordered system each cell block occupies rows ``(i-1)*bz+1 .. i*bz``
    with the pressure unknown first, and MRST builds

        I  = rldecode((1:bz:ncv)', bz)   % pressure row of each block
        J  = (1:ncv)'
        Id = non-pressure cell rows + non-cell rows
        D  = sparse([I; Id], [J; Id], [w(1:ncv); ones(numel(Id),1)])

    i.e. the first (pressure) row of every cell block is *replaced* by the
    weighted sum of the block's rows while the remaining rows stay identity
    rows.  The returned ``D`` left-multiplies the system (``A = D*A``,
    ``b = D*b``), which preserves the solution.
    """
    import scipy.sparse as _sp
    w = _np.asarray(w, dtype=float).ravel()
    ndof = int(ndof)
    ncv = int(min(ncv, ndof))
    bz = max(1, int(bz))

    pressure_rows = _np.arange(0, ncv, bz, dtype=int)     # first row of each block
    I = _np.repeat(pressure_rows, bz)                      # length ncv
    J = _np.arange(ncv, dtype=int)

    id_idx = _np.arange(ncv, dtype=int)
    id_idx = id_idx[id_idx % bz != 0]                      # non-pressure cell rows
    id_idx = _np.concatenate([id_idx, _np.arange(ncv, ndof, dtype=int)])

    rows = _np.concatenate([I, id_idx])
    cols = _np.concatenate([J, id_idx])
    vals = _np.concatenate([w[:ncv], _np.ones(id_idx.size, dtype=float)])
    return _sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
