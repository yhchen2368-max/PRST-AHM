from .amgcl_cpr_solver_ad import AMGCL_CPRSolverAD


class AMGCL_CPRSolverBlockAD(AMGCL_CPRSolverAD):
    """MRST-style AMGCL block CPR solver.

    The scalar ``AMGCL_CPRSolverAD`` path operates on the already assembled
    sparse AD matrix.  MRST's ``AMGCL_CPRSolverBlockAD`` instead reduces
    non-cell unknowns, reorders cell unknowns into cell-major blocks and then
    calls AMGCL's block CPR preconditioner.  This class mirrors that flow and
    uses the optional local ``pyamgcl_block_cpr_capi_ext`` extension for the
    actual AMGCL block solve.
    """

    def __init__(self, blockSize=0, pressureWeight=1.0, tolerance=1e-8,
                 maxIterations=100, verbose=False, strategy='mrst', **kwargs):
        parent_keys = {
            'diagonalTol', 'couplingTol', 'decoupling', 'doApplyScalingCPR',
            'useSYMRCMOrdering', 'extraReport', 'directSolveThreshold',
            'iluDropTol', 'iluFillFactor',
        }
        parent_kwargs = {k: v for k, v in kwargs.items() if k in parent_keys}
        super().__init__(pressureWeight=pressureWeight, tolerance=tolerance,
                         maxIterations=maxIterations, verbose=verbose,
                         blockSize=blockSize, strategy=strategy, **parent_kwargs)
        self.blockSize = int(blockSize)
        mrst_defaults = {
            'solver': 'bicgstab',
            'preconditioner': 'amg',
            'coarsening': 'aggregation',
            'relaxation': 'spai0',
            's_relaxation': 'ilu0',
            'cpr_blocksolver': True,
            'aggr_eps_strong': 0.08,
            'aggr_over_interp': 1.0,
            'aggr_relax': 2.0/3.0,
            'npre': 1,
            'npost': 1,
            'ncycle': 1,
            'direct_coarse': True,
            'coarse_enough': -1,
            'max_levels': -1,
            'ilu_damping': 1.0,
            'iluk_k': 1,
            'ilut_p': 2,
            'ilut_tau': 0.01,
            'gmres_m': 30,
            'reuseMode': False,
            'update_sprecond': False,
            'update_ptransfer': False,
        }
        mrst_defaults.update({k: v for k, v in kwargs.items() if k in mrst_defaults})
        self.amgcl_setup.update(mrst_defaults)
        self.reductionStrategy = str(kwargs.get('reductionStrategy', 'schur'))
        # The exact Schur complement of the well block is what makes the
        # reduced cell solve consistent with the full system; the cheap
        # 'diagonal' approximation leaves a full-system residual O(1) on
        # well-dominated decks (SPE9), so Newton stalls around CNV_O.
        self.schurApproxType = str(kwargs.get('schurApproxType', 'full'))
        self.schurWeight = float(kwargs.get('schurWeight', 1.0))
        self._last_block_prep = {}
        # Set per solve from the Newton iteration number; see _apply_reuse_mode.
        self._reset_amgcl_cache = False

    def _num_cells(self, problem=None, model=None):
        if isinstance(problem, dict) and isinstance(problem.get('State', None), dict):
            p = problem['State'].get('pressure', None)
            if p is not None:
                try:
                    return int(__import__('numpy').asarray(p).size)
                except Exception:
                    pass
        if model is not None:
            grid = getattr(model, 'G', None)
            if isinstance(grid, dict):
                cells = grid.get('cells', {})
                if isinstance(cells, dict):
                    nc = int(cells.get('num', 0))
                    if nc > 0:
                        return nc
        return 0

    def _cell_major_permutation(self, nc, bz):
        import numpy as _np
        return _np.arange(nc * bz, dtype=int).reshape((bz, nc)).T.ravel()

    def _apply_diagonal_schur(self, Acc, bc, Acn, Anc, Ann, bn, nc, bz):
        """Apply MRST AMGCLSolverBlockAD diagonal Schur reduction."""
        import numpy as _np
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _spla

        lu = _spla.splu(Ann.tocsc())
        x_n0 = _np.asarray(lu.solve(_np.asarray(bn, dtype=float).ravel()), dtype=float).ravel()
        bc_red = _np.asarray(bc - Acn.dot(x_n0), dtype=float).ravel()

        schur_type = self.schurApproxType.lower()
        if schur_type in ('full', 'exact'):
            active_rows = _np.unique(Acn.nonzero()[0]).astype(int)
            active_cols = _np.unique(Anc.nonzero()[1]).astype(int)
            if active_rows.size and active_cols.size:
                n_to_c = lu.solve(Anc[:, active_cols].toarray())
                fill = _np.asarray(Acn[active_rows, :].dot(n_to_c), dtype=float)
                rr, cc = _np.nonzero(fill)
                if rr.size:
                    vals = (-float(self.schurWeight) * fill[rr, cc]).astype(float)
                    correction = _sp.coo_matrix(
                        (vals, (active_rows[rr], active_cols[cc])),
                        shape=Acc.shape,
                    ).tocsr()
                    Acc = (Acc + correction).tocsr()
                    self._last_block_prep['SchurActiveCells'] = int(_np.unique(active_rows % int(nc)).size)
                    self._last_block_prep['SchurInsertedValues'] = int(vals.size)
                else:
                    self._last_block_prep['SchurActiveCells'] = 0
                    self._last_block_prep['SchurInsertedValues'] = 0
            else:
                self._last_block_prep['SchurActiveCells'] = 0
                self._last_block_prep['SchurInsertedValues'] = 0
            return Acc, bc_red, lu

        row_cells = _np.unique(Acn.nonzero()[0] % int(nc))
        col_cells = _np.unique(Anc.nonzero()[1] % int(nc))
        active_cells = _np.intersect1d(row_cells, col_cells, assume_unique=False)

        rows = []
        cols = []
        vals = []
        w = float(self.schurWeight)
        if active_cells.size:
            all_cell_cols = _np.concatenate([active_cells + k * int(nc) for k in range(int(bz))]).astype(int)
            solved = lu.solve(Anc[:, all_cell_cols].toarray())
            solved_by_col = {int(col): i for i, col in enumerate(all_cell_cols)}
        else:
            solved = None
            solved_by_col = {}

        for c in active_cells:
            cell_ix = _np.asarray([c + k * nc for k in range(bz)], dtype=int)
            solve_cols = [solved_by_col.get(int(ix), -1) for ix in cell_ix]
            if any(ix < 0 for ix in solve_cols):
                continue
            n_to_c = solved[:, solve_cols]
            fill = _np.asarray(Acn[cell_ix, :].dot(n_to_c), dtype=float)
            if not _np.any(fill):
                continue
            for i in range(bz):
                for j in range(bz):
                    v = -w * float(fill[i, j])
                    if v != 0.0:
                        rows.append(int(cell_ix[i]))
                        cols.append(int(cell_ix[j]))
                        vals.append(v)

        if vals:
            correction = _sp.coo_matrix((vals, (rows, cols)), shape=Acc.shape).tocsr()
            Acc = (Acc + correction).tocsr()
        self._last_block_prep['SchurActiveCells'] = int(active_cells.size)
        self._last_block_prep['SchurInsertedValues'] = int(len(vals))
        return Acc, bc_red, lu

    def _apply_cpr_strategy(self):
        """MRST ``AMGCL_CPRSolverAD.prepareProblemCPR`` strategy switch.

        Maps the four MRST strategies onto the AMGCL setup:
          * 'mrst'      : use_drs=true with eps=-1e8 (AMGCL DRS effectively
                          off; the decoupling is carried by the MRST D
                          scaling in _apply_mrst_cpr_scaling)
          * 'mrst_drs'  : same; the mrst_drs dynamic-row-sum filter runs
                          inside getScalingInternalCPR, not in AMGCL
          * 'amgcl'     : use_drs=false (pure AMGCL aggregation)
          * 'amgcl_drs' : use_drs=true with diagonalTol/couplingTol (AMGCL
                          applies its own DRS with those thresholds)
        """
        import numpy as _np
        setup = self.amgcl_setup
        strategy = str(self.strategy).lower()
        if strategy in ('mrst', 'mrst_drs'):
            setup['use_drs'] = True
            setup['drs_eps_ps'] = -1e8
            setup['drs_eps_dd'] = -1e8
        elif strategy == 'amgcl':
            setup['use_drs'] = False
        elif strategy == 'amgcl_drs':
            setup['use_drs'] = True
            setup['drs_eps_ps'] = float(self.couplingTol) \
                if _np.isfinite(self.couplingTol) else -1e8
            setup['drs_eps_dd'] = float(self.diagonalTol) \
                if _np.isfinite(self.diagonalTol) else -1e8
        else:
            raise ValueError('Unknown CPR strategy %r' % (self.strategy,))
        self._last_block_prep['CPRStrategy'] = strategy
        self._last_block_prep['AMGCLUseDRS'] = bool(setup.get('use_drs', False))
        self._last_block_prep['AMGCLDRSEpsDD'] = float(setup.get('drs_eps_dd', -1e8))
        self._last_block_prep['AMGCLDRSEpsPS'] = float(setup.get('drs_eps_ps', -1e8))

    def _apply_reuse_mode(self, problem):
        """MRST ``AMGCL_CPRSolverAD.prepareProblemCPR``'s reuse rule.

        MRST only reuses the AMG hierarchy when the preconditioner can be
        partially refreshed against the new matrix -- ``update_sprecond`` or
        ``update_ptransfer`` -- and it calls ``resetAMGCL()`` at the start of
        every Newton loop so the hierarchy is rebuilt once per time step
        rather than kept for the whole simulation.  Without that reset the
        hierarchy goes on ageing across time steps, which is not a mode MRST
        ever runs and not one worth measuring against it.

        An explicit ``reuseMode`` with neither update flag keeps whatever the
        caller asked for, and takes no reset -- that is this class's own
        pre-existing knob, left alone.
        """
        setup = self.amgcl_setup
        if setup.get('update_sprecond', False) or setup.get('update_ptransfer', False):
            setup['reuseMode'] = True
            iteration_no = None
            if isinstance(problem, dict):
                iteration_no = problem.get('iterationNo', None)
            if iteration_no is None:
                iteration_no = getattr(problem, 'iterationNo', None)
            self._reset_amgcl_cache = (iteration_no == 1)
        else:
            self._reset_amgcl_cache = False
        self._last_block_prep['AMGCLReuse'] = bool(setup.get('reuseMode', False))
        self._last_block_prep['AMGCLResetCache'] = bool(self._reset_amgcl_cache)

    def _apply_mrst_cpr_scaling(self, A, b, nc, bz):
        """MRST ``AMGCL_CPRSolverAD.applyScaling`` on the reduced cell system.

        Operates on the cell-major reordered reduced system (A, b).  For the
        'mrst'/'mrst_drs' strategies the quasi-IMPES / true-IMPES / none row
        weights from getScalingInternalCPR are applied through MRST's D
        matrix (each block's first pressure row is replaced by the weighted
        sum of the block rows, the others stay identity) and AMGCL is handed
        a ones-on-pressure row-weight override (the MRST D already carries
        the decoupling).  For 'amgcl_drs' the weights are passed as the AMGCL
        DRS row weights.  For 'amgcl' nothing is scaled.
        """
        import numpy as _np
        from .amgcl_cpr_solver_ad import _get_scaling_internal_cpr, _build_mrst_scaling_D
        strategy = str(self.strategy).lower()
        decoupling = str(self.decoupling).lower()
        if strategy in ('mrst', 'mrst_drs'):
            w = _get_scaling_internal_cpr(
                A, b, int(bz), int(nc), decoupling, strategy,
                diagonal_tol=self.diagonalTol,
                coupling_tol=self.couplingTol,
            )
            w = _np.asarray(w, dtype=float).ravel()
            ncv = A.shape[0]
            D = _build_mrst_scaling_D(w, ndof=ncv, ncv=ncv, bz=bz)
            A = (D @ A).tocsr()
            b = D @ b
            override = _np.zeros((ncv,), dtype=float)
            override[::bz] = 1.0
            self.amgcl_setup['drs_row_weights'] = override.tolist()
            self._last_block_prep['MRSTRowWeighted'] = True
            self._last_block_prep['MRSTRowWeightZeroCount'] = int(
                _np.count_nonzero(_np.abs(w) < 1.0e-30))
        elif strategy == 'amgcl_drs':
            w = _get_scaling_internal_cpr(
                A, b, int(bz), int(nc), 'quasiimpes', 'amgcl',
                diagonal_tol=self.diagonalTol,
                coupling_tol=self.couplingTol,
            )
            self.amgcl_setup['drs_row_weights'] = \
                _np.asarray(w, dtype=float).ravel().tolist()
        else:  # 'amgcl'
            self.amgcl_setup['drs_row_weights'] = []
        return A, b

    def _solve_reduced_block_cpr(self, Acell_cm, bcell_cm, nc, bz, cell_major=True):
        import numpy as _np
        from PRSTCore.solvers.linearsolvers.pyamgcl import pyamgcl_block_cpr_capi_ext

        if not cell_major:
            perm = self._cell_major_permutation(nc, bz)
            Acell_cm = Acell_cm[perm, :][:, perm].tocsr()
            bcell_cm = _np.asarray(bcell_cm, dtype=float).ravel()[perm]
        conv0 = __import__('time').perf_counter()
        bptr, bcol, bval = self._scalar_csr_to_bcsr(Acell_cm, bz)
        conv_time = float(__import__('time').perf_counter() - conv0)

        reuse_flags = 0
        if bool(self.amgcl_setup.get('reuseMode', False)):
            reuse_flags |= 1
        if bool(self.amgcl_setup.get('update_sprecond', False)):
            reuse_flags |= 2
        if bool(self.amgcl_setup.get('update_ptransfer', False)):
            reuse_flags |= 4
        # resetAMGCL(): drop the cached hierarchy so this solve rebuilds it.
        if self._reset_amgcl_cache:
            reuse_flags |= 8
        out = pyamgcl_block_cpr_capi_ext.solve_bcsr_block_cpr(
            bptr,
            bcol,
            bval,
            _np.asarray(bcell_cm, dtype=float),
            int(bz),
            float(self.tolerance),
            int(self.maxIterations),
            0,
            str(self.amgcl_setup.get('solver', 'bicgstab')),
            str(self.amgcl_setup.get('coarsening', 'aggregation')),
            str(self.amgcl_setup.get('relaxation', 'spai0')),
            str(self.amgcl_setup.get('s_relaxation', 'ilu0')),
            bool(self.amgcl_setup.get('use_drs', False)),
            float(self.amgcl_setup.get('drs_eps_dd', 0.2)),
            float(self.amgcl_setup.get('drs_eps_ps', 0.02)),
            float(self.amgcl_setup.get('aggr_eps_strong', 0.08)),
            float(self.amgcl_setup.get('aggr_over_interp', 1.0)),
            float(self.amgcl_setup.get('aggr_relax', 2.0/3.0)),
            int(self.amgcl_setup.get('npre', 1)),
            int(self.amgcl_setup.get('npost', 1)),
            int(self.amgcl_setup.get('ncycle', 1)),
            bool(self.amgcl_setup.get('direct_coarse', True)),
            int(self.amgcl_setup.get('coarse_enough', -1)),
            int(self.amgcl_setup.get('max_levels', -1)),
            float(self.amgcl_setup.get('ilu_damping', 1.0)),
            int(self.amgcl_setup.get('iluk_k', 1)),
            int(self.amgcl_setup.get('ilut_p', 2)),
            float(self.amgcl_setup.get('ilut_tau', 0.01)),
            int(self.amgcl_setup.get('gmres_m', 30)),
            int(reuse_flags),
        )
        if len(out) >= 6:
            x_cm, iters, err, kernel_time, did_setup, update_time = out
        else:
            x_cm, iters, err, kernel_time = out
            did_setup = True
            update_time = 0.0
        x_cm = _np.asarray(x_cm, dtype=float).ravel()
        self._last_block_prep['AMGCLDidSetup'] = bool(did_setup)
        self._last_block_prep['AMGCLPartialUpdateTime'] = float(update_time)
        if cell_major:
            return x_cm, int(iters), float(err), float(kernel_time), conv_time
        x_grouped = _np.empty_like(x_cm)
        x_grouped[perm] = x_cm
        return x_grouped, int(iters), float(err), float(kernel_time), conv_time

    def _scalar_csr_to_bcsr(self, A, bz):
        """Convert cell-major scalar CSR into MRST-style BCSR buffers."""
        import numpy as _np

        bsr = A.tobsr(blocksize=(int(bz), int(bz)))
        bsr.sort_indices()
        return (
            _np.asarray(bsr.indptr, dtype=_np.int32),
            _np.asarray(bsr.indices, dtype=_np.int32),
            _np.asarray(bsr.data, dtype=float).reshape((-1, int(bz) * int(bz))).ravel(order='C'),
        )

    def solveLinearProblem(self, problem, model=None):
        import numpy as _np
        import time as _time
        import scipy.sparse as _sp

        t0 = _time.perf_counter()
        A, b = self._get_system(problem)
        if not _sp.issparse(A):
            return super().solveLinearProblem(problem, model)
        A = A.tocsr()
        b = _np.asarray(b, dtype=float).ravel()
        n = int(A.shape[0])
        nc = self._num_cells(problem=problem, model=model)
        bz = int(self.blockSize or self.amgcl_setup.get('block_size', 0) or 0)
        if bz <= 0:
            # Auto-derive the per-cell block size: it must match the model's
            # number of cell components.  A fixed default of 2 on a 3-phase
            # deck (SPE9.DATA: n = 3*nc + wells) pushed ~9k cell unknowns
            # into the "non-cell" Schur block, so every solve factored a
            # 9104-by-9104 splu (~10 s) before the (fast) AMGCL kernel even
            # ran.  Prefer the model, fall back to the system's integer cell
            # ratio for the grouped primary-variable ordering.
            try:
                from .select_linear_solver_ad import get_component_count
                bz = int(get_component_count(model)) if model is not None else 0
            except Exception:
                bz = 0
            if bz <= 0 and nc > 0:
                bz = max(1, int(round(n / nc)))
            if bz <= 0:
                return super().solveLinearProblem(problem, model)
            self.blockSize = bz
            self.amgcl_setup['block_size'] = bz
        ncv = int(nc * bz)
        if nc <= 0 or ncv <= 0 or ncv > n:
            return super().solveLinearProblem(problem, model)

        self._last_block_prep = {}
        # MRST AMGCL_CPRSolverAD.prepareProblemCPR: strategy -> AMGCL setup
        # (use_drs / drs_eps), then the trueIMPES equation weighting that
        # MRST applies to each cell equation before the well reduction.
        self._apply_cpr_strategy()
        self._apply_reuse_mode(problem)
        # Rebind: the weighting builds a *new* scaled matrix rather than
        # editing this one in place.  Dropping the return left the solve
        # running with decoupling='trueIMPES' but no weighting applied,
        # which is MRST's 'none' wearing another name -- SPE10 model 2
        # gave byte-identical residuals for the two.
        A, b = self._apply_true_impes_scaling(A, b, nc, bz, problem, model)

        prep0 = _time.perf_counter()
        Acc = A[:ncv, :ncv].tocsr()
        bc = b[:ncv].copy()
        lu = None
        Anc = None
        has_noncell = n > ncv
        if self.verbose:
            print(f"      block-cpr prep: n={n}, cells={nc}, block={bz}, noncell={n - ncv}")
        if has_noncell:
            Acn = A[:ncv, ncv:].tocsr()
            Anc = A[ncv:, :ncv].tocsr()
            Ann = A[ncv:, ncv:].tocsc()
            bn = b[ncv:].copy()
            if self.verbose:
                print(f"      block-cpr schur: Ann={Ann.shape}, Acn nnz={Acn.nnz}, Anc nnz={Anc.nnz}")
            Acc, bc, lu = self._apply_diagonal_schur(Acc, bc, Acn, Anc, Ann, bn, nc, bz)
        tprep = float(_time.perf_counter() - prep0)
        if self.verbose:
            print(f"      block-cpr prep done: {tprep:.2f}s, Acc nnz={Acc.nnz}")

        try:
            # MRST reorders the reduced system to cell-major before the CPR
            # scaling/AMGCL solve; do the same here.  Row scaling (D*A, D*b)
            # for mrst/mrst_drs preserves the solution.
            perm = self._cell_major_permutation(nc, bz)
            Acc_cm = Acc[perm, :][:, perm].tocsr()
            bc_cm = _np.asarray(bc, dtype=float).ravel()[perm]
            Acc_cm, bc_cm = self._apply_mrst_cpr_scaling(Acc_cm, bc_cm, nc, bz)
            solve0 = _time.perf_counter()
            if self.verbose:
                print("      block-cpr kernel: AMGCL block CPR start")
            xcell_cm, iters, kernel_res, kernel_time, conv_time = \
                self._solve_reduced_block_cpr(Acc_cm, bc_cm, nc, bz, cell_major=True)
            xcell = _np.empty_like(xcell_cm)
            xcell[perm] = xcell_cm
            tsolve = float(_time.perf_counter() - solve0)
            if self.verbose:
                print(f"      block-cpr kernel done: wall={tsolve:.2f}s, kernel={kernel_time:.2f}s, iters={iters}, err={kernel_res:.3e}")
            dx = _np.zeros(n, dtype=float)
            dx[:ncv] = xcell
            if has_noncell:
                dx[ncv:] = _np.asarray(lu.solve(b[ncv:] - Anc.dot(xcell)), dtype=float).ravel()
            rcell = bc - Acc.dot(xcell)
            reduced_relres = float(_np.linalg.norm(rcell) / max(_np.linalg.norm(bc), 1.0e-30))
            rfin = b - A.dot(dx)
            full_relres = float(_np.linalg.norm(rfin) / max(_np.linalg.norm(b), 1.0e-30))
            relres = float(kernel_res)
            elapsed = float(_time.perf_counter() - t0)
            report = self.getSolveReport(
                Iterations=iters,
                Residual=relres,
                SolverTime=elapsed,
                LinearSolutionTime=tsolve,
                PreparationTime=tprep,
                PostProcessTime=max(0.0, elapsed - tprep - tsolve),
                Converged=bool(_np.all(_np.isfinite(dx)) and relres <= self.tolerance),
                FlagGMRES=0 if relres <= self.tolerance else 1,
                PreconditionerReport={
                    'Type': 'amgcl-block-cpr-mrst',
                    'Kernel': 'pyamgcl_block_cpr_capi_ext',
                    'KernelResidualEstimate': kernel_res,
                    'ReducedSystemResidual': reduced_relres,
                    'FullSystemResidual': full_relres,
                    'KernelTime': kernel_time,
                    'BlockConversionTime': conv_time,
                    'BlockSize': bz,
                    'Cells': nc,
                    'CellUnknowns': ncv,
                    'NonCellUnknowns': int(n - ncv),
                    'ReductionStrategy': self.reductionStrategy,
                    'SchurApproxType': self.schurApproxType,
                    **dict(self._last_block_prep),
                    'Setup': dict(self.amgcl_setup),
                },
            )
            return dx, relres, report
        except Exception as exc:
            if self.verbose:
                print(f"      block-cpr fallback: {exc}")
            dx, res, report = super().solveLinearProblem(problem, model)
            if isinstance(report, dict):
                report['BlockCPRFallbackReason'] = str(exc)
            return dx, res, report
