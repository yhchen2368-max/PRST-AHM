import numpy as _np
import time as _time

from .linear_solver_ad import LinearSolverAD


class AMGCLSolverAD(LinearSolverAD):
    """AMGCL-style AD linear solver scaffold with MRST-like options."""

    def __init__(self, tolerance=1e-6, maxIterations=100, verbose=False,
                 reuseMode=1, extraReport=False, usePyAMGCL=True, **kwargs):
        super().__init__(verbose=verbose, tolerance=tolerance,
                         maxIterations=maxIterations, extraReport=extraReport,
                         reduceToCell=True)
        self.reuseMode = int(reuseMode)
        self.usePyAMGCL = bool(usePyAMGCL)
        self._pyamgcl_error = None
        self._cached_preconditioner = None
        self._cached_solver = None
        self._cache_build_count = 0
        self._cache_reuse_count = 0
        self.amgcl_setup = {
            'solver': kwargs.get('solver', 'bicgstab'),
            'preconditioner': kwargs.get('preconditioner', 'amg'),
            'relaxation': kwargs.get('relaxation', 'spai0'),
            'coarsening': kwargs.get('coarsening', 'smoothed_aggregation'),
            'block_size': int(kwargs.get('block_size', 1)),
            'update_sprecond': bool(kwargs.get('update_sprecond', False)),
            'update_ptransfer': bool(kwargs.get('update_ptransfer', False)),
        }

    def setParameterGroup(self, group, value):
        self.amgcl_setup[str(group).lower()] = value

    def resetAMGCL(self):
        self._cached_preconditioner = None
        self._cached_solver = None

    def cleanupSolver(self, A=None, b=None, **kwargs):
        # MRST cleanupSolver() for reuseMode>1 resets MEX internal state.
        if self.reuseMode > 1:
            self.resetAMGCL()
        return self

    def _load_pyamgcl(self):
        if not self.usePyAMGCL:
            return None
        try:
            from PRSTCore.solvers.linearsolvers import pyamgcl as _pyamgcl
            return _pyamgcl
        except Exception as exc:
            self._pyamgcl_error = str(exc)
            return None

    def _build_prm(self):
        # pyamgcl uses a property tree, where string values are accepted.
        prm = {}
        setup = self.amgcl_setup
        if 'solver' in setup:
            prm['solver.type'] = str(setup['solver'])
        if 'preconditioner' in setup:
            prm['precond.class'] = str(setup['preconditioner'])
        if 'relaxation' in setup:
            prm['relax.type'] = str(setup['relaxation'])
        if 'coarsening' in setup:
            prm['precond.coarsening.type'] = str(setup['coarsening'])
        if 'block_size' in setup:
            prm['precond.block_size'] = int(setup['block_size'])
        return prm

    def _solve_with_pyamgcl(self, A, b):
        pyamgcl = self._load_pyamgcl()
        if pyamgcl is None:
            return None

        import scipy.sparse as _sp
        Asp = _sp.csr_matrix(A)
        prm = self._build_prm()
        # The local extension receives preconditioner parameters in amgcl()
        # and solver parameters separately in solver(); do not pass the
        # flattened solver/precond wrapper keys into the preconditioner.
        # This bundled extension uses its compiled default AMG hierarchy;
        # unsupported runtime preconditioner subtrees generate warnings.
        precond_prm = {}
        use_cache = self.reuseMode > 1
        reused = False

        if not use_cache or self._cached_preconditioner is None or self._cached_solver is None:
            P = pyamgcl.amgcl(Asp, prm=precond_prm)
            S = pyamgcl.solver(P, prm={'type': str(self.amgcl_setup.get('solver', 'bicgstab'))})
            if use_cache:
                self._cached_preconditioner = P
                self._cached_solver = S
                self._cache_build_count += 1
        else:
            S = self._cached_solver
            reused = True
            self._cache_reuse_count += 1

        if use_cache and reused:
            # MRST reuseMode=2: reuse preconditioner but solve potentially updated matrix.
            x = _np.asarray(S(Asp, b), dtype=float).ravel()
        else:
            x = _np.asarray(S(b), dtype=float).ravel()
        iters = int(getattr(S, 'iters', 1))
        error = float(getattr(S, 'error', _np.nan))
        relres = float(_np.linalg.norm(A.dot(x) - b) / max(_np.linalg.norm(b), 1e-30))
        if not _np.isfinite(error):
            error = relres
        return x, iters, error, relres, reused

    def solveLinearProblem(self, problem, model=None):
        t0 = _time.perf_counter()
        A, b = self._get_system(problem)
        residual_history = []
        iter_count = 0
        backend = 'scipy/dense-fallback'

        solved = self._solve_with_pyamgcl(A, b)
        if solved is not None:
            x, iter_count, amgcl_err, relres, reused = solved
            residual_history.extend([amgcl_err, relres])
            backend = 'pyamgcl'
            cache_action = 'reuse' if reused else 'build'
        else:
            cache_action = 'fallback'
            try:
                import scipy.sparse as _sp
                import scipy.sparse.linalg as _spla
                Asp = _sp.csr_matrix(A)
                if str(self.amgcl_setup.get('solver', '')).lower() == 'gmres':
                    x, info = _spla.gmres(Asp, b, rtol=self.tolerance, atol=0.0,
                                          maxiter=self.maxIterations)
                else:
                    x, info = _spla.bicgstab(Asp, b, rtol=self.tolerance, atol=0.0,
                                             maxiter=self.maxIterations)
                relres = float(_np.linalg.norm(A.dot(x) - b) / max(_np.linalg.norm(b), 1e-30))
                iter_count = self.maxIterations if info and info > 0 else 1
                residual_history.append(relres)
                if info < 0:
                    x = _np.linalg.solve(A, b)
                    relres = float(_np.linalg.norm(A.dot(x) - b) / max(_np.linalg.norm(b), 1e-30))
                    residual_history.append(relres)
            except Exception:
                x = _np.linalg.solve(A, b)
                relres = float(_np.linalg.norm(A.dot(x) - b) / max(_np.linalg.norm(b), 1e-30))
                residual_history.append(relres)
                iter_count = 1

        converged = relres <= self.tolerance
        elapsed = float(_time.perf_counter() - t0)
        report = self.getSolveReport(
            Iterations=int(iter_count),
            Residual=relres,
            SolverTime=elapsed,
            LinearSolutionTime=elapsed,
            Converged=converged,
            ReuseMode=self.reuseMode,
            Backend=backend,
            CacheAction=cache_action,
            CacheBuildCount=int(self._cache_build_count),
            CacheReuseCount=int(self._cache_reuse_count),
            AMGCLSetup=dict(self.amgcl_setup),
        )
        if self._pyamgcl_error:
            report['PyAMGCLError'] = self._pyamgcl_error
        if self.extraReport:
            report['ResidualHistory'] = residual_history
        return _np.asarray(x, dtype=float).ravel(), relres, report
