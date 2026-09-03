"""Linear solver backed by PETSc's KSP, with a pressure/rest field split.

A black-oil Jacobian is not one problem but two stuck together.  The
pressure unknowns carry an elliptic operator, which multigrid solves in a
handful of iterations and a direct factorisation solves in time growing
faster than the square of the unknown count.  Everything else -- saturations,
compositions, well unknowns -- is hyperbolic and local, and an incomplete
factorisation handles it.  Applying one method to the whole system means
paying the elliptic price on the hyperbolic part, or the reverse.

That observation is CPR, and it is what MRST's ``CPRSolverAD`` and
``AMGCL_CPRSolverAD`` implement.  This module expresses it with PETSc's
preconditioner algebra instead of by hand, in two shapes:

``'cpr'``
    ``PCCOMPOSITE`` in multiplicative mode over two stages: a
    ``PCFIELDSPLIT`` restricted to the pressure field, preconditioned by
    algebraic multigrid, followed by an incomplete factorisation of the
    *whole* system.  This is the classical two-stage CPR and the closest of
    the two to what MRST applies.

``'fieldsplit'``
    a single ``PCFIELDSPLIT`` in multiplicative mode over the pressure and
    non-pressure fields, with multigrid on the first and an incomplete
    factorisation on the second.  A block Gauss-Seidel rather than a true
    two-stage method: the second stage sees only its own block, so it
    cannot correct the coupling the first stage left behind.  Cheaper per
    iteration, and on well-scaled systems it converges in about as many.

Both keep their ``KSP`` between calls.  Building a multigrid hierarchy is
most of the cost of a solve, and consecutive Newton iterations differ in the
matrix entries but not in its sparsity pattern, so PETSc is told the pattern
is unchanged and reuses the setup.

Everything here is sequential (``COMM_SELF``).  The reason to go through
PETSc rather than a bespoke preconditioner is that distributing it later is
a change of communicator and matrix type, not a rewrite.
"""

from __future__ import annotations

import time as _time

import numpy as _np

from .linear_solver_ad import LinearSolverAD

try:
    import scipy.sparse as _sp
except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere
    _sp = None


def check_petsc():
    """Whether petsc4py is importable in this interpreter."""
    try:
        import petsc4py  # noqa: F401
        from petsc4py import PETSc  # noqa: F401
        return True
    except Exception:
        return False


def petsc_has_preconditioner(name):
    """Whether this PETSc build was configured with ``name``.

    Optional packages -- hypre above all -- are a configure-time choice, and
    a build without one raises only when the preconditioner is first set up,
    which is deep inside the first Newton iteration of a long run.  Asking up
    front turns that into a fallback.
    """
    if not check_petsc():
        return False
    from petsc4py import PETSc

    matrix = PETSc.Mat().createAIJ(size=(2, 2), comm=PETSc.COMM_SELF)
    matrix.setUp()
    matrix.setValue(0, 0, 2.0)
    matrix.setValue(1, 1, 2.0)
    matrix.assemble()
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(matrix)
    try:
        pc = ksp.getPC()
        if name == 'hypre':
            pc.setType('hypre')
            pc.setHYPREType('boomeramg')
        else:
            pc.setType(name)
        ksp.setUp()
        return True
    except Exception:
        return False
    finally:
        ksp.destroy()
        matrix.destroy()


class PETScSolverAD(LinearSolverAD):
    """Solve the linearised system with PETSc, splitting off the pressure.

    Parameters
    ----------
    strategy : {'cpr', 'fieldsplit'}
        Which preconditioner shape to build; see the module docstring.
    pressure_precond : {'gamg', 'hypre', 'ilu', 'lu'}
        What preconditions the pressure block.  ``'hypre'`` uses BoomerAMG
        and falls back to ``'gamg'`` when this PETSc lacks it.
    second_stage : str
        The full-system smoother applied after the pressure stage.
        ``'ilu'`` is the usual CPR choice; ``'bjacobi'`` and ``'sor'`` are
        the cheaper alternatives worth measuring against it, since on a
        large system the second stage is applied once per Krylov iteration
        and the pressure stage is too.
    second_stage_options : dict, optional
        Passed to the second stage.  ``levels`` and ``ordering`` are
        understood for ``'ilu'``/``'icc'``/``'lu'``/``'cholesky'``
        (defaulting to ILU(0) in natural order, stated rather than
        inherited); any other key is forwarded to the PC's own setter, so
        ``{'pc_sor_omega': 1.0, 'pc_sor_its': 2}`` reaches ``'sor'``.
    ksp_type : str
        Krylov method for the outer solve.  ``'fgmres'`` by default because
        both preconditioners above are allowed to vary between applications,
        which the flexible variant tolerates and plain GMRES does not.
    """

    def __init__(self, strategy='cpr', pressure_precond='gamg',
                 second_stage='ilu', second_stage_options=None,
                 ksp_type='fgmres', tolerance=1e-4,
                 maxIterations=100, verbose=False, reuse_setup=True,
                 decoupling='quasiIMPES', decoupling_strategy='mrst',
                 diagonalTol=1e-2, couplingTol=0.0,
                 decouple=True, equilibrate=False, **kwargs):
        super().__init__(tolerance=tolerance, maxIterations=maxIterations,
                         verbose=verbose, **kwargs)
        if not check_petsc():
            raise ImportError(
                'petsc4py is not importable; PETScSolverAD cannot be used')
        if strategy not in ('cpr', 'fieldsplit'):
            raise ValueError('strategy must be "cpr" or "fieldsplit", got %r' % strategy)
        self.strategy = strategy
        self.pressure_precond = self._resolve_pressure_precond(pressure_precond)
        self.second_stage = second_stage
        # Options for whatever the second stage is; see
        # _configure_second_stage.  ``levels`` and ``ordering`` are
        # understood for the factorisations, anything else is forwarded to
        # the matching PC setter.
        self.second_stage_options = dict(second_stage_options or {})
        self.ksp_type = ksp_type
        self.reuse_setup = bool(reuse_setup)
        # Which weights combine the component balances into a pressure
        # equation; see PRSTCore.ad_core.cpr_decoupling. MRST's default is
        # trueIMPES, which needs the model's fluid state; quasiIMPES works
        # from the matrix alone and is the safe default for a solver that
        # may be handed a bare system.
        self.decoupling = str(decoupling)
        # MRST spells this strategy on AMGCL_CPRSolverAD, but that name
        # is already taken here by the preconditioner's shape, and one
        # attribute cannot mean both.
        self.decoupling_strategy = str(decoupling_strategy)
        self.diagonalTol = float(diagonalTol)
        self.couplingTol = float(couplingTol)
        # Off only for a system that is already elliptic in its leading
        # block, or to demonstrate what the decoupling is worth.
        self.decouple = bool(decouple)
        # Row equilibration of the decoupled system, off by default.
        #
        # It rescues a system whose decoupled entries span twenty orders of
        # magnitude, which is why it was added -- but it scales every row by
        # its own largest entry, and that is only harmless while the largest
        # entry is the diagonal.  quasiIMPES makes the pressure coefficient
        # exactly one, so it is; trueIMPES makes it about 1e-8, so it is
        # not, and equilibrating leaves the pressure block with off-diagonal
        # entries larger than its diagonal -- multigrid on that converges
        # nowhere.  Measured on SPE9: trueIMPES takes 16 iterations without
        # it and stalls at 200 with it, while quasiIMPES is unaffected.
        self.equilibrate = bool(equilibrate)

        # Cached PETSc objects; rebuilt when the system's shape or sparsity
        # pattern changes, reused when only the values do.
        self._ksp = None
        self._mat = None
        self._rhs = None
        self._sol = None
        self._pattern = None
        self._indptr = None
        self._indices = None
        self.lastIterations = 0
        self.lastConvergedReason = None
        self.lastDecoupling = None
        self._precond_fallback_used = False

    @staticmethod
    def _resolve_pressure_precond(requested):
        if requested == 'hypre' and not petsc_has_preconditioner('hypre'):
            # Not an error: a PETSc without hypre still has its own
            # multigrid, and silently solving nothing would be worse than
            # solving it slightly differently.
            return 'gamg'
        return requested

    # ------------------------------------------------------------------
    # PETSc object management
    # ------------------------------------------------------------------
    def _pattern_key(self, A):
        """What must be unchanged for a cached setup to stay valid."""
        return (A.shape, int(A.nnz), int(A.indptr[-1]),
                hash(A.indices.tobytes()) if A.nnz else 0)

    def _decoupling_operator(self, A, nc, ncomp, problem=None, model=None):
        """The row combination that turns component balances into a pressure equation.

        Without it there is no elliptic block to give multigrid.  A
        deck-derived system stores its equations grouped by component --
        every cell's water balance, then every cell's oil balance, then gas
        -- and its unknowns grouped by variable, pressure first.  So the
        leading nc by nc corner is the water balance differentiated
        with respect to pressure, not a pressure operator, and a field split
        taken on indices alone hands multigrid a matrix it cannot help.
        Measured on SPE9: 200 iterations, relative residual 0.8.

        The weights themselves come from
        :mod:, which carries MRST's three
        strategies.  trueIMPES needs the model's fluid state, so it
        falls back to quasiIMPES when this solver was handed a bare
        matrix -- with a note on the report rather than in silence, since
        the two are different preconditioners.
        """
        from PRSTCore.ad_core import cpr_decoupling as _cpr

        state = None
        if isinstance(problem, dict):
            state = problem.get('State')
        requested = _cpr._normalise(self.decoupling)
        used = requested
        if requested in ('trueimpes', 'simple') and (model is None or state is None):
            used = 'quasiimpes'
        try:
            weights = _cpr.decoupling_weights(used, A, nc, ncomp,
                                              model=model, state=state)
        except Exception:
            if used == 'quasiimpes':
                raise
            used = 'quasiimpes'
            weights = _cpr.quasi_impes_weights(A, nc, ncomp)
        if _cpr._normalise(self.decoupling_strategy) == 'mrstdrs':
            weights = _cpr.apply_dynamic_row_sum(
                weights, A, nc, ncomp,
                diagonal_tol=self.diagonalTol, coupling_tol=self.couplingTol)
        self.lastDecoupling = used
        return _cpr.decoupling_operator(weights, A.shape[0], nc, ncomp)

    @staticmethod
    def _equilibrate_rows(A, b):
        """Scale each equation so its largest coefficient is one.

        Multiplying row ``i`` of ``A`` and entry ``i`` of ``b`` by the same
        number leaves the solution untouched, so this is free accuracy: it
        only decides what the preconditioner sees.  What it sees matters.
        The decoupling weights are bounded but not small -- on SPE10 model 2
        some cells need weights of order 1e14 to combine balances whose
        transmissibilities reach 1e-21 -- and the resulting spread made
        PETSc's incomplete factorisation overflow.  The failure surfaced as
        a solve that stopped at iteration zero with a non-finite residual,
        while the assembled matrix handed to it was entirely finite.

        This equilibrates every row on the same footing, which is the
        difference between it and normalising the weights: that rescales
        only the pressure equations, leaving them inconsistent with the
        saturation ones, and cost SPE10 model 1 three times its iteration
        count.
        """
        scale = _np.abs(A).max(axis=1).toarray().ravel()
        scale[scale <= 0.0] = 1.0
        inverse = 1.0 / scale
        scaled = _sp.diags(inverse, 0, shape=A.shape, format='csr') @ A
        return scaled.tocsr(), b * inverse

    @staticmethod
    def _with_explicit_diagonal(A):
        """``A`` with a stored entry on every diagonal position.

        PETSc's incomplete factorisation refuses a matrix whose diagonal has
        structurally absent entries ("Matrix is missing diagonal entries"),
        and a black-oil Jacobian has several: a well closure equation
        constrains other unknowns rather than its own, and the cell rows for
        the ``x`` variable lose their diagonal wherever the phase-state flag
        that selects Sg, Rs or Rv switches them off.

        The missing positions are spliced in through COO, which keeps an
        explicitly stored zero.  Sparse *addition* does not: scipy prunes
        the result, so ``A + diags(zeros)`` comes back with exactly the
        pattern it started with and PETSc raises again, one Newton
        iteration later and with nothing to show what was tried.
        """
        n = A.shape[0]
        rows = _np.repeat(_np.arange(n), _np.diff(A.indptr))
        present = _np.zeros(n, dtype=bool)
        present[rows[A.indices == rows]] = True
        if present.all():
            return A
        missing = _np.flatnonzero(~present)
        coo = A.tocoo()
        padded = _sp.csr_matrix(
            (_np.concatenate([coo.data, _np.zeros(missing.size)]),
             (_np.concatenate([coo.row, missing]),
              _np.concatenate([coo.col, missing]))),
            shape=A.shape)
        padded.sort_indices()
        return padded

    def _pressure_index(self, n, problem=None, model=None):
        """The rows holding cell pressures.

        Deck-derived systems are stored grouped by primary variable -- every
        cell pressure, then every cell saturation, then the well unknowns --
        so the pressure block is the leading ``nc`` rows.  Getting this wrong
        does not raise; it builds multigrid for a block that is not elliptic
        and the solve stops converging.
        """
        if isinstance(problem, dict) and isinstance(problem.get('State'), dict):
            pressure = problem['State'].get('pressure')
            if pressure is not None:
                nc = int(_np.asarray(pressure).size)
                if 0 < nc <= n:
                    return _np.arange(nc, dtype=int)
        if model is not None:
            grid = getattr(model, 'G', None)
            if isinstance(grid, dict):
                cells = grid.get('cells', {})
                if isinstance(cells, dict):
                    nc = int(cells.get('num', 0))
                    if 0 < nc <= n:
                        return _np.arange(nc, dtype=int)
        raise ValueError(
            'PETScSolverAD needs the number of cells to locate the pressure '
            'block; pass a model or a problem carrying State["pressure"]')

    @staticmethod
    def _component_count(n, nc, problem=None, model=None):
        """How many cell balances the system holds per cell.

        Read off the equation names the assembly already records, so a
        two-phase run is not decoupled as if it had three.  Falling back to
        the row count over the cell count is right whenever the well
        unknowns are a small remainder, which they are.
        """
        if isinstance(problem, dict):
            types = problem.get('types')
            if types is not None:
                cell_rows = sum(1 for t in types if t == 'cell')
                if nc and cell_rows % nc == 0 and cell_rows:
                    return cell_rows // nc
        return max(1, n // nc) if nc else 1

    def _build(self, A, pressure_rows):
        from petsc4py import PETSc

        n = A.shape[0]
        comm = PETSc.COMM_SELF
        # Kept so a later solve on the same pattern can push new values in
        # without rebuilding: PETSc holds its own copy of the index arrays,
        # and re-reading them out of the matrix each time to hand them
        # straight back is both slower and easy to get wrong.
        self._indptr = A.indptr.astype(PETSc.IntType)
        self._indices = A.indices.astype(PETSc.IntType)
        mat = PETSc.Mat().createAIJ(
            size=A.shape, comm=comm,
            csr=(self._indptr, self._indices, A.data.astype(float)))
        mat.assemble()

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(mat)
        ksp.setType(self.ksp_type)
        ksp.setTolerances(rtol=float(self.tolerance), atol=1e-50,
                          max_it=int(self.maxIterations))
        # The initial guess is zero and the right-hand side is a Newton
        # update, so there is nothing to carry over between solves.
        ksp.setInitialGuessNonzero(False)

        is_p = PETSc.IS().createGeneral(
            pressure_rows.astype(PETSc.IntType), comm=comm)
        rest = _np.setdiff1d(_np.arange(n, dtype=int), pressure_rows,
                             assume_unique=True)
        is_rest = PETSc.IS().createGeneral(rest.astype(PETSc.IntType), comm=comm)

        pc = ksp.getPC()
        if self.strategy == 'fieldsplit':
            self._configure_fieldsplit(pc, is_p, is_rest)
        else:
            self._configure_cpr(pc, is_p, is_rest, comm)
        ksp.setUp()
        self._finish_subsolvers(pc)

        self._ksp, self._mat = ksp, mat
        self._rhs = mat.createVecLeft()
        self._sol = mat.createVecRight()

    def _configure_fieldsplit(self, pc, is_p, is_rest):
        from petsc4py import PETSc

        pc.setType('fieldsplit')
        pc.setFieldSplitIS(('pressure', is_p), ('rest', is_rest))
        pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

    def _configure_cpr(self, pc, is_p, is_rest, comm):
        from petsc4py import PETSc

        # Stage one restricted to the pressure field, stage two over the
        # whole system: PCCOMPOSITE applies them multiplicatively, which is
        # the two-stage form -- stage two sees the residual stage one left.
        pc.setType('composite')
        pc.setCompositeType(PETSc.PC.CompositeType.MULTIPLICATIVE)
        pc.addCompositePCType('fieldsplit')
        pc.addCompositePCType(self.second_stage)
        stage_one = pc.getCompositePC(0)
        stage_one.setType('fieldsplit')
        stage_one.setFieldSplitIS(('pressure', is_p))
        stage_one.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    #: Which second-stage types are incomplete/complete factorisations, and
    #: so need the pivot shift and accept a fill level.
    _FACTOR_TYPES = ('ilu', 'icc', 'lu', 'cholesky')

    def _configure_second_stage(self, pc):
        """The full-system smoother, made pivot-proof and stated outright.

        A black-oil Jacobian reaches ILU with exact zeros on its diagonal:
        a well closure equation constrains other unknowns than its own, and
        a deck that declares a phase it does not have -- SPE10 model 2
        declares gas and holds none -- contributes a balance that is
        identically zero in every cell.  _with_explicit_diagonal gives
        those positions a stored entry so the factorisation will accept the
        matrix at all, but the entry is zero and the pivot still fails.
        PETSc reports it as KSP_DIVERGED_PC_FAILED at iteration zero, which
        reads like the solve diverging rather than never starting.

        Shifting the offending pivots is what the option exists for, and it
        perturbs only the preconditioner: the Krylov iteration still works
        on the true operator, so the answer is unchanged.

        The fill level and ordering are set here rather than left to PETSc.
        They *are* ILU(0) in natural order by default, so this changes
        nothing today -- but the cost of a CPR second stage is decided
        almost entirely by those two numbers, and a default that is right
        is still a default: it can differ between PETSc builds, and reading
        the timings while unsure which factorisation produced them is how
        the wrong half gets optimised.  Reordering in particular is a trap.
        It usually shortens a direct factorisation and usually lengthens an
        incomplete one, because ILU(0) keeps only the sparsity it was given
        and a permutation changes what that sparsity is worth.
        """
        from petsc4py import PETSc

        pc.setType(self.second_stage)
        options = dict(self.second_stage_options or {})
        if self.second_stage in self._FACTOR_TYPES:
            pc.setFactorShift(PETSc.Mat.FactorShiftType.NONZERO)
            pc.setFactorLevels(int(options.pop('levels', 0)))
            ordering = options.pop('ordering', 'natural')
            if ordering:
                pc.setFactorOrdering(ord_type=str(ordering))
        # Everything else goes through the options database rather than a
        # method, because petsc4py exposes setters for the factorisations
        # and almost nothing else -- 'sor' has no setOmega, 'bjacobi' no
        # setBlocks.  Keys are PETSc option names without the leading dash
        # ('pc_sor_omega', 'pc_bjacobi_blocks'); the PC's own prefix is
        # prepended so a sub-PC inside the composite gets its own.
        if options:
            database = PETSc.Options()
            prefix = pc.getOptionsPrefix() or ''
            for name, value in options.items():
                database[prefix + str(name)] = value
            pc.setFromOptions()

    def _finish_subsolvers(self, pc):
        """Point the pressure block at multigrid, once the splits exist.

        The sub-solvers only come into being when the outer preconditioner
        is set up, so this cannot be folded into the configuration above.
        """
        if self.strategy == 'fieldsplit':
            splits = pc.getFieldSplitSubKSP()
            self._set_pressure_pc(splits[0])
            if len(splits) > 1:
                splits[1].setType('preonly')
                self._configure_second_stage(splits[1].getPC())
            return
        stage_one = pc.getCompositePC(0)
        stage_one.setUp()
        self._set_pressure_pc(stage_one.getFieldSplitSubKSP()[0])
        self._configure_second_stage(pc.getCompositePC(1))

    def _set_pressure_pc(self, ksp):
        ksp.setType('preonly')
        sub = ksp.getPC()
        if self.pressure_precond == 'hypre':
            sub.setType('hypre')
            sub.setHYPREType('boomeramg')
        else:
            sub.setType(self.pressure_precond)

    # ------------------------------------------------------------------
    # LinearSolverAD interface
    # ------------------------------------------------------------------
    def solveLinearSystem(self, A, b, problem=None, model=None):
        if _sp is None or not _sp.issparse(A):
            A = _sp.csr_matrix(_np.asarray(A, dtype=float))
        A = A.tocsr().astype(float)
        A.sort_indices()
        b = _np.asarray(b, dtype=float).ravel()

        pressure_rows = self._pressure_index(A.shape[0], problem, model)
        if self.decouple:
            nc = pressure_rows.size
            ncomp = self._component_count(A.shape[0], nc, problem, model)
            if ncomp > 1:
                M = self._decoupling_operator(A, nc, ncomp,
                                              problem=problem, model=model)
                A = (M @ A).tocsr()
                A.sort_indices()
                b = M @ b
        if self.equilibrate:
            A, b = self._equilibrate_rows(A, b)
        A = self._with_explicit_diagonal(A)
        key = self._pattern_key(A)
        if self._ksp is None or key != self._pattern or not self.reuse_setup:
            self.destroy()
            self._build(A, pressure_rows)
            self._pattern = key
        else:
            # Same sparsity, new values: overwrite the stored entries and
            # let PETSc keep the multigrid hierarchy it already built.
            self._mat.setValuesCSR(self._indptr, self._indices,
                                   A.data.astype(float))
            self._mat.assemble()
            # The values changed, so the hierarchy built from them is stale
            # unless PETSc is told the pattern held. SAME_NONZERO_PATTERN is
            # what lets it rebuild only the parts that depend on the values.
            self._ksp.setOperators(self._mat, self._mat)

        self._rhs.setArray(b)
        self._sol.set(0.0)
        self._ksp.solve(self._rhs, self._sol)
        self.lastIterations = int(self._ksp.getIterationNumber())
        self.lastConvergedReason = int(self._ksp.getConvergedReason())
        return self._sol.getArray().copy()

    def _fall_back_to_boomeramg(self):
        """Swap smoothed aggregation for classical AMG after a failed solve.

        PETSc's own GAMG aggregates by strength of connection, which a
        permeability field spanning six orders of magnitude defeats: on
        SPE10 model 1 it stalls at a relative residual of 0.75 after 200
        iterations, where BoomerAMG converges in 26.  On a smooth field the
        preference is the other way round -- SPE9's gamg solve is five times
        faster -- so the choice cannot be made once for all models, and
        making it by trying is cheaper than making it by guessing.

        The switch is one-way and happens at most once: rebuilding the
        hierarchy costs more than a solve, and a model heterogeneous enough
        to need it on one Newton step needs it on all of them.
        """
        if self.pressure_precond != 'gamg' or self._precond_fallback_used:
            return False
        if not petsc_has_preconditioner('hypre'):
            return False
        self._precond_fallback_used = True
        self.pressure_precond = 'hypre'
        self.destroy()
        if self.verbose:
            print('PETScSolverAD: gamg did not converge; switching to '
                  'hypre/boomeramg for the rest of the run')
        return True

    def solveLinearProblem(self, problem, model=None):
        started = _time.perf_counter()
        A, b = self._get_system(problem)
        x = self.solveLinearSystem(A, b, problem=problem, model=model)
        if self.lastConvergedReason is not None and self.lastConvergedReason <= 0:
            if self._fall_back_to_boomeramg():
                x = self.solveLinearSystem(A, b, problem=problem, model=model)

        residual_norm = float(_np.linalg.norm(A @ x - b))
        scale = float(_np.linalg.norm(b))
        relative = residual_norm / scale if scale > 0 else residual_norm
        self.lastResidual = relative
        self.iterations += 1
        elapsed = float(_time.perf_counter() - started)
        # A negative reason is a PETSc divergence code. Reporting it as
        # unconverged lets the nonlinear solver cut the timestep rather than
        # take a Newton step built on a failed solve.
        converged = self.lastConvergedReason > 0
        report = self.getSolveReport(
            Iterations=self.lastIterations,
            Residual=relative,
            SolverTime=elapsed,
            LinearSolutionTime=elapsed,
            Converged=converged,
            ConvergedReason=self.lastConvergedReason,
            Preconditioner='petsc-%s(%s,%s)' % (
                self.strategy, self.pressure_precond, self.second_stage),
            Decoupling=getattr(self, 'lastDecoupling', self.decoupling),
        )
        if self.verbose:
            print('PETScSolverAD[%s]: %d iterations, relative residual %.3e'
                  % (self.strategy, self.lastIterations, relative))
        return x, relative, report

    def destroy(self):
        """Release the cached PETSc objects.

        PETSc objects are reference counted in C and are not freed by Python
        going out of scope alone; a long history match building a solver per
        realisation would otherwise accumulate multigrid hierarchies.
        """
        self._indptr = None
        self._indices = None
        for name in ('_sol', '_rhs', '_ksp', '_mat'):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass
                setattr(self, name, None)
        self._pattern = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
