"""MRST-compatible automatic linear solver selection.

This module ports ``selectLinearSolverAD`` from MRST.  The selector keeps
solver policy in one place while returning the existing PRSTCore solver objects.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import numpy as np

from .backslash_solver_ad import BackslashSolverAD
from .agmg_solver_ad import AGMGSolverAD
from .amgcl_solver_ad import AMGCLSolverAD
from .amgcl_cpr_solver_ad import AMGCL_CPRSolverAD
from .cpr_solver_ad import CPRSolverAD
from .gmres_ilu_solver_ad import GMRES_ILUSolverAD
from .mumps_solver_ad import MUMPSSolverAD, check_mumps
from .petsc_solver_ad import PETScSolverAD, check_petsc, petsc_has_preconditioner


def select_linear_solver_ad(model: Any, **kwargs: Any):
    """Select a linear solver using MRST's ``selectLinearSolverAD`` policy.

    Supported options use MRST's camel-case names and Python snake-case
    aliases.  The returned object is one of PRSTCore's ``LinearSolverAD``
    implementations.
    """
    opt = {
        'useAMGCL': True,
        'useAGMG': True,
        'useILU': True,
        'useSYMRCMOrdering': False,
        'useCPR': True,
        'useAMGCLCPR': True,
        'useMUMPS': True,
        'usePETSc': True,
        # Unknowns above which BoomerAMG beats PETSc's own GAMG; see the
        # measurements where it is used.
        'PETScBoomerAMGThreshold': 60000,
        # MRST's AMGCL_CPRSolverAD spells this 'decoupling'.
        'CPRDecoupling': 'trueIMPES',
        'BackslashThreshold': 10000,
        'tolerance': 1e-4,
    }
    aliases = {
        'use_amgcl': 'useAMGCL',
        'use_agmg': 'useAGMG',
        'use_ilu': 'useILU',
        'use_symrcm_ordering': 'useSYMRCMOrdering',
        'use_cpr': 'useCPR',
        'use_amgcl_cpr': 'useAMGCLCPR',
        'use_mumps': 'useMUMPS',
        'use_petsc': 'usePETSc',
        'petsc_boomeramg_threshold': 'PETScBoomerAMGThreshold',
        'cpr_decoupling': 'CPRDecoupling',
        'decoupling': 'CPRDecoupling',
        'backslash_threshold': 'BackslashThreshold',
    }
    for key, value in kwargs.items():
        opt[aliases.get(key, key)] = value

    solver_kwargs = dict(kwargs.get('solver_arg', {}))
    solver_kwargs.setdefault('tolerance', opt['tolerance'])
    verbose = bool(kwargs.get('verbose', kwargs.get('Verbose', False)))
    ncomp = get_component_count(model)
    nc = _number_of_cells(model)
    ndof = ncomp * nc

    solver = BackslashSolverAD(verbose=verbose)
    if ndof <= int(opt['BackslashThreshold']):
        return solver

    amgcl_ok = bool(opt['useAMGCL']) and check_amgcl()
    agmg_ok = bool(opt['useAGMG']) and check_agmg()
    diagonal = _is_diagonal_backend(model)

    # PETSc first among the iterative options. Its CPR is the same two-stage
    # method as the AMGCL one below, but the preconditioner is assembled
    # inside PETSc rather than driven from a Python callback per Krylov
    # iteration, and the multigrid hierarchy survives between Newton steps.
    # Measured on SPE9's first system: 0.08 s against 2.76 s, with the well
    # solution unchanged. It stays behind an availability check, so an
    # environment without petsc4py keeps exactly the ordering it had.
    if bool(opt['usePETSc']) and check_petsc():
        # Which multigrid depends on the size, and both directions were
        # measured. On SPE9 (27k unknowns) PETSc's own GAMG solves in
        # 0.08 s against BoomerAMG's 0.44 s -- hypre's setup dominates when
        # there is little to amortise it over. On Norne (135k) it reverses:
        # BoomerAMG needs 393 Krylov iterations across three report steps
        # where GAMG needs 1948, and wins overall. The crossover sits
        # between them.
        #
        # Getting this wrong is not fatal either way: the solver falls back
        # from GAMG to BoomerAMG if it stalls, which is what rescues a
        # strongly heterogeneous field like SPE10.
        precond = 'gamg'
        if ndof >= int(opt['PETScBoomerAMGThreshold']) and petsc_has_preconditioner('hypre'):
            precond = 'hypre'
        # trueIMPES is MRST's default decoupling and the fastest of the
        # four on SPE9 (16 Krylov iterations against quasiIMPES's 20). It
        # needs the model's fluid state, and the solver falls back to
        # quasiIMPES by itself when handed a bare matrix.
        solver = PETScSolverAD(
            strategy='cpr', pressure_precond=precond,
            decoupling=str(opt.get('CPRDecoupling', 'trueIMPES')),
            tolerance=float(opt['tolerance']), maxIterations=200,
            verbose=verbose,
            **_supported(solver_kwargs, {'extraReport'}),
        )
        _set_solver_ordering_reduction(model, solver, ncomp, opt)
        return solver

    if bool(opt['useMUMPS']) and check_mumps():
        return MUMPSSolverAD(
            tolerance=float(opt['tolerance']), verbose=verbose,
            **_supported(solver_kwargs, {'extraReport', 'ordering'}),
        )

    if bool(opt['useAMGCLCPR']) and amgcl_ok and bool(opt['useCPR']):
        solver = AMGCL_CPRSolverAD(
            tolerance=float(opt['tolerance']), maxIterations=50,
            verbose=verbose,
            blockSize=ncomp,
            **_supported(solver_kwargs, {'extraReport'}),
        )
        _set_solver_ordering_reduction(model, solver, ncomp, opt)
        return solver

    if bool(opt['useCPR']) and not diagonal:
        pressure_kwargs = {'tolerance': 1e-3, 'maxIterations': 25,
                           'verbose': verbose}
        if agmg_ok:
            pressure_solver = AGMGSolverAD(**pressure_kwargs)
        elif amgcl_ok:
            pressure_solver = AMGCLSolverAD(**pressure_kwargs)
        else:
            pressure_solver = BackslashSolverAD(verbose=verbose)
        return CPRSolverAD(
            ellipticSolver=pressure_solver,
            tolerance=float(opt['tolerance']),
            verbose=verbose,
            **_supported(solver_kwargs, {'extraReport'}),
        )

    if amgcl_ok:
        solver = AMGCLSolverAD(
            tolerance=float(opt['tolerance']), maxIterations=50,
            verbose=verbose, preconditioner='relaxation',
            relaxation='ilu0', block_size=ncomp,
            **_supported(solver_kwargs, {'extraReport'}),
        )
        _set_solver_ordering_reduction(model, solver, ncomp, opt)
        return solver

    if bool(opt['useILU']):
        solver = GMRES_ILUSolverAD(
            tolerance=float(opt['tolerance']), maxIterations=100,
            reorderEquations=False, verbose=verbose,
            **_supported(solver_kwargs, {'extraReport'}),
        )
        _set_solver_ordering_reduction(model, solver, ncomp, opt)
        return solver

    return solver


def selectLinearSolverAD(model: Any, *args: Any, **kwargs: Any):
    """MATLAB-compatible public alias.

    Positional arguments are accepted as alternating option/value pairs.
    """
    if len(args) % 2:
        raise TypeError('Solver options must be supplied as name/value pairs')
    for i in range(0, len(args), 2):
        kwargs[str(args[i])] = args[i + 1]
    return select_linear_solver_ad(model, **kwargs)


def get_component_count(model: Any) -> int:
    """Estimate MRST's number of cell components for a model."""
    validate = getattr(model, 'validateModel', None)
    if callable(validate):
        try:
            validated = validate()
            if validated is not None:
                model = validated
        except TypeError:
            pass

    names = getattr(model, 'getComponentNames', lambda: [])()
    names_count = len(names) if names is not None else 0
    class_name = model.__class__.__name__
    if class_name == 'GenericBlackOilModel' and hasattr(model, 'water'):
        # PRSTCore's black-oil Jacobian is ordered by active primary phases
        # (p, sW, sG), whereas MRST's component count is model-specific.
        ncomp = int(bool(getattr(model, 'water', False)))
        ncomp += int(bool(getattr(model, 'oil', False)))
        ncomp += int(bool(getattr(model, 'gas', False)))
    elif hasattr(model, 'gas'):
        active = int(bool(getattr(model, 'water', False)))
        active += int(bool(getattr(model, 'oil', False)))
        active += int(bool(getattr(model, 'gas', False)))
        ncomp = names_count + active
    else:
        ncomp = names_count
    if bool(getattr(model, 'thermal', False)):
        ncomp += 1
    return max(1, int(ncomp))


def check_amgcl() -> bool:
    """Return whether the scalar PyAMGCL extension is usable.

    ``amgcl`` and ``solver`` are plain Python classes that wrap the compiled
    ``pyamgcl_ext``; they exist whether or not the extension was built for
    this interpreter, so their being callable says nothing.  Asking the
    package directly is what distinguishes a usable build from a stub that
    raises on construction.
    """
    try:
        module = importlib.import_module('PRSTCore.solvers.linearsolvers.pyamgcl')
        return bool(module.has_pyamgcl_ext())
    except Exception:
        return False


def check_amgcl_block_cpr() -> bool:
    """Return whether the block-CPR extension is usable.

    Built separately from the scalar one and against the bare Python C API,
    so an interpreter with no pybind11 can have this and not the other.
    """
    try:
        module = importlib.import_module('PRSTCore.solvers.linearsolvers.pyamgcl')
        return bool(module.has_block_cpr_ext())
    except Exception:
        return False


def check_agmg() -> bool:
    """Return whether an AGMG backend is available in the current Python env."""
    for name in ('pyamg', 'agmg'):
        try:
            importlib.import_module(name)
            return True
        except Exception:
            continue
    return False


def _number_of_cells(model: Any) -> int:
    grid = getattr(model, 'G', None)
    if isinstance(grid, dict):
        cells = grid.get('cells', {})
        if isinstance(cells, dict):
            return int(cells.get('num', 0))
    cells = getattr(getattr(grid, 'cells', None), 'num', 0)
    return int(cells)


def _is_diagonal_backend(model: Any) -> bool:
    backend = getattr(model, 'AutoDiffBackend', None)
    if backend is None:
        return False
    name = getattr(backend, '__name__', backend.__class__.__name__)
    return 'DiagonalAutoDiffBackend' in str(name)


def _set_solver_ordering_reduction(model: Any, solver: Any, ncomp: int,
                                    opt: Dict[str, Any]) -> None:
    nc = _number_of_cells(model)
    ndof = nc * ncomp
    ordering: Optional[np.ndarray] = None
    if bool(opt.get('useSYMRCMOrdering', False)):
        ordering = _symrcm_ordering(model, nc, ncomp)
    if ordering is None:
        ordering = np.arange(ndof, dtype=int)

    solver.variableOrdering = ordering
    solver.equationOrdering = ordering
    if _is_diagonal_backend(model) or isinstance(solver, AMGCLSolverAD):
        solver.reduceToCell = False
        solver.keepNumber = ndof
    else:
        solver.reduceToCell = True


def _symrcm_ordering(model: Any, nc: int, ncomp: int) -> Optional[np.ndarray]:
    """Build cell-major ordering using SciPy reverse Cuthill-McKee when possible."""
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        operators = getattr(model, 'operators', None)
        neighbors = operators.get('N') if isinstance(operators, dict) else None
        if neighbors is None:
            return None
        neighbors = np.asarray(neighbors, dtype=int)
        if neighbors.ndim != 2 or neighbors.shape[1] != 2:
            return None
        rows = np.r_[neighbors[:, 0], neighbors[:, 1]]
        cols = np.r_[neighbors[:, 1], neighbors[:, 0]]
        graph = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(nc, nc)).tocsr()
        cell_order = reverse_cuthill_mckee(graph, symmetric_mode=True)
        return np.concatenate([cell_order * ncomp + j for j in range(ncomp)])
    except Exception:
        return None


def _supported(values: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if key in allowed}


__all__ = [
    'select_linear_solver_ad', 'selectLinearSolverAD', 'get_component_count',
    'check_amgcl', 'check_agmg', 'check_mumps',
]
