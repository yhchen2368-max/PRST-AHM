"""Direct sparse solver backed by MUMPS (MUltifrontal Massively Parallel Solver).

Python port of MRST's ``MUMPSSolverAD`` (see ``mumps4/MUMPSSolverAD.m``),
which wraps MATLAB's ``dmumps`` MEX interface. This module targets the
``python-mumps`` Cython bindings (conda-forge packages ``python-mumps`` +
``mumps-seq``) instead, since PRSTCore is pure Python.
"""

import os as _os
import sys as _sys
import time as _time
import threading as _threading

import numpy as _np

from .linear_solver_ad import LinearSolverAD

_mumps_module = None
_mumps_probe_done = False
_mumps_probe_error = None
_dll_dirs_registered = False


def _register_conda_dll_dirs():
    """Ensure the active conda env's own DLLs win DLL search over any
    system-wide install (notably Microsoft MPI's msmpi.dll, which sits
    ahead of the conda env on PATH and crashes MUMPS's factor step for
    anything bigger than a handful of unknowns if it shadows the env's
    bundled runtime)."""
    global _dll_dirs_registered
    if _dll_dirs_registered or _sys.platform != 'win32':
        return
    candidates = [
        _sys.prefix,
        _os.path.join(_sys.prefix, 'Library', 'bin'),
        _os.path.join(_sys.prefix, 'Library', 'usr', 'bin'),
        _os.path.join(_sys.prefix, 'Scripts'),
    ]
    existing = [d for d in candidates if _os.path.isdir(d)]
    for d in existing:
        try:
            _os.add_dll_directory(d)
        except (AttributeError, OSError):
            pass
    if existing:
        path = _os.environ.get('PATH', '')
        prefix = _os.pathsep.join(existing)
        if not path.startswith(prefix):
            _os.environ['PATH'] = prefix + _os.pathsep + path
    _dll_dirs_registered = True


def _load_mumps():
    global _mumps_module, _mumps_probe_done, _mumps_probe_error
    if _mumps_probe_done:
        return _mumps_module
    _mumps_probe_done = True
    _register_conda_dll_dirs()
    try:
        import mumps as _m
        _mumps_module = _m
    except Exception as exc:  # pragma: no cover - depends on optional install
        _mumps_probe_error = exc
        _mumps_module = None
    return _mumps_module


def check_mumps() -> bool:
    """Return whether the optional ``python-mumps`` backend is usable."""
    return _load_mumps() is not None


class MUMPSSolverAD(LinearSolverAD):
    """Direct solver using the MUMPS multifrontal sparse LU factorization.

    SYNOPSIS:
        solver = MUMPSSolverAD()

    Mirrors MRST's ``MUMPSSolverAD`` (JOB=6: analysis + factorization +
    solve in one call). ``ordering`` selects the MUMPS fill-reducing
    ordering; unsupported choices for the installed MUMPS build silently
    fall back to ``'auto'``.
    """

    def __init__(self, verbose=False, tolerance=1e-8, ordering='auto',
                 extraReport=False):
        super().__init__(verbose=verbose, tolerance=tolerance,
                         extraReport=extraReport, reduceToCell=False)
        self.ordering = str(ordering)
        # MUMPS Fortran calls have been observed to crash on threads with a
        # small default stack (notably Windows' 1 MiB main-thread stack) for
        # anything beyond toy problem sizes, so the factor+solve run on a
        # dedicated thread with a generous stack.
        self._stack_size = 64 * 1024 * 1024

    def _run_in_worker(self, fn):
        result = {}

        def target():
            try:
                result['value'] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised on join
                result['error'] = exc

        prev_stack_size = _threading.stack_size()
        try:
            _threading.stack_size(self._stack_size)
            t = _threading.Thread(target=target)
            t.start()
            t.join()
        finally:
            try:
                _threading.stack_size(prev_stack_size)
            except (ValueError, RuntimeError):
                pass
        if 'error' in result:
            raise result['error']
        return result['value']

    def solveLinearProblem(self, problem, model=None):
        t0 = _time.perf_counter()
        A, b = self._get_system(problem)

        mumps = _load_mumps()
        if mumps is None:
            raise RuntimeError(
                'MUMPSSolverAD requires the optional python-mumps backend. '
                'Install it with: conda install -c conda-forge python-mumps mumps-seq'
            ) from _mumps_probe_error

        import scipy.sparse as _sp
        Acsr = A.tocsr().astype(float) if _sp.issparse(A) else _sp.csr_matrix(_np.asarray(A, dtype=float))
        b = _np.asarray(b, dtype=float).ravel()
        n = Acsr.shape[0]

        ordering = self.ordering
        try:
            if ordering not in mumps.possible_orderings():
                ordering = 'auto'
        except Exception:
            ordering = 'auto'

        tprep0 = _time.perf_counter()

        def factor_and_solve():
            ctx = mumps.Context()
            ctx.set_matrix(Acsr)
            ctx.analyze(ordering=ordering)
            ctx.factor()
            return ctx.solve(b)

        dx = _np.asarray(self._run_in_worker(factor_and_solve), dtype=float).ravel()
        elapsed_total = float(_time.perf_counter() - tprep0)

        rfin = b - Acsr.dot(dx)
        bnorm = _np.linalg.norm(b)
        relres = float(_np.linalg.norm(rfin) / bnorm) if bnorm > 0 else float(_np.linalg.norm(rfin))
        converged = bool(_np.all(_np.isfinite(dx)))
        elapsed = float(_time.perf_counter() - t0)

        report = self.getSolveReport(
            Iterations=0,
            Residual=relres,
            SolverTime=elapsed,
            LinearSolutionTime=elapsed_total,
            PreparationTime=max(0.0, elapsed - elapsed_total),
            PostProcessTime=0.0,
            Converged=converged,
            PreconditionerReport={
                'Type': 'MUMPS-direct',
                'Ordering': ordering,
                'N': int(n),
                'NNZ': int(Acsr.nnz),
            },
        )
        if self.extraReport:
            report['ResidualHistory'] = [relres]
        return dx, relres, report

    def getDescription(self):
        d = 'MUltifrontal Massively Parallel Solver'
        sn = 'MUMPS' + self.id
        return d, sn
