"""AMGCL bindings, in two independent extensions.

``pyamgcl_ext`` is the upstream pybind11 module wrapped by :class:`amgcl` and
:class:`solver` below -- a scalar AMG preconditioner plus Krylov solver.
``pyamgcl_block_cpr_capi_ext`` is a separate, self-contained extension built
straight against the Python C API; it carries the block-CPR preconditioner
that ``AMGCL_CPRSolverBlockAD`` calls, and it shares no code with the
pybind11 one.

They are built separately, and one can be present without the other: an
interpreter with no pybind11 available gets the C-API extension only.
Importing this package therefore must not require both.  It used to open
with a bare ``from . import pyamgcl_ext``, so on such an interpreter the
block-CPR extension sitting right beside it became unreachable,
``check_amgcl()`` returned False, and ``selectLinearSolverAD`` fell all the
way back to a direct solve -- a hundredfold slowdown on a field model,
reported by nothing.  Whichever extension is present is now exposed, and the
classes that need the absent one raise when constructed rather than on
import.
"""

import sys

import numpy
import scipy
from scipy.sparse.linalg import LinearOperator

try:
    from . import pyamgcl_ext
except ImportError as _exc:  # no pybind11 build for this interpreter
    pyamgcl_ext = None
    _PYAMGCL_EXT_ERROR = _exc
else:
    _PYAMGCL_EXT_ERROR = None

try:
    from . import pyamgcl_block_cpr_capi_ext
except ImportError as _exc:  # no C-API build for this interpreter
    pyamgcl_block_cpr_capi_ext = None
    _BLOCK_CPR_ERROR = _exc
else:
    _BLOCK_CPR_ERROR = None


def has_pyamgcl_ext():
    """Whether the pybind11 scalar AMG extension is importable."""
    return pyamgcl_ext is not None


def has_block_cpr_ext():
    """Whether the C-API block-CPR extension is importable."""
    return pyamgcl_block_cpr_capi_ext is not None


def _require_ext():
    if pyamgcl_ext is None:
        raise ImportError(
            'pyamgcl_ext is not built for CPython %d.%d. The block-CPR '
            'extension is %s; build the missing one with '
            'scripts/build_pyamgcl.py. Original error: %s'
            % (sys.version_info[0], sys.version_info[1],
               'available' if pyamgcl_block_cpr_capi_ext is not None else 'also absent',
               _PYAMGCL_EXT_ERROR))


# Subclassing a type that may not exist: fall back to ``object`` so the
# module still imports, and let ``_require_ext`` produce the real diagnosis
# at construction time.
_SolverBase = pyamgcl_ext.solver if pyamgcl_ext is not None else object
_AmgclBase = pyamgcl_ext.amgcl if pyamgcl_ext is not None else object


class solver(_SolverBase):
    """
    Iterative solver with preconditioning
    """
    def __init__(self, P, prm={}):
        _require_ext()
        self.P = P
        pyamgcl_ext.solver.__init__(self, self.P, prm)

    def __repr__(self):
        return self.P.__repr__()

    def __call__(self, *args):
        """
        Solves the system for the given system matrix and the right-hand side.

        In case single argument is given, it is considered to be the right-hand
        side. The matrix given at the construction is used for solution.

        In case two arguments are given, the first one should be a new system
        matrix, and the second is the right-hand side. In this case the
        preconditioner passed on construction of the solver is still used. This
        may be of use for solution of non-steady-state PDEs, where the
        discretized system matrix slightly changes on each time step, but the
        preconditioner built for one of previous time steps is still able to
        approximate the system matrix.  This saves time needed for rebuilding
        the preconditioner.

        Parameters
        ----------
        A : the new system matrix (optional)
        rhs : the right-hand side
        """
        if len(args) == 1:
            return pyamgcl_ext.solver.__call__(self, args[0])
        elif len(args) == 2:
            Acsr = args[0].tocsr()
            return pyamgcl_ext.solver.__call__(self, Acsr.indptr, Acsr.indices, Acsr.data, args[1])
        else:
            raise TypeError('solver takes one or two arguments')


class amgcl(_AmgclBase):
    """
    Algebraic multigrid hierarchy to be used as a preconditioner
    """
    def __init__(self, A, prm={}):
        """
        Creates algebraic multigrid hierarchy to be used as preconditioner.

        Parameters
        ----------
        A     The system matrix in scipy.sparse format
        prm   Dictionary with amgcl parameters
        """
        _require_ext()
        Acsr = A.tocsr()
        self.shape = A.shape
        pyamgcl_ext.amgcl.__init__(self, Acsr.indptr, Acsr.indices, Acsr.data, prm)
