"""Which automatic-differentiation representation a model assembles with.

Port of MRST's ``AutoDiffBackend``/``DiagonalAutoDiffBackend``
(mrst-2026a/autodiff/ad-core/backends).  The backend is the one object that
decides how a residual's derivatives are stored, and every other part of the
assembly is written against the shared :class:`ADValue` interface so that
swapping it changes cost and nothing else.

The two available representations differ only in how a per-cell property's
Jacobian is held between operations:

:class:`AutoDiffBackend`
    one assembled sparse matrix per intermediate value.  Every elementwise
    operation is then sparse-matrix algebra -- correct, and what the rest of
    MRST's numerics were checked against.

:class:`DiagonalAutoDiffBackend`
    a short list of generalized diagonals.  A per-cell property depends on
    only the handful of primary variables in its own cell, so its Jacobian
    has one entry per row per variable and needs no sparse structure at all;
    the arithmetic becomes plain array operations.  On a black-oil property
    chain this is several times cheaper, and the difference grows with the
    grid.

Both produce the same numbers.  The diagonal one materialises a sparse
Jacobian at the points where the structure genuinely stops being diagonal --
the flux divergence, and stacking the equations into one system -- which is
where a linear solve needs a real matrix anyway.
"""

from __future__ import annotations

from .adi import SparseADI
from .diagonal_adi import DiagonalADI


class AutoDiffBackend:
    """Assemble with one sparse Jacobian per value (MRST's default)."""

    #: The :class:`ADValue` subclass this backend seeds.
    ad_class = SparseADI

    def variable(self, value, nvar, offset, group=None):
        """Seed a primary variable: an identity block at ``offset``.

        ``group`` names the offsets of the variables this one is seeded
        with.  An assembled Jacobian has no use for it -- the matrix is as
        wide as the system either way -- but the diagonal representation
        does, so every caller passes it and the backend decides whether it
        matters.
        """
        return self.ad_class.variable(value, nvar, offset)

    def constant(self, value, nvar):
        """A value with no derivatives, of the given system width."""
        return self.ad_class.constant(value, nvar)

    def __repr__(self):
        return '%s()' % type(self).__name__

    def getBackendDescription(self):
        """Port of MRST's ``getBackendDescription``."""
        return 'Sparse (one assembled Jacobian per value)'


class DiagonalAutoDiffBackend(AutoDiffBackend):
    """Assemble per-cell properties as generalized diagonals."""

    ad_class = DiagonalADI

    def variable(self, value, nvar, offset, group=None):
        # Full width from the start, so every value in the chain has the
        # same shape and no operation has to widen one to meet another.
        return self.ad_class.variable(value, nvar, offset, group)

    def getBackendDescription(self):
        return 'Diagonal (one entry per row per primary variable)'


#: Short names accepted by ``init_eclipse_problem_ad``'s ``AutoDiffBackend``
#: option and by the benchmark harness.
BACKENDS = {
    'sparse': AutoDiffBackend,
    'diagonal': DiagonalAutoDiffBackend,
}


def get_backend(spec):
    """Resolve a backend from a name, a class, an instance or ``None``.

    ``None`` means the sparse backend, so a model that was never told
    anything keeps the representation every existing result was produced
    with.
    """
    if spec is None:
        return AutoDiffBackend()
    if isinstance(spec, AutoDiffBackend):
        return spec
    if isinstance(spec, type) and issubclass(spec, AutoDiffBackend):
        return spec()
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in BACKENDS:
            return BACKENDS[key]()
        raise ValueError('unknown AutoDiffBackend %r; expected one of %s'
                         % (spec, ', '.join(sorted(BACKENDS))))
    raise TypeError('cannot use %r as an AutoDiffBackend' % (spec,))
