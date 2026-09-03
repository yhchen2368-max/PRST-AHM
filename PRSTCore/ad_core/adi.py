"""Sparse forward automatic differentiation used by deck-driven models.

This module is a compact Python port of MRST's ``ADI`` representation in
``core/utils/ADI.m`` and ``initVariablesADI.m``.  Like MRST after
``combineEquations``, every object carries a numerical value and one
assembled sparse Jacobian with respect to the flattened primary-variable
vector.  It deliberately keeps the operations vector-only, matching the
column-vector contract of the original ADI class.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ADValue:
    """What the two automatic-differentiation representations have in common.

    :class:`SparseADI` stores one assembled sparse Jacobian;
    :class:`~PRSTCore.ad_core.diagonal_adi.DiagonalADI` stores a short list
    of generalized diagonals.  They are interchangeable through a
    long elementwise property chain and differ only in cost, so the code
    between them -- the free functions below, the PVT and relative
    permeability evaluation, the well model -- should not name either.

    It exists so that ``isinstance(x, SparseADI)``, which appears wherever
    a routine has to tell an AD value from a plain array, can ask the
    question it means instead of naming one implementation.  Those tests
    are what silently excluded the diagonal representation from every code
    path that has one.

    The shared interface is ``val``, ``nvar``, ``scale_rows``,
    ``_broadcast``, ``_pair`` and the arithmetic operators.
    """

    __slots__ = ()

    # Without this, `numpy_array * ad` (or /, +, -) dispatches through
    # numpy's own __mul__ first, which iterates elementwise via __getitem__
    # and produces an object array of length-1 AD values instead of one AD
    # vector.  Setting it here makes numpy return NotImplemented so Python
    # falls back to the reflected operator, for every representation --
    # SparseADI carried this and DiagonalADI did not, so the same
    # array-on-the-left expression was correct in one and silently wrong in
    # the other.
    __array_ufunc__ = None

    def scale_rows(self, factor):
        """This value with row ``i``'s derivative multiplied by ``factor[i]``.

        The one operation every branch-selecting helper needs: a maximum, a
        ``where`` and a table interpolation all come down to scaling each
        row's derivative and adding the results.  Each representation can
        do it without materialising the other's.
        """
        raise NotImplementedError

    @classmethod
    def combine_scaled(cls, value, terms):
        """``value``, carrying ``sum(ad.derivative * factor)`` over ``terms``.

        The branch-selecting helpers all have this shape, and expressing it
        as one call rather than as ``a.scale_rows(f) + b.scale_rows(g)``
        followed by a value substitution matters: that spelling builds three
        intermediate AD values per call, each copying a Jacobian, on the
        hottest path in the assembly.  Here each representation builds its
        result once.

        ``terms`` is a sequence of ``(ad_value, per_row_factor)``.
        """
        raise NotImplementedError


def is_ad(value):
    """Whether ``value`` carries derivatives, in either representation."""
    return isinstance(value, ADValue)


def _is_foreign_ad(value):
    """An AD value in some representation other than :class:`SparseADI`.

    Ordered so the overwhelmingly common operand -- a SparseADI -- settles it
    on the first test.  This runs on every arithmetic operation in an
    assembly, millions of times per Newton step.
    """
    return not isinstance(value, SparseADI) and isinstance(value, ADValue)


def as_sparse(value):
    """``value`` in the assembled-Jacobian representation.

    The conversion point for anything that needs a real sparse matrix: a
    linear solve, an adjoint, stacking equations into one system.  Plain
    arrays and :class:`SparseADI` values pass through untouched.
    """
    return value.to_sparse() if _is_foreign_ad(value) else value


class SparseADI(ADValue):
    """Column-vector value with a sparse first-derivative matrix."""

    __slots__ = ('val', 'jac')

    # Without this, `numpy_array * spadi` (or /, +, -) dispatches through
    # numpy's own __mul__ first (SparseADI has no __array_priority__), which
    # silently iterates elementwise via __getitem__ instead of deferring to
    # SparseADI.__rmul__ -- producing a garbage object array of length-1
    # SparseADI values rather than a proper SparseADI. Setting this makes
    # numpy return NotImplemented so Python falls back to our reflected
    # operators, as intended for every numeric-array-on-the-left site.
    __array_ufunc__ = None

    def __init__(self, value, jacobian):
        self.val = np.asarray(value, dtype=float).reshape(-1)
        self.jac = sp.csr_matrix(jacobian, dtype=float)
        if self.jac.shape[0] != self.val.size:
            raise ValueError('ADI value and Jacobian row counts must agree')

    @property
    def nvar(self):
        return self.jac.shape[1]

    @classmethod
    def variable(cls, value, nvar, offset):
        """Port one ``initVariablesADI`` identity Jacobian block."""
        value = np.asarray(value, dtype=float).reshape(-1)
        n = value.size
        if offset < 0 or offset + n > nvar:
            raise ValueError('ADI identity block is outside the primary vector')
        rows = np.arange(n, dtype=int)
        cols = rows + int(offset)
        jac = sp.csr_matrix((np.ones(n), (rows, cols)), shape=(n, int(nvar)))
        return cls(value, jac)

    @classmethod
    def constant(cls, value, nvar):
        value = np.asarray(value, dtype=float).reshape(-1)
        return cls(value, sp.csr_matrix((value.size, int(nvar))))

    def copy(self):
        return SparseADI(self.val.copy(), self.jac.copy())

    def _broadcast(self, size):
        if self.val.size == size:
            return self
        if self.val.size != 1:
            raise ValueError('ADI vectors have incompatible lengths')
        return SparseADI(np.full(size, self.val[0]), sp.vstack([self.jac] * size, format='csr'))

    @staticmethod
    def _numeric(value, size):
        out = np.asarray(value, dtype=float).reshape(-1)
        if out.size == size:
            return out
        if out.size == 1:
            return np.full(size, out[0])
        raise ValueError('ADI and numeric vectors have incompatible lengths')

    @staticmethod
    def _pair(left, right):
        # A diagonal operand meeting a sparse one is promoted, not rejected.
        # The two representations legitimately meet inside a single residual:
        # the flux divergence leaves the diagonal form at ``linear_map``
        # while the well source terms are still in it, and the conservation
        # equation adds the two together.  Sparse is the general form, so it
        # is the one that survives.
        left = left.to_sparse() if _is_foreign_ad(left) else left
        right = right.to_sparse() if _is_foreign_ad(right) else right
        if isinstance(left, SparseADI) and isinstance(right, SparseADI):
            if left.nvar != right.nvar:
                raise ValueError('ADI variables use different primary vectors')
            n = max(left.val.size, right.val.size)
            return left._broadcast(n), right._broadcast(n)
        if isinstance(left, SparseADI):
            return left, SparseADI._numeric(right, left.val.size)
        if isinstance(right, SparseADI):
            return SparseADI._numeric(left, right.val.size), right
        return left, right

    @staticmethod
    def _scale_rows(jac, scale):
        # Equivalent to jac.multiply(scale.reshape(-1, 1)).tocsr(), but
        # scipy's generic sparse .multiply() broadcasting path (COO
        # round-trip + index-dtype resolution on every call) dominates
        # runtime in PVT-heavy simulations (profiled: ~45% of total
        # simulate_schedule_ad time on the SPE1 benchmark). Scaling each
        # row of an already-CSR matrix by a per-row scalar is just
        # repeating `scale` once per stored nonzero and multiplying
        # `.data` directly -- skip scipy's generic path entirely.
        jac = jac.tocsr()
        scale = np.asarray(scale, dtype=float).reshape(-1)
        if jac.nnz == 0:
            return jac.copy()
        row_mult = np.repeat(scale, np.diff(jac.indptr))
        return sp.csr_matrix((jac.data * row_mult, jac.indices, jac.indptr), shape=jac.shape)

    def scale_rows(self, factor):
        """Port of :meth:`ADValue.scale_rows` for the assembled Jacobian."""
        return SparseADI(self.val, self._scale_rows(self.jac, factor))

    @classmethod
    def combine_scaled(cls, value, terms):
        """Port of :meth:`ADValue.combine_scaled` for assembled Jacobians."""
        jacobian = None
        for ad, factor in terms:
            scaled = cls._scale_rows(ad.jac, factor)
            jacobian = scaled if jacobian is None else jacobian + scaled
        return SparseADI(value, jacobian)

    def __neg__(self):
        return SparseADI(-self.val, -self.jac)

    def __add__(self, other):
        left, right = self._pair(self, other)
        if isinstance(right, SparseADI):
            return SparseADI(left.val + right.val, left.jac + right.jac)
        return SparseADI(left.val + right, left.jac)

    __radd__ = __add__

    def __sub__(self, other):
        # is_ad, not isinstance(other, SparseADI): negating a
        # diagonal operand keeps it an AD value, and asking for the concrete
        # class here sent it through np.asarray instead, which cannot
        # represent it.  _pair in __add__ then does the promotion.
        return self + (-other if is_ad(other) else -np.asarray(other, dtype=float))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        left, right = self._pair(self, other)
        if isinstance(right, SparseADI):
            jac = self._scale_rows(left.jac, right.val) + self._scale_rows(right.jac, left.val)
            return SparseADI(left.val * right.val, jac)
        return SparseADI(left.val * right, self._scale_rows(left.jac, right))

    __rmul__ = __mul__

    def reciprocal(self):
        inv = 1.0 / self.val
        return SparseADI(inv, self._scale_rows(self.jac, -inv * inv))

    def __truediv__(self, other):
        if is_ad(other):
            return self * other.reciprocal()
        return self * (1.0 / np.asarray(other, dtype=float))

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    def __pow__(self, exponent):
        """Port ADI.m ``power`` for an ADI base (numeric or ADI exponent)."""
        if is_ad(exponent):
            # d(u^v) = u^v * (v/u) du + u^v * log(u) dv
            base, exp = self._pair(self, exponent)
            value = base.val ** exp.val
            jac = (self._scale_rows(base.jac, value * (exp.val / base.val)) +
                   self._scale_rows(exp.jac, value * np.log(base.val)))
            return SparseADI(value, jac)
        # ADI.m broadcasts a length-1 base against a longer exponent.
        exponent = np.asarray(exponent, dtype=float).reshape(-1)
        base = self if exponent.size == 1 else self._broadcast(exponent.size)
        exponent = self._numeric(exponent, base.val.size)
        value = base.val ** exponent
        return SparseADI(value, self._scale_rows(base.jac, exponent * base.val ** (exponent - 1.0)))

    def __rpow__(self, base):
        """Port ADI.m ``power`` for a numeric base and an ADI exponent."""
        base = np.asarray(base, dtype=float).reshape(-1)
        exp = self if base.size == 1 else self._broadcast(base.size)
        base = self._numeric(base, exp.val.size)
        value = base ** exp.val
        return SparseADI(value, self._scale_rows(exp.jac, value * np.log(base)))

    def exp(self):
        value = np.exp(self.val)
        return SparseADI(value, self._scale_rows(self.jac, value))

    def log(self):
        return SparseADI(np.log(self.val), self._scale_rows(self.jac, 1.0 / self.val))

    def sum(self):
        """Port of ``ADI.m``'s ``sum``: the total, and its derivative.

        A scalar objective assembled from a vector of per-well or
        per-cell terms ends here, and the single Jacobian row that comes
        back is exactly the ``dg/dx`` an adjoint wants. Callers probe for
        this with ``hasattr(x, 'sum')`` and fall back to ``numpy.sum``,
        which cannot see through an AD object -- so its absence turned
        into a TypeError rather than a wrong number.
        """
        value = np.array([self.val.sum()])
        jac = self.jac
        if jac.nnz == 0:
            jac_row = sp.csr_matrix((1, jac.shape[1]))
        else:
            csr = jac.tocsr()
            # Column sums without materialising the full ``nvar``-wide
            # result.  ``np.bincount`` then ``csr_matrix(colsum)`` scans
            # all ``nvar`` columns twice to place a handful of nonzeros --
            # for a well's per-perforation surface rate that is ~2 ms per
            # call (T142 nvar ~ 382k), i.e. most of a 111-well assembly.
            # Sorting the (small) stored entries and reducing each column
            # group touches only the nonzeros, then the COO -> CSR
            # conversion places exactly those entries.
            idx = csr.indices
            data = csr.data
            if idx.size:
                order = np.argsort(idx, kind='stable')
                idx_s = idx[order]
                data_s = data[order]
                uniq, counts = np.unique(idx_s, return_counts=True)
                if uniq.size:
                    ends = np.cumsum(counts)
                    starts = ends - counts
                    colsum = np.add.reduceat(data_s, starts)
                    indptr = np.array([0, uniq.size])
                    jac_row = sp.csr_matrix((colsum, uniq, indptr),
                                            shape=(1, jac.shape[1]))
                else:
                    jac_row = sp.csr_matrix((1, jac.shape[1]))
            else:
                jac_row = sp.csr_matrix((1, jac.shape[1]))
        return SparseADI(value, jac_row)

    def __getitem__(self, index):
        return SparseADI(self.val[index], self.jac[index, :])

    def linear_map(self, matrix):
        """Port ADI.m ``mtimes`` for a numeric left matrix."""
        matrix = sp.csr_matrix(matrix)
        return SparseADI(matrix @ self.val, matrix @ self.jac)

    @staticmethod
    def concat(items):
        """Port ADI.m ``combineEquations`` after Jacobian assembly."""
        # Stacking equations of different heights has no single column per
        # row, so this is where a diagonal representation ends regardless.
        items = [as_sparse(item) for item in items]
        first = next((item for item in items if isinstance(item, SparseADI)), None)
        if first is None:
            return np.concatenate([np.asarray(item, dtype=float).reshape(-1) for item in items])
        nvar = first.nvar
        values = []
        jacobians = []
        for item in items:
            if isinstance(item, SparseADI):
                if item.nvar != nvar:
                    raise ValueError('ADI variables use different primary vectors')
                values.append(item.val)
                jacobians.append(item.jac)
            else:
                value = np.asarray(item, dtype=float).reshape(-1)
                values.append(value)
                jacobians.append(sp.csr_matrix((value.size, nvar)))
        return SparseADI(np.concatenate(values), sp.vstack(jacobians, format='csr'))

    @staticmethod
    def scatter(indices, part, size):
        """Insert a row subset into a zero ADI vector of ``size`` rows."""
        part = as_sparse(part)
        if not isinstance(part, SparseADI):
            raise TypeError('SparseADI.scatter requires an ADI subset')
        indices = np.asarray(indices, dtype=int).reshape(-1)
        if indices.size != part.val.size:
            raise ValueError('ADI scatter indices and subset rows must agree')
        coo = part.jac.tocoo()
        jac = sp.csr_matrix((coo.data, (indices[coo.row], coo.col)),
                            shape=(int(size), part.nvar))
        value = np.zeros(int(size), dtype=float)
        # Repeated indices accumulate, because the Jacobian above already
        # does -- COO to CSR sums duplicate entries.  A plain ``value[idx] =``
        # keeps only the last write, so a batched scatter over two well
        # perforations completed in the same cell would return a value that
        # its own derivative did not describe.
        np.add.at(value, indices, part.val)
        return SparseADI(value, jac)


def _ad_class(*values):
    """The AD representation to build the result in, or ``None`` for arrays.

    When the operands disagree, the general representation wins and the
    others are promoted, the same rule as :meth:`SparseADI._pair`: the two
    forms meet legitimately where a residual adds a flux term, already
    materialised, to a source term that is still diagonal.
    """
    # A loop rather than a set comprehension: this is called for every
    # maximum, minimum and select in the assembly, and building a set of
    # types to then look at its length cost more than the comparison it was
    # standing in for.
    found = None
    for value in values:
        if isinstance(value, ADValue):
            cls = type(value)
            if found is None:
                found = cls
            elif found is not cls:
                return SparseADI
    return found


def ad_maximum(left, right):
    """Elementwise maximum with MRST ADI.m's active-branch derivative."""
    cls = _ad_class(left, right)
    if cls is None:
        return np.maximum(np.asarray(left), np.asarray(right))
    left, right = cls._pair(left, right)
    if is_ad(left) and is_ad(right):
        choose_left = left.val > right.val
        value = np.maximum(left.val, right.val)
        return cls.combine_scaled(value, ((left, choose_left),
                                         (right, ~choose_left)))
    if is_ad(left):
        # ADI.m's max(u, v) with a numeric v delegates to max(v, u), i.e. the
        # numeric-left branch below with the roles swapped -- so the ADI
        # derivative survives when the two are *equal*, and the test is >=
        # rather than >. (The both-ADI branch above is the opposite way
        # round: `inx = u.val > v.val` gives a tie to the right operand.)
        choose_left = left.val >= right
        return cls.combine_scaled(np.maximum(left.val, right),
                                  ((left, choose_left),))
    choose_right = right.val >= left
    return cls.combine_scaled(np.maximum(left, right.val),
                              ((right, choose_right),))


def ad_minimum(left, right):
    return -ad_maximum(-left if is_ad(left) else -np.asarray(left),
                       -right if is_ad(right) else -np.asarray(right))


def ad_abs(x):
    """``|x|`` with a sign-consistent derivative (matches MRST ADI.m's own
    ``abs`` implementation: ``d|x| = sign(x) dx``)."""
    return type(x).combine_scaled(np.abs(x.val), ((x, np.sign(x.val)),))


def ad_select(mask, when_true, when_false):
    """Select ADI rows using a value-only branch flag (MRST upwinding).

    Either branch may be a plain array -- a constant on one side of a
    switch is ordinary. Only the pair being constant is an error, since
    then there is no derivative to select and the caller wants
    ``numpy.where``.

    The false branch was accepted as a constant from the start and the
    true branch was not, which made two-point relative-permeability
    end-point scaling raise on every deck that used it. Three-point
    scaling passes AD on that side, so the asymmetry went unnoticed.
    """
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    cls = _ad_class(when_true, when_false)
    if cls is None:
        raise TypeError('ad_select requires at least one AD value')
    true, false = cls._pair(when_true, when_false)
    if not is_ad(true):
        true = cls.constant(true, false.nvar)._broadcast(mask.size)
    if not is_ad(false):
        false = cls.constant(false, true.nvar)._broadcast(mask.size)
    true = true._broadcast(mask.size)
    false = false._broadcast(mask.size)
    value = np.where(mask, true.val, false.val)
    return cls.combine_scaled(value, ((true, mask), (false, ~mask)))


def ad_interp_linear(x, y, query):
    """MRST ``interpTable`` value/slope rule for an ADI query vector."""
    if not is_ad(query):
        raise TypeError('ad_interp_linear requires an AD query')
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size == 1:
        # A one-row table is a constant: no slope, so no derivative either.
        return type(query).constant(np.full(query.val.size, y[0]), query.nvar)
    bins = np.searchsorted(x, query.val, side='right') - 1
    np.clip(bins, 0, x.size - 2, out=bins)
    # The slope belongs to the table's segments, not to the query points, so
    # it has only x.size-1 distinct values.  Computing it per query point --
    # y[bins+1] - y[bins] over x[bins+1] - x[bins] -- gathered four arrays as
    # long as the grid where one gather off a table-sized array does; on
    # SPE10 model 2 that is four 1.12M-element temporaries per call, and this
    # is called about twenty times per assembly.  Same arithmetic, same bits.
    slope = (np.diff(y) / np.diff(x))[bins]
    value = y[bins] + slope * (query.val - x[bins])
    return type(query).combine_scaled(value, ((query, slope),))
