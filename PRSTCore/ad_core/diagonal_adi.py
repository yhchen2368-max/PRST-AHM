"""Python port of the core performance idea behind MRST's
``DiagonalAutoDiffBackend.m``/``DiagonalJacobian.m`` (mrst-2026a/autodiff/
ad-core/backends/diagonal, ~1100 lines combined): avoid materializing a full
sparse Jacobian for the long chains of *per-cell* elementwise nonlinear
property evaluations (PVT b-factors/viscosities, relative permeability,
mobilities, densities, ...) that dominate a black-oil residual assembly on
large grids.

Scope: this is the elementwise-chain fast path, not a full port of the
diagonal backend's block-system/Schur-complement linear-solve machinery
(``getBlockSystemCSR``/``applySchurComplementBlockSystemCSR`` in the MRST
source), which is a substantially larger, separate undertaking. The
observation this *does* implement: almost every row of a per-cell property's
Jacobian, and of a single upstream-indexed ("gather") flux term, depends on
only a handful of primary-variable columns -- so the Jacobian can be stored
as a short dense array of "generalized diagonals" instead of a full sparse
matrix, making the arithmetic O(n) array operations instead of sparse-matrix
algebra. Operations that don't preserve this structure (``linear_map`` by an
arbitrary matrix, e.g. a TPFA divergence operator with several neighbors per
cell; ``concat``/``scatter`` for stacking multiple equations into one system)
fall back to materializing a :class:`PRSTCore.ad_core.adi.SparseADI`, which is
exactly where a real solve needs a genuine sparse matrix anyway.

The derivatives of one value are held as a list of *groups*.  A group is a
row map shared by several primary variables, together with one dense
``(nrow, nvariable)`` array -- which is what ``DiagonalJacobian`` stores in
MRST, and the reason this file has that shape rather than a list of
one-column blocks.  The one-column form did the same arithmetic in a Python
loop: a black-oil property depends on three variables, so every multiply was
six ``(n,)`` products, three additions and a dictionary merge, where the
dense form is a single pass over an ``(n, 3)`` array -- one that MRST's
compiled ``mexDiagProductMult`` can take whole, which the loop could not
reach at all because it never had the three columns in one buffer.

Public API mirrors :class:`SparseADI` (``value``/``jac`` semantics via
:meth:`to_sparse`, ``variable``/``constant``/``concat``/``scatter``, and the
arithmetic operators) so callers can adopt it as a drop-in accelerator for
the property-evaluation portion of a residual assembly.
"""

from __future__ import annotations

import numpy as _np

from .adi import ADValue as _ADValue, SparseADI as _SparseADI


def _kernel():
    """MRST's compiled diagonal arithmetic, or ``None``.

    ``mexDiagMult``/``mexDiagProductMult`` fuse a scaling and a product
    rule into one pass over the dense derivative array.  numpy needs three
    passes and two temporaries for the product rule, and on a grid where
    that array is several megabytes the cost is memory traffic, not
    arithmetic.
    """
    try:
        from .mex import load_face_operators
        return load_face_operators()
    except Exception:
        return None


_KERNEL = _kernel()


class _Group:
    """A row map, and which primary variables are stored against it.

    Port of the idea in MRST's ``FixedWidthJacobian``: the column indices
    are not stored, they are *derived* from each variable's offset and a
    shared ``map`` -- the index array some gather was taken with.  Two
    groups are then combinable exactly when they carry the same map, which
    is a comparison of two references rather than of two arrays.

    Storing the columns outright loses that.  Every ``lam[upstream]``,
    ``bW[upstream]``, ``rho[c1]`` builds its own column array, so terms
    that describe the same dependency never merge: a black-oil flux term
    reached 45 of them on a face-length value, and materialising it cost
    more than the sparse representation the diagonal one is supposed to
    beat.  Holding the map by reference collapses those to one group per
    map, and the variables within a group to columns of one array.

    ``offsets`` is kept sorted so two groups built by different routes
    compare equal when they describe the same thing, which is what lets the
    fast paths below recognise the common case.
    """

    __slots__ = ('offsets', 'map', 'key')

    def __init__(self, offsets, index=None):
        self.offsets = tuple(int(offset) for offset in offsets)
        self.map = index
        # id() of the map is what makes this cheap. It is only ever compared
        # against other keys built while both maps are alive -- the values
        # holding them are the caller's own locals -- so an id cannot be
        # recycled underneath a live comparison.  ``None`` is a singleton,
        # so every unmapped group shares one key, which is right: unmapped
        # means "column = offset + row" for all of them.
        self.key = id(index)

    def with_offsets(self, offsets):
        """The same map, over a different set of variables."""
        group = _Group.__new__(_Group)
        group.offsets = tuple(int(offset) for offset in offsets)
        group.map = self.map
        group.key = self.key
        return group

    def gather(self, index, n):
        """The map of ``self[index]``, where the source has ``n`` rows.

        A gather of a gather composes the maps; the result is still one
        map, so the representation does not deepen with the expression.

        The unmapped case keeps the caller's own index array rather than
        computing ``arange(n)[index]``, which is equal to it.  That is not
        a micro-optimisation: sharing the array is what lets two gathers
        taken with the same index recognise each other and merge, and
        building an equal-but-distinct copy would quietly turn the merge
        off.  Anything that is not already an integer array -- a slice, a
        boolean mask, a scalar -- has to be resolved against the row range,
        and is rare enough not to matter.
        """
        if self.map is None:
            if isinstance(index, _np.ndarray) and index.dtype.kind in 'iu':
                return _Group(self.offsets, index)
            resolved = _np.arange(n, dtype=_np.int64)[index]
            return _Group(self.offsets, _np.atleast_1d(resolved))
        return _Group(self.offsets, _np.atleast_1d(self.map[index]))

    def to_array(self, n):
        """Column indices ``(n, nvariable)``, built only when a matrix is."""
        offsets = _np.asarray(self.offsets, dtype=_np.int64)
        rows = self.rows(n)
        return rows[:, None] + offsets[None, :]

    def rows(self, n):
        """The map itself, as an explicit array of source rows."""
        if self.map is None:
            return _np.arange(n, dtype=_np.int64)
        return _np.asarray(self.map, dtype=_np.int64).reshape(-1)

    def broadcast(self, size):
        """The map for a length-one value repeated to ``size`` rows."""
        if self.map is None:
            return _Group(self.offsets, _np.zeros(size, dtype=_np.int64))
        first = _np.asarray(self.map).reshape(-1)[0]
        return _Group(self.offsets, _np.full(size, first, dtype=_np.int64))

    def same_columns(self, other):
        """Whether two groups address exactly the same matrix entries."""
        return self.key == other.key and self.offsets == other.offsets


class DiagonalADI(_ADValue):
    """A per-row-sparse ("generalized diagonal") automatic-differentiation
    value: ``val`` (n,) plus ``groups``, a list of ``(group, deriv)`` pairs
    where ``group`` is a :class:`_Group` naming a row map and a set of
    primary-variable offsets, and ``deriv`` is the dense
    ``(n, len(group.offsets))`` array of derivatives in them.
    """

    __slots__ = ("val", "groups", "_nvar")

    def __init__(self, value, groups, nvar: int, compacted: bool = False):
        self.val = _np.asarray(value, dtype=float).reshape(-1)
        # ``compacted`` is an assertion by the caller that the list is
        # already canonical -- one entry per map, each derivative already
        # two-dimensional.  Every operator below produces exactly that, and
        # re-deriving it would put a dictionary build on the hot path of an
        # expression that has thousands of nodes.
        self.groups = groups if compacted else self._compact(groups)
        self._nvar = int(nvar)

    # ------------------------------------------------------------------
    # Canonical form
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(group, deriv):
        """One ``(nrow, nvariable)`` array, whatever shape arrived.

        A scalar index -- ``bW[cell]`` in the well model, one perforation
        at a time -- leaves the derivatives one-dimensional, and so does a
        single-variable group.  ``val`` is reshaped on the way in, so
        without the same treatment here the object looks fine until
        ``to_sparse`` tries to stack the groups and reports something about
        dimensions, a long way from the indexing that caused it.
        """
        deriv = _np.asarray(deriv, dtype=float)
        width = len(group.offsets)
        if deriv.ndim == 2:
            return deriv
        if deriv.ndim == 0:
            return deriv.reshape(1, 1)
        if width == 1:
            return deriv.reshape(-1, 1)
        if deriv.size == width:
            return deriv.reshape(1, width)
        raise ValueError('derivative array of shape %r does not fit %d '
                         'variables' % (deriv.shape, width))

    @classmethod
    def _compact(cls, groups):
        """One group per row map, with the variables as its columns.

        ``__add__`` and the product rule in ``__mul__`` both grow the list
        by concatenation, so its length would otherwise track the *depth*
        of the expression rather than the number of dependencies: a
        black-oil property chain reached a hundred-odd entries over three
        variables.  Every later operation then costs a pass per entry, and
        ``to_sparse`` pays for each one.  MRST's ``DiagonalJacobian`` never
        has this problem -- it keeps one dense column per primary variable
        and accumulates into it, which is what merging on the map produces
        here.
        """
        if not groups:
            return []
        if len(groups) == 1:
            group, deriv = groups[0]
            return [(group, cls._normalise(group, deriv))]
        merged = {}
        order = []
        for group, deriv in groups:
            deriv = cls._normalise(group, deriv)
            previous = merged.get(group.key)
            if previous is None:
                merged[group.key] = (group, deriv)
                order.append(group.key)
            else:
                merged[group.key] = cls._merge(previous[0], previous[1],
                                               group, deriv)
        return [merged[key] for key in order]

    @staticmethod
    def _merge(left_group, left, right_group, right):
        """Add two groups that share a row map into one.

        Equal offsets is the overwhelmingly common case -- both operands of
        a product depend on the same three primary variables -- and is a
        plain array addition.  Unequal offsets happen where a chain picks
        up a variable partway through, and need the union.
        """
        if left_group.offsets == right_group.offsets:
            return left_group, left + right
        offsets = sorted(set(left_group.offsets) | set(right_group.offsets))
        position = {offset: index for index, offset in enumerate(offsets)}
        nrow = max(left.shape[0], right.shape[0])
        out = _np.zeros((nrow, len(offsets)))
        for source_group, source in ((left_group, left), (right_group, right)):
            for column, offset in enumerate(source_group.offsets):
                out[:, position[offset]] += source[:, column]
        return left_group.with_offsets(offsets), out

    # ------------------------------------------------------------------
    @property
    def nvar(self) -> int:
        return self._nvar

    @classmethod
    def variable(cls, value, nvar, offset, group=None):
        """Seed a primary variable.

        ``group`` is the offsets of *every* variable this one is seeded
        alongside -- all the cell variables of a black-oil system, say.
        Giving it makes the value full width from the start, with a one in
        its own column and zeros in its siblings', which is what MRST's
        ``initVariablesAD`` produces for a ``DiagonalJacobian``: one width
        for the whole group, fixed before any arithmetic happens.

        Leaving it out stores one column, which is honest about the
        dependency but turns out to be the wrong trade.  Two values over
        different variable subsets cannot be added or multiplied without
        first widening both to the union -- an allocation and a copy per
        operation -- and a property chain hits that on nearly every step,
        which cost more than the columns of zeros it was avoiding.  It is
        also the reason to prefer MRST's arrangement rather than derive
        one: the fixed width is what makes every operand of every operation
        the same shape, and only then can the compiled product rule take a
        pair of them whole.
        """
        value = _np.asarray(value, dtype=float).reshape(-1)
        n = value.size
        offset = int(offset)
        if group is None:
            offsets = (offset,)
        else:
            offsets = tuple(sorted({int(o) for o in group} | {offset}))
        deriv = _np.zeros((n, len(offsets)))
        deriv[:, offsets.index(offset)] = 1.0
        return cls(value, [(_Group(offsets), deriv)], nvar, compacted=True)

    @classmethod
    def constant(cls, value, nvar):
        value = _np.asarray(value, dtype=float).reshape(-1)
        return cls(value, [], nvar, compacted=True)

    def copy(self) -> "DiagonalADI":
        # _Group is immutable, so only the derivatives need copying.
        return DiagonalADI(self.val.copy(),
                           [(g, d.copy()) for g, d in self.groups],
                           self._nvar, compacted=True)

    def to_sparse(self) -> _SparseADI:
        """Materialize the full sparse Jacobian (port of ``ADI`` conversion).

        One COO built from every group's entries at once, rather than a
        sparse matrix per group summed in a loop: each of those additions
        walks both operands' full structure, so the loop cost grew with the
        square of the group count for no gain.  COO to CSR sums duplicate
        (row, column) entries, which is what two groups landing on the same
        column mean.
        """
        import scipy.sparse as sp

        n = self.val.size
        if not self.groups:
            return _SparseADI(self.val, sp.csr_matrix((n, self._nvar)))
        rows = _np.arange(n)
        all_rows, all_columns, all_values = [], [], []
        for group, deriv in self.groups:
            all_rows.append(_np.repeat(rows, deriv.shape[1]))
            all_columns.append(group.to_array(n).reshape(-1))
            all_values.append(deriv.reshape(-1))
        if len(all_rows) == 1:
            row, column, value = all_rows[0], all_columns[0], all_values[0]
        else:
            row = _np.concatenate(all_rows)
            column = _np.concatenate(all_columns)
            value = _np.concatenate(all_values)
        jac = sp.coo_matrix((value, (row, column)),
                            shape=(n, self._nvar)).tocsr()
        return _SparseADI(self.val, jac)

    def dense_derivatives(self, offsets):
        """``(n, len(offsets))`` derivatives against the given variables.

        The face operators want a cell value's derivatives laid out one
        column per variable group, which is what this representation now
        holds -- so the common case hands the array straight over instead
        of copying it a column at a time.  A value that has been gathered
        is not cell-length any more and has no such reading; the caller
        checks for that, and the error here is the backstop.
        """
        offsets = tuple(int(offset) for offset in offsets)
        n = self.val.size
        if len(self.groups) == 1:
            group, deriv = self.groups[0]
            if group.map is None and group.offsets == offsets:
                return deriv
        out = _np.zeros((n, len(offsets)))
        position = {offset: index for index, offset in enumerate(offsets)}
        for group, deriv in self.groups:
            if group.map is not None:
                raise ValueError('cannot read cell derivatives from a value '
                                 'that has been gathered')
            for column, offset in enumerate(group.offsets):
                index = position.get(offset)
                if index is not None:
                    out[:, index] += deriv[:, column]
        return out

    # ------------------------------------------------------------------
    # Broadcasting helpers
    # ------------------------------------------------------------------
    def _broadcast(self, size: int) -> "DiagonalADI":
        if self.val.size == size:
            return self
        if self.val.size != 1:
            raise ValueError("DiagonalADI vectors have incompatible lengths")
        groups = [(g.broadcast(size), _np.repeat(d, size, axis=0))
                  for g, d in self.groups]
        return DiagonalADI(_np.full(size, self.val[0]), groups, self._nvar,
                           compacted=True)

    @staticmethod
    def _pair(left, right):
        if isinstance(left, DiagonalADI) and isinstance(right, DiagonalADI):
            if left.nvar != right.nvar:
                raise ValueError("DiagonalADI variables use different primary vectors")
            n = max(left.val.size, right.val.size)
            return left._broadcast(n), right._broadcast(n)
        if isinstance(left, DiagonalADI):
            return left, _np.asarray(right, dtype=float)
        if isinstance(right, DiagonalADI):
            return _np.asarray(left, dtype=float), right
        return left, right

    @staticmethod
    def _defers_to_sparse(other):
        """Whether ``other`` is an assembled-Jacobian value.

        Where the two representations meet -- a residual adding a flux term
        that has already gone through ``linear_map`` to a well source term
        that has not -- the general one wins.  Each operator below hands the
        expression over rather than trying to coerce a sparse matrix into a
        set of diagonals, which it cannot represent.
        """
        return isinstance(other, _SparseADI)

    # ------------------------------------------------------------------
    # Arithmetic (stays diagonal: scaling every group's derivatives by a
    # per-row factor, or concatenating both operands' group lists, never
    # changes which columns a row depends on)
    # ------------------------------------------------------------------
    @staticmethod
    def _scaled(deriv, factor):
        """``deriv * factor[:, None]``, through the kernel where it fits.

        The kernel wants one factor per row of a contiguous array; a
        length-one factor broadcasting over many rows, or a derivative that
        a gather left non-contiguous, goes the numpy way.
        """
        if (_KERNEL is not None and deriv.shape[0] == factor.size
                and deriv.flags['C_CONTIGUOUS']):
            return _KERNEL.diag_mult(deriv, factor)
        return deriv * factor.reshape(-1, 1)

    def scale_rows(self, factor):
        """Port of :meth:`ADValue.scale_rows` for the generalized diagonals.

        Every group holds one derivative per row per variable, so scaling
        row ``i`` is a plain elementwise multiply -- no sparse structure is
        touched, which is the whole reason this representation is cheaper
        than an assembled Jacobian for a long property chain.
        """
        factor = _np.asarray(factor, dtype=float).reshape(-1)
        return DiagonalADI(self.val,
                           [(g, self._scaled(d, factor)) for g, d in self.groups],
                           self._nvar, compacted=True)

    @classmethod
    def combine_scaled(cls, value, terms):
        """Port of :meth:`ADValue.combine_scaled` for generalized diagonals."""
        groups = []
        nvar = None
        for ad, factor in terms:
            factor = _np.asarray(factor, dtype=float).reshape(-1)
            groups.extend((g, cls._scaled(d, factor)) for g, d in ad.groups)
            nvar = ad.nvar
        return cls(value, groups, nvar)

    def __neg__(self):
        return DiagonalADI(-self.val, [(g, -d) for g, d in self.groups],
                           self._nvar, compacted=True)

    @staticmethod
    def _align(left, right):
        """``(group, left_deriv, right_deriv)`` when both sides match.

        The overwhelmingly common shape: both operands carry exactly one
        group, over the same map and the same variables.  Everything below
        can then work on the two dense arrays directly -- which is what the
        compiled kernels need -- instead of going through the general merge.
        """
        if len(left.groups) != 1 or len(right.groups) != 1:
            return None
        left_group, left_deriv = left.groups[0]
        right_group, right_deriv = right.groups[0]
        if not left_group.same_columns(right_group):
            return None
        return left_group, left_deriv, right_deriv

    def __add__(self, other):
        if self._defers_to_sparse(other):
            return self.to_sparse() + other
        left, right = self._pair(self, other)
        if isinstance(right, DiagonalADI):
            aligned = self._align(left, right)
            if aligned is not None:
                group, left_deriv, right_deriv = aligned
                return DiagonalADI(left.val + right.val,
                                   [(group, left_deriv + right_deriv)],
                                   left._nvar, compacted=True)
            return DiagonalADI(left.val + right.val,
                               left.groups + right.groups, left._nvar)
        return DiagonalADI(left.val + right, left.groups, left._nvar,
                           compacted=True)

    __radd__ = __add__

    def __sub__(self, other):
        if self._defers_to_sparse(other):
            return self.to_sparse() - other
        return self + (-other if isinstance(other, DiagonalADI) else -_np.asarray(other, dtype=float))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if self._defers_to_sparse(other):
            return self.to_sparse() * other
        left, right = self._pair(self, other)
        if isinstance(right, DiagonalADI):
            # Product rule: d(f*g) = g*df + f*dg.  Both terms are still one
            # column per row per variable (scaling a group's derivatives by
            # the OTHER operand's per-row *value* array).
            aligned = self._align(left, right)
            if aligned is not None:
                group, left_deriv, right_deriv = aligned
                if (_KERNEL is not None and left_deriv.flags['C_CONTIGUOUS']
                        and right_deriv.flags['C_CONTIGUOUS']
                        and left_deriv.shape == right_deriv.shape):
                    deriv = _KERNEL.diag_product_mult(right.val, left_deriv,
                                                      left.val, right_deriv)
                else:
                    deriv = (left_deriv * right.val.reshape(-1, 1)
                             + right_deriv * left.val.reshape(-1, 1))
                return DiagonalADI(left.val * right.val, [(group, deriv)],
                                   left._nvar, compacted=True)
            groups = ([(g, self._scaled(d, right.val)) for g, d in left.groups]
                      + [(g, self._scaled(d, left.val)) for g, d in right.groups])
            return DiagonalADI(left.val * right.val, groups, left._nvar)
        right = _np.asarray(right, dtype=float).reshape(-1)
        if right.size == 1:
            scalar = float(right[0])
            return DiagonalADI(left.val * scalar,
                               [(g, d * scalar) for g, d in left.groups],
                               left._nvar, compacted=True)
        return DiagonalADI(left.val * right,
                           [(g, self._scaled(d, right)) for g, d in left.groups],
                           left._nvar, compacted=True)

    __rmul__ = __mul__

    def _chain(self, value, scale):
        """A scalar elementwise function's result: new value, scaled rows."""
        scale = _np.asarray(scale, dtype=float).reshape(-1)
        return DiagonalADI(value,
                           [(g, self._scaled(d, scale)) for g, d in self.groups],
                           self._nvar, compacted=True)

    def reciprocal(self) -> "DiagonalADI":
        inv = 1.0 / self.val
        return self._chain(inv, -inv * inv)

    def __truediv__(self, other):
        if self._defers_to_sparse(other):
            return self.to_sparse() / other
        if isinstance(other, DiagonalADI):
            return self * other.reciprocal()
        return self * (1.0 / _np.asarray(other, dtype=float))

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    def __pow__(self, exponent):
        exponent = float(_np.asarray(exponent, dtype=float))
        return self._chain(self.val**exponent,
                           exponent * self.val ** (exponent - 1.0))

    def exp(self) -> "DiagonalADI":
        value = _np.exp(self.val)
        return self._chain(value, value)

    def log(self) -> "DiagonalADI":
        return self._chain(_np.log(self.val), 1.0 / self.val)

    # ------------------------------------------------------------------
    # Indexing: a fancy/array gather (e.g. property[upstream_cell]) keeps
    # exactly one dependency per output row per variable, just relabeled --
    # still diagonal. A slice is the same special case with an implicit
    # arange index.
    # ------------------------------------------------------------------
    def __getitem__(self, index) -> "DiagonalADI":
        n = self.val.size
        val = self.val[index]
        groups = [(g.gather(index, n), self._normalise(g, d[index]))
                  for g, d in self.groups]
        return DiagonalADI(val, groups, self._nvar, compacted=True)

    # ------------------------------------------------------------------
    # Operations that do NOT preserve the diagonal structure in general:
    # fall back to a materialized SparseADI.
    # ------------------------------------------------------------------
    def sum(self) -> _SparseADI:
        """The total, and its derivative -- as a single sparse row.

        Collapsing every row into one leaves a derivative that depends on
        as many columns as the whole vector did, so the result cannot be a
        generalized diagonal and comes back assembled.  A well's surface
        rate is summed over its perforations this way.
        """
        return self.to_sparse().sum()

    def linear_map(self, matrix) -> _SparseADI:
        return self.to_sparse().linear_map(matrix)

    @staticmethod
    def concat(items):
        converted = [item.to_sparse() if isinstance(item, DiagonalADI) else item for item in items]
        return _SparseADI.concat(converted)

    @staticmethod
    def scatter(indices, part, size):
        """Insert a row subset into a zero vector of ``size`` rows.

        This one *does* stay diagonal, unlike ``linear_map`` and ``concat``
        above: the rows that were not written have no derivative at all, and
        the rows that were keep exactly the columns they had in ``part``.
        Materialising a sparse matrix here -- which is what this used to do
        -- forced every caller that scatters in a loop (the PVTO/PVTG region
        assembly builds its tables that way) off the diagonal representation
        for the whole rest of the chain.

        Untouched rows are parked on the group's own offsets with a zero
        coefficient, which contributes nothing to the assembled Jacobian.

        That is only worth doing when most rows are written.  A diagonal
        group is dense in the row count, so scattering a handful of rows
        into a whole grid stores ``size`` numbers to represent a few --
        which is what the well model does, one perforation at a time, and
        it made a full assembly *slower* than the sparse representation it
        was supposed to beat.  Below the threshold the sparse form, which
        stores only the entries that exist, is the right one.
        """
        indices = _np.asarray(indices, dtype=int).reshape(-1)
        if not isinstance(part, DiagonalADI):
            return _SparseADI.scatter(indices, part, size)
        if indices.size != part.val.size:
            raise ValueError("ADI scatter indices and subset rows must agree")
        size = int(size)
        if indices.size < 0.5 * size:
            return _SparseADI.scatter(indices, part.to_sparse(), size)
        value = _np.zeros(size, dtype=float)
        value[indices] = part.val
        groups = []
        nrow = part.val.size
        for group, deriv in part.groups:
            # The map has to be materialised here: rows outside ``indices``
            # are not a gather of anything, they are absent, and they get
            # parked on the group's own offsets with a zero coefficient.
            full_map = _np.zeros(size, dtype=_np.int64)
            full_deriv = _np.zeros((size, deriv.shape[1]))
            full_map[indices] = group.rows(nrow)
            full_deriv[indices] = deriv
            groups.append((_Group(group.offsets, full_map), full_deriv))
        return DiagonalADI(value, groups, part.nvar, compacted=True)
