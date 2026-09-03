"""Face values that know they depend on two cells, and the divergence of them.

Port of the representation behind MRST's ``FixedWidthJacobian`` and
``discreteDivergence`` (mrst-2026a/autodiff/ad-core/backends/diagonal).

A TPFA flux across a face depends on exactly two cells -- the ones the face
separates -- and on the handful of primary variables each of them carries.
MRST stores that as a *dense* array: one row per face, one column per
(side, variable) pair, with the neighbour table supplying the cell indices.
Nothing about which columns are involved needs storing, because the layout
says it.

The generalized-diagonal representation in
:mod:`PRSTCore.ad_core.diagonal_adi` cannot express that.  It keeps a list
of ``(columns, values)`` blocks, and every gather -- ``lam[upstream]``,
``bW[upstream]``, ``rho[c1]`` -- adds one.  A black-oil flux term reached
33 blocks on a face-length value, and turning that into the sparse matrix
the divergence operator wanted cost more than never leaving the sparse
representation at all: measured on Norne, 0.695 of the 1.32 seconds an
assembly took, which is why the diagonal backend came out *slower* than the
sparse one it was meant to beat.

Here the width is fixed at ``2 * nvariable`` no matter how deep the
expression goes, so:

* arithmetic is one operation on a dense array rather than one per block;
* the divergence assembles the cell-length Jacobian straight from that
  array, without ever building the face-length one.

What this module does not do is MRST's block-CSR output format
(``getBlockSystemCSR``) or its Schur complement over the well unknowns.
Those exist to feed AMGCL's block solver; PETSc takes an ordinary CSR
matrix, so the assembly stops at one.
"""

from __future__ import annotations

import numpy as _np

from .adi import SparseADI as _SparseADI, is_ad as _is_ad

try:
    import scipy.sparse as _sp
except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere
    _sp = None

#: Which side of a face a gather takes its value from.
LEFT, RIGHT = 0, 1


def _face_kernel():
    """The compiled face arithmetic, looked up once.

    A missing kernel costs speed and nothing else: every operation below has
    a numpy twin, and the two are checked against each other.
    """
    from . import mex as _mex
    return _mex.load_face_operators()


_KERNEL = _face_kernel()


class CellVariableLayout:
    """Where each cell variable sits in the primary-variable vector.

    Deck-derived systems group by variable -- every cell's pressure, then
    every cell's water saturation, then the third -- so variable ``k`` of
    cell ``c`` is column ``offsets[k] + c``.  Everything here is written
    against that; an interleaved ordering would need different offsets and
    nothing else.
    """

    __slots__ = ('ncell', 'offsets', 'nvar')

    def __init__(self, ncell, ngroup, nvar):
        self.ncell = int(ncell)
        self.offsets = _np.arange(ngroup, dtype=_np.int64) * int(ncell)
        self.nvar = int(nvar)

    @property
    def ngroup(self):
        return int(self.offsets.size)

    def columns(self, cells):
        """The columns cell indices ``cells`` occupy, one row per variable."""
        cells = _np.asarray(cells, dtype=_np.int64)
        return cells[None, :] + self.offsets[:, None]


class FaceValue:
    """A per-face value and its derivatives with respect to its two cells.

    ``val`` has one entry per face.  ``deriv`` has shape
    ``(nface, 2, ngroup)``: ``deriv[f, side, k]`` is the derivative of face
    ``f``'s value with respect to variable ``k`` in the cell on that side,
    where side 0 is ``neighbours[f, 0]`` and side 1 is ``neighbours[f, 1]``.

    A face value that depends on only one of its cells -- an upstream-
    weighted mobility, say -- carries zeros on the other side rather than a
    different shape.  That is what keeps every face value combinable with
    every other without bookkeeping, and it is the same trade MRST makes.
    """

    __slots__ = ('val', 'deriv', 'layout', 'neighbours')

    __array_ufunc__ = None

    def __init__(self, val, deriv, layout, neighbours):
        self.val = _np.asarray(val, dtype=float).reshape(-1)
        self.deriv = _np.asarray(deriv, dtype=float)
        self.layout = layout
        self.neighbours = neighbours
        if self.deriv.shape != (self.val.size, 2, layout.ngroup):
            raise ValueError(
                'face derivatives must have shape (nface, 2, ngroup); got %r '
                'for %d faces and %d variables'
                % (self.deriv.shape, self.val.size, layout.ngroup))

    # ------------------------------------------------------------ builders --
    @classmethod
    def constant(cls, values, layout, neighbours):
        values = _np.asarray(values, dtype=float).reshape(-1)
        nface = neighbours.shape[0]
        if values.size == 1:
            values = _np.full(nface, values[0])
        return cls(values, _np.zeros((values.size, 2, layout.ngroup)),
                   layout, neighbours)

    @classmethod
    def gather(cls, cell_value, layout, neighbours, side):
        """One side of every face, taken from a cell-length value.

        ``side`` is :data:`LEFT`, :data:`RIGHT`, or a per-face boolean where
        True selects the left cell -- the upstream weighting.
        """
        cells, deriv = _cell_derivatives(cell_value, layout)
        nface = neighbours.shape[0]

        if isinstance(side, (int, _np.integer)):
            index = neighbours[:, int(side)]
            if _KERNEL is not None:
                take = _KERNEL.left_jac if int(side) == 0 else _KERNEL.right_jac
                return cls(cells[index], take(deriv, neighbours), layout, neighbours)
            out = _np.zeros((nface, 2, layout.ngroup))
            out[:, int(side), :] = deriv[index, :]
            return cls(cells[index], out, layout, neighbours)

        take_left = _np.asarray(side, dtype=bool).reshape(-1)
        index = _np.where(take_left, neighbours[:, 0], neighbours[:, 1])
        if _KERNEL is not None:
            return cls(cells[index],
                       _KERNEL.upwind_jac(deriv, neighbours, take_left),
                       layout, neighbours)
        out = _np.zeros((nface, 2, layout.ngroup))
        rows = _np.arange(nface)
        out[rows, _np.where(take_left, 0, 1), :] = deriv[index, :]
        return cls(cells[index], out, layout, neighbours)

    @classmethod
    def gradient(cls, cell_value, layout, neighbours):
        """``value[right] - value[left]`` across every face at once.

        MRST's two-point gradient.  Doing it as one operation rather than
        two gathers and a subtraction halves the traffic, and it is the
        shape every flux potential starts from.
        """
        cells, deriv = _cell_derivatives(cell_value, layout)
        left, right = neighbours[:, 0], neighbours[:, 1]
        values = cells[right] - cells[left]
        if _KERNEL is not None:
            return cls(values, _KERNEL.two_point_gradient_jac(deriv, neighbours),
                       layout, neighbours)
        out = _np.zeros((neighbours.shape[0], 2, layout.ngroup))
        out[:, 0, :] = -deriv[left, :]
        out[:, 1, :] = deriv[right, :]
        return cls(values, out, layout, neighbours)

    @classmethod
    def average(cls, cell_value, layout, neighbours):
        """``(value[left] + value[right]) / 2`` -- MRST's face average."""
        cells, deriv = _cell_derivatives(cell_value, layout)
        left, right = neighbours[:, 0], neighbours[:, 1]
        values = 0.5 * (cells[left] + cells[right])
        if _KERNEL is not None:
            return cls(values, _KERNEL.face_average_jac(deriv, neighbours),
                       layout, neighbours)
        out = _np.zeros((neighbours.shape[0], 2, layout.ngroup))
        out[:, 0, :] = 0.5 * deriv[left, :]
        out[:, 1, :] = 0.5 * deriv[right, :]
        return cls(values, out, layout, neighbours)

    # ---------------------------------------------------------- arithmetic --
    def _pair(self, other):
        if isinstance(other, FaceValue):
            return other.val, other.deriv
        value = _np.asarray(other, dtype=float)
        if value.ndim == 0:
            value = value.reshape(1)
        return value.reshape(-1), None

    def __neg__(self):
        return FaceValue(-self.val, -self.deriv, self.layout, self.neighbours)

    def __add__(self, other):
        value, deriv = self._pair(other)
        return FaceValue(self.val + value,
                         self.deriv if deriv is None else self.deriv + deriv,
                         self.layout, self.neighbours)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other if isinstance(other, FaceValue)
                       else -_np.asarray(other, dtype=float))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        value, deriv = self._pair(other)
        if deriv is None:
            if _KERNEL is not None and value.size == self.val.size:
                scaled = _KERNEL.diag_mult(self.deriv, value)
            else:
                scaled = self.deriv * value[:, None, None]
            return FaceValue(self.val * value, scaled, self.layout, self.neighbours)
        # The product rule, fused: one pass over both derivative arrays
        # rather than two multiplies and an add, each allocating.
        if _KERNEL is not None:
            product = _KERNEL.diag_product_mult(value, self.deriv, self.val, deriv)
        else:
            product = (self.deriv * value[:, None, None]
                       + deriv * self.val[:, None, None])
        return FaceValue(self.val * value, product, self.layout, self.neighbours)

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = 1.0 / self.val
        return FaceValue(inverse, self.deriv * (-inverse * inverse)[:, None, None],
                         self.layout, self.neighbours)

    def __truediv__(self, other):
        if isinstance(other, FaceValue):
            return self * other.reciprocal()
        return self * (1.0 / _np.asarray(other, dtype=float))

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    # ------------------------------------------------------------- outputs --
    def divergence(self, ncell):
        """``sum of fluxes out of each cell``, as a cell-length AD value.

        Port of ``discreteDivergence``: cell ``neighbours[f, 0]`` gains face
        ``f``'s flux and cell ``neighbours[f, 1]`` loses it.  The Jacobian is
        built in a single pass from the dense derivative array -- there is no
        face-length matrix in between, which is the whole point.

        The result is genuinely sparse: a cell's row has one entry per
        variable per neighbouring face, so it is no longer expressible as a
        fixed number of diagonals.  That is the boundary MRST draws too.
        """
        if _sp is None:
            raise RuntimeError('scipy is required to assemble a divergence')
        left, right = self.neighbours[:, 0], self.neighbours[:, 1]
        nface = self.val.size
        ngroup = self.layout.ngroup

        value = (_np.bincount(left, weights=self.val, minlength=ncell)
                 - _np.bincount(right, weights=self.val, minlength=ncell))

        # Four contributions per face per variable: each of the two cells
        # receives the derivative with respect to each of the two cells,
        # with the sign of the cell that receives it.
        rows = _np.concatenate([_np.repeat(left, ngroup),
                                _np.repeat(left, ngroup),
                                _np.repeat(right, ngroup),
                                _np.repeat(right, ngroup)])
        columns_left = (_np.repeat(left, ngroup)
                        + _np.tile(self.layout.offsets, nface))
        columns_right = (_np.repeat(right, ngroup)
                         + _np.tile(self.layout.offsets, nface))
        cols = _np.concatenate([columns_left, columns_right,
                                columns_left, columns_right])
        d_left = self.deriv[:, 0, :].ravel()
        d_right = self.deriv[:, 1, :].ravel()
        vals = _np.concatenate([d_left, d_right, -d_left, -d_right])

        jacobian = _sp.coo_matrix((vals, (rows, cols)),
                                  shape=(ncell, self.layout.nvar)).tocsr()
        return _SparseADI(value, jacobian)

    def to_sparse(self):
        """This face value as an ordinary face-length :class:`SparseADI`.

        Only for code that has not been converted; the point of the class is
        to reach :meth:`divergence` without passing through here.
        """
        if _sp is None:
            raise RuntimeError('scipy is required to materialise a face value')
        nface = self.val.size
        ngroup = self.layout.ngroup
        rows = _np.repeat(_np.arange(nface), 2 * ngroup)
        cols = _np.concatenate([
            (self.neighbours[:, 0][:, None] + self.layout.offsets[None, :]),
            (self.neighbours[:, 1][:, None] + self.layout.offsets[None, :]),
        ], axis=1).ravel()
        vals = self.deriv.reshape(nface, 2 * ngroup).ravel()
        jacobian = _sp.coo_matrix((vals, (rows, cols)),
                                  shape=(nface, self.layout.nvar)).tocsr()
        return _SparseADI(self.val, jacobian)


class DivergenceAssembler:
    """The sparsity of ``div(flux)``, worked out once for a grid.

    Port of what ``getMexDiscreteDivergenceJacPrecomputes`` prepares and
    ``mexDiscreteDivergenceJac`` consumes.  The pattern of the assembled
    matrix depends only on the neighbour table, so it can be built once and
    reused for every Newton iteration of every timestep; only the numbers
    change.

    Without the precompute the assembly has to hand scipy an unordered
    ``(row, column, value)`` list and let it sort -- four million entries
    per phase on Norne -- and that sort is slower than the sparse matrix
    product it was supposed to replace.  Measured: 0.42 s against 0.21 s.
    With it, each assembly is one ``bincount`` into a fixed array.

    The fixed pattern is worth having for its own sake as well: it never
    changes between Newton iterations, so PETSc can keep the preconditioner
    setup it built from the first one.
    """

    __slots__ = ('ncell', 'layout', 'neighbours', 'indptr', 'indices',
                 'nnz', '_positions', '_kernel', '_face_pos', '_faces',
                 '_cells', '_cell_index')

    def __init__(self, neighbours, ncell, layout, use_kernel=True):
        if _sp is None:
            raise RuntimeError('scipy is required to plan a divergence')
        self.ncell = int(ncell)
        self.layout = layout
        self.neighbours = _np.asarray(neighbours, dtype=_np.int64)

        from . import mex as _mex
        self._kernel = _mex.load_discrete_divergence() if use_kernel else None
        if self._kernel is not None:
            (self._face_pos, self._faces, self._cells,
             self._cell_index) = divergence_precomputes(self.neighbours, self.ncell)
            self.indptr = None
            self.indices = None
            self.nnz = 0
            self._positions = None
            return
        self._face_pos = self._faces = self._cells = self._cell_index = None

        rows, cols = self._pattern()
        # Order the entries the way CSR wants them, then collapse the
        # duplicates -- two faces of the same cell contributing to the same
        # column -- recording where each one lands.
        order = _np.lexsort((cols, rows))
        sorted_rows, sorted_cols = rows[order], cols[order]
        first = _np.ones(sorted_rows.size, dtype=bool)
        first[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | \
                    (sorted_cols[1:] != sorted_cols[:-1])
        slot = _np.cumsum(first) - 1

        self.nnz = int(slot[-1]) + 1 if slot.size else 0
        self.indices = sorted_cols[first]
        counts = _np.bincount(sorted_rows[first], minlength=self.ncell)
        self.indptr = _np.zeros(self.ncell + 1, dtype=_np.int64)
        _np.cumsum(counts, out=self.indptr[1:])

        positions = _np.empty(order.size, dtype=_np.int64)
        positions[order] = slot
        self._positions = positions

    def _pattern(self):
        """Every ``(row, column)`` a face contributes to, with repeats."""
        left, right = self.neighbours[:, 0], self.neighbours[:, 1]
        nface = left.size
        ngroup = self.layout.ngroup
        offsets = _np.tile(self.layout.offsets, nface)
        columns_left = _np.repeat(left, ngroup) + offsets
        columns_right = _np.repeat(right, ngroup) + offsets
        rows_left = _np.repeat(left, ngroup)
        rows_right = _np.repeat(right, ngroup)
        rows = _np.concatenate([rows_left, rows_left, rows_right, rows_right])
        cols = _np.concatenate([columns_left, columns_right,
                                columns_left, columns_right])
        return rows, cols

    def assemble(self, face, accumulation=None):
        """``div(face)`` as a cell-length :class:`SparseADI`.

        The compiled kernel does the whole assembly when it is available.
        The pure-Python path below sums into precomputed slots, so it too
        never sorts; it is there so a build without the extension is slower
        and not wrong, and the two are checked against each other.

        ``accumulation`` is the cell-local part of the same equation, shaped
        ``(ncell, ngroup)``.  MRST folds it into the diagonal here rather
        than adding a second matrix afterwards, which is the difference
        between one pass and two.  Only the compiled path takes it.
        """
        if self._kernel is not None:
            return self._assemble_compiled(face, accumulation)
        left, right = self.neighbours[:, 0], self.neighbours[:, 1]
        value = (_np.bincount(left, weights=face.val, minlength=self.ncell)
                 - _np.bincount(right, weights=face.val, minlength=self.ncell))
        d_left = face.deriv[:, 0, :].ravel()
        d_right = face.deriv[:, 1, :].ravel()
        contributions = _np.concatenate([d_left, d_right, -d_left, -d_right])
        data = _np.bincount(self._positions, weights=contributions,
                            minlength=self.nnz)
        jacobian = _sp.csr_matrix((data, self.indices, self.indptr),
                                  shape=(self.ncell, self.layout.nvar))
        if accumulation is not None:
            jacobian = jacobian + _accumulation_matrix(
                accumulation, self.ncell, self.layout)
        return _SparseADI(value, jacobian)

    def _assemble_compiled(self, face, accumulation=None):
        """MRST's ``mexDiscreteDivergenceJac``, through the built kernel.

        The kernel is asked for CSR.  MRST's original writes CSC, because
        a MATLAB sparse matrix is CSC; converting one here cost more than
        the assembly saved, so the CSR variant exists alongside it.
        """
        left, right = self.neighbours[:, 0], self.neighbours[:, 1]
        value = (_np.bincount(left, weights=face.val, minlength=self.ncell)
                 - _np.bincount(right, weights=face.val, minlength=self.ncell))
        diagonal = _np.ascontiguousarray(face.deriv, dtype=float)
        acc = (None if accumulation is None
               else _np.ascontiguousarray(accumulation, dtype=float))
        data, indices, indptr = self._kernel.divergence_jac(
            int(face.val.size), int(self.ncell), int(self.layout.ngroup),
            self._face_pos, self._faces, self._cells, self._cell_index,
            acc, diagonal, True)
        jacobian = _sp.csr_matrix((data, indices, indptr),
                                  shape=(self.ncell, self.layout.nvar))
        return _SparseADI(value, jacobian)


def _accumulation_matrix(accumulation, ncell, layout):
    """A cell-local ``(ncell, ngroup)`` term as a diagonal-per-group matrix."""
    accumulation = _np.asarray(accumulation, dtype=float)
    cells = _np.arange(ncell)
    rows = _np.tile(cells, layout.ngroup)
    cols = _np.concatenate([cells + int(offset) for offset in layout.offsets])
    vals = _np.concatenate([accumulation[:, k] for k in range(layout.ngroup)])
    return _sp.csr_matrix((vals, (rows, cols)), shape=(ncell, layout.nvar))


def divergence_precomputes(neighbours, ncell):
    """Port of ``getMexDiscreteDivergenceJacPrecomputes.m``.

    Each face appears twice -- once seen from each of its cells -- and the
    pairs are sorted so one cell's connections are contiguous and ordered by
    the neighbour's index, which is the order CSC wants.

    ``cells`` carries the neighbour one-based *and signed*: the sign records
    which column of the neighbour table this cell sat in, and that is how
    the kernel knows whether to read the face's left or right derivative.
    ``cell_index`` says where the diagonal belongs among a cell's
    neighbours, so the kernel places it without searching.
    """
    neighbours = _np.asarray(neighbours, dtype=_np.int64)
    nface = neighbours.shape[0]
    face_ids = _np.arange(nface, dtype=_np.int64)
    # [self, other, face], both directions, one-based to match MRST.
    connections = _np.concatenate([
        _np.stack([neighbours[:, 0] + 1, neighbours[:, 1] + 1, face_ids + 1], axis=1),
        _np.stack([neighbours[:, 1] + 1, neighbours[:, 0] + 1, face_ids + 1], axis=1),
    ], axis=0)
    order = _np.lexsort((connections[:, 2], connections[:, 1], connections[:, 0]))
    connections = connections[order]

    own = connections[:, 0]
    other = connections[:, 1]
    face = connections[:, 2]
    # +1 when this cell is *not* the neighbour table's first column, -1 when it is.
    sign = 2 * (neighbours[face - 1, 0] + 1 != own).astype(_np.int64) - 1

    per_cell = _np.bincount(own - 1, minlength=ncell)
    face_pos = _np.zeros(ncell + 1, dtype=_np.int64)
    _np.cumsum(per_cell, out=face_pos[1:])
    cell_index = _np.bincount(own - 1, weights=(other < own).astype(float),
                              minlength=ncell).astype(_np.int64)
    return face_pos, face - 1, sign * other, cell_index


def _cell_derivatives(cell_value, layout):
    """``(values, deriv)`` for a cell-length value, ``deriv`` (ncell, ngroup).

    Accepts a plain array (no derivatives), a
    :class:`~PRSTCore.ad_core.diagonal_adi.DiagonalADI`, or a
    :class:`~PRSTCore.ad_core.adi.SparseADI`.  The sparse case is the
    expensive one -- it has to read a diagonal per variable group out of an
    assembled matrix -- and exists so a partially converted assembly still
    produces the right answer while it is being converted.
    """
    if not _is_ad(cell_value):
        values = _np.asarray(cell_value, dtype=float).reshape(-1)
        return values, _np.zeros((values.size, layout.ngroup))

    values = _np.asarray(cell_value.val, dtype=float).reshape(-1)
    deriv = _np.zeros((values.size, layout.ngroup))

    dense = getattr(cell_value, 'dense_derivatives', None)
    if dense is not None:
        # DiagonalADI already stores exactly this: one dense column per
        # variable group.  In the ordinary case it is the very array the
        # property chain has been accumulating into, and comes back without
        # a copy.
        return values, dense(layout.offsets)

    jacobian = getattr(cell_value, 'jac', None)
    if jacobian is not None:
        jacobian = jacobian.tocsr()
        for group in range(layout.ngroup):
            start = int(layout.offsets[group])
            block = jacobian[:, start:start + layout.ncell]
            deriv[:, group] = block.diagonal()
        return values, deriv

    raise TypeError('cannot read cell derivatives from %r' % type(cell_value))


def upwind_flag(potential):
    """True where the *left* cell of the face is upstream.

    MRST's ``faceUpstr`` takes ``N(:,1)`` when its flag is true.  The
    convention is pinned here so the flux assembly and the gather cannot
    drift apart -- getting it backwards produces a plausible-looking system
    that transports in the wrong direction.
    """
    values = potential.val if _is_ad(potential) or isinstance(potential, FaceValue) \
        else _np.asarray(potential, dtype=float)
    return _np.asarray(values, dtype=float).reshape(-1) <= 0.0
