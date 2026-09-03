import numpy as _np

from PRSTCore.ad_core.adi import SparseADI, ad_interp_linear, ad_maximum, ad_select, is_ad


class _RaggedTables:
    """Many small interpolation tables, searched all at once.

    A PVTO or PVTG keyword is a list of records, each an independent
    ``(x, y)`` table, and every cell picks one of them.  Interpolating the
    obvious way means looping over the records and scattering each one's
    rows into a full-length result -- forty-one iterations for Norne, each
    building four grid-sized AD values, so a single property evaluation
    performed some three hundred full-size sparse operations.

    The trick that removes the loop: each record's ``x`` is increasing, so
    adding a per-record offset large enough to separate them makes the
    concatenation of all records increasing too.  One ``searchsorted`` into
    that then answers "which segment of which record" for every cell at
    once, and the interpolation becomes ``y0 + slope * (query - x0)`` with
    ``x0``, ``y0`` and ``slope`` gathered per row -- two AD operations,
    whatever the number of records.

    The tables do not change, so the flattened form is built once.
    """

    __slots__ = ('start', 'x', 'y', 'offsets', 'global_x', 'span')

    def __init__(self, tables):
        xs, ys, starts = [], [], []
        position = 0
        for x, y in tables:
            x = _np.asarray(x, dtype=float).ravel()
            y = _np.asarray(y, dtype=float).ravel()
            # ad_interp_linear sorts its table; do the same so a record given
            # in descending order interpolates identically either way.
            order = _np.argsort(x)
            starts.append(position)
            xs.append(x[order])
            ys.append(y[order])
            position += x.size
        starts.append(position)

        self.start = _np.asarray(starts, dtype=_np.int64)
        self.x = _np.concatenate(xs) if xs else _np.zeros(0)
        self.y = _np.concatenate(ys) if ys else _np.zeros(0)

        # A stride wider than any record's own range, so record k's shifted
        # values cannot overlap record k+1's.  ``+1`` keeps it positive when
        # every table is a single point.
        span = float(_np.ptp(self.x)) if self.x.size else 0.0
        self.span = span + 1.0
        record_of = _np.repeat(_np.arange(len(tables), dtype=_np.int64),
                               _np.diff(self.start))
        self.global_x = self.x + record_of * self.span
        self.offsets = record_of

    def segment(self, record, query):
        """``(x0, y0, slope)`` per row, for ``query`` inside ``record``.

        Rows whose record holds a single point get a zero slope and that
        point's value, which is what ``ad_interp_linear`` does for a
        one-row table.
        """
        record = _np.asarray(record, dtype=_np.int64).ravel()
        query = _np.asarray(query, dtype=float).ravel()
        first = self.start[record]
        last = self.start[record + 1] - 1

        position = _np.searchsorted(self.global_x,
                                    query + record * self.span,
                                    side='right') - 1
        # Clip into this record, leaving room for the segment's right end.
        position = _np.clip(position, first, _np.maximum(last - 1, first))

        single = last <= first
        right = _np.where(single, position, position + 1)
        x0 = self.x[position]
        x1 = self.x[right]
        y0 = self.y[position]
        y1 = self.y[right]
        width = x1 - x0
        slope = _np.where(single, 0.0, (y1 - y0) / _np.where(width == 0.0, 1.0, width))
        return x0, y0, slope


def _interp_ragged(tables, record, query):
    """``y(query)`` in each row's own table, as one AD expression.

    ``query`` carries the derivatives; the table lookup is a value-only
    branch, exactly as ``ad_interp_linear`` treats its own bin choice.
    """
    x0, y0, slope = tables.segment(record, query.val if is_ad(query) else query)
    return query * slope + (y0 - slope * x0)

class PVTable:
    """Simple PVT table wrapper with linear interpolation.

    Assumes first column is pressure and remaining columns are properties.
    Provides `interp(P)` returning interpolated rows and `get_col(i, P)`.
    """
    def __init__(self, pressures, cols, names=None):
        self.p = _np.asarray(pressures, dtype=float)
        # sort by pressure
        order = _np.argsort(self.p)
        self.p = self.p[order]
        self.cols = _np.asarray(cols, dtype=float)
        if self.cols.ndim == 1:
            self.cols = self.cols.reshape(-1, 1)
        self.cols = self.cols[order, :]
        self.ncols = self.cols.shape[1]
        self.names = names or [f'col{i}' for i in range(self.ncols)]

    def interp(self, P):
        P_arr = _np.atleast_1d(_np.asarray(P, dtype=float))
        out = _np.zeros((P_arr.size, self.ncols), dtype=float)
        for i in range(self.ncols):
            out[:, i] = _np.interp(P_arr, self.p, self.cols[:, i], left=self.cols[0, i], right=self.cols[-1, i])
        return out if out.shape[0] > 1 else out[0]

    def get_col(self, idx, P):
        if isinstance(idx, str):
            idx = self.names.index(idx)
        return self.interp(P)[..., int(idx)]


def parse_pvt_table(tbl):
    """Parse a PVT-like table (list/array/ragged) and return a PVTable.

    Returns None if `tbl` is None or cannot be parsed.
    """
    if tbl is None:
        return None
    # If numpy array
    if isinstance(tbl, _np.ndarray):
        if tbl.ndim == 1:
            # Try to detect repeated column layout by testing interleaved offsets
            arr1 = tbl.astype(float)
            L = arr1.size
            for ncols in range(2, 9):
                # Test each possible offset where pressure might appear
                for offset in range(ncols):
                    # collect interleaved sequence for this offset
                    seq = arr1[offset::ncols]
                    if seq.size < 3:
                        continue
                    # check monotonic non-decreasing property (pressure)
                    if not _np.all(_np.diff(seq) >= -1e-12):
                        continue
                    # Try to extract complete rows using this ncols/offset
                    rows = []
                    i = 0
                    while True:
                        base = offset + i * ncols
                        if base + ncols > L:
                            break
                        row = arr1[base: base + ncols]
                        rows.append(row)
                        i += 1
                    if len(rows) >= 2:
                        mat = _np.vstack(rows)
                        pressures = mat[:, 0]
                        cols = mat[:, 1:]
                        return PVTable(pressures, cols)
            # fallback: treat as pressures only
            return PVTable(arr1, _np.zeros((arr1.size, 0)))
        elif tbl.ndim == 2:
            pressures = tbl[:, 0]
            cols = tbl[:, 1:]
            return PVTable(pressures, cols)
    # If list-like
    try:
        rows = list(tbl)
    except Exception:
        return None
    # If rows are scalar
    if all(not hasattr(r, '__len__') or isinstance(r, (str, bytes)) for r in rows):
        try:
            arr = _np.asarray(rows, dtype=float)
            return PVTable(arr, _np.zeros((arr.size, 0)))
        except Exception:
            return None
    # Rows are sequences: try uniform length
    row_lens = [len(r) if hasattr(r, '__len__') else 1 for r in rows]
    if min(row_lens) == max(row_lens):
        try:
            arr = _np.asarray(rows, dtype=float)
            pressures = arr[:, 0]
            cols = arr[:, 1:]
            return PVTable(pressures, cols)
        except Exception:
            pass
    # Ragged rows: pad with NaN
    pressures = []
    cols = []
    maxc = 0
    for r in rows:
        if hasattr(r, '__len__'):
            pressures.append(float(r[0]))
            rowcols = [_np.asarray(x, dtype=float) for x in r[1:]]
            flat = _np.concatenate([c.ravel() for c in rowcols]) if rowcols else _np.array([])
            cols.append(flat)
            if flat.size > maxc:
                maxc = flat.size
    cols_arr = _np.full((len(cols), maxc), _np.nan, dtype=float)
    for i, c in enumerate(cols):
        cols_arr[i, : c.size] = c
    return PVTable(pressures, cols_arr)


class PVTOHandler:
    """Handler for PVTO-style tables that may be Rs-indexed or P-indexed.

    Provides methods `rs_of_p(P)`, `bo_of_p(P)`, `mu_o_of_p(P)` by
    detecting whether the parsed PVTable is Rs->(P,Bo,Mu) or P->(...).
    """
    def __init__(self, pvtable: PVTable):
        self.raw = pvtable
        if pvtable is None:
            self.mode = None
            return
        # If the PVTable.p values are small (typical Rs in Mscf/stb), treat
        # them as Rs and expect cols first column to be pressure.
        median_p = _np.median(pvtable.p)
        if median_p < 100:  # heuristic: Rs-based table
            self.mode = 'rs-indexed'
            # rs_values are pvtable.p, pressures in cols[:,0]
            self.rs = pvtable.p
            self.pressures = pvtable.cols[:, 0]
            self.bo_vals = pvtable.cols[:, 1] if pvtable.ncols >= 2 else None
            self.mu_vals = pvtable.cols[:, 2] if pvtable.ncols >= 3 else None
        else:
            self.mode = 'p-indexed'
            self.pressures = pvtable.p
            self.bo_vals = pvtable.cols[:, 0] if pvtable.ncols >= 1 else None
            self.mu_vals = pvtable.cols[:, 1] if pvtable.ncols >= 2 else None
            # rs may be in cols as well
            self.rs = pvtable.cols[:, 2] if pvtable.ncols >= 3 else None

    def bo_of_p(self, P):
        if self.raw is None:
            return None
        P_arr = _np.atleast_1d(_np.asarray(P, dtype=float))
        if self.mode == 'rs-indexed':
            # interpolate rs->P to get rs(P) then bo(rs(P)) by inverse mapping
            # create function P(rs) and invert via interpolation over rs grid
            # Here we build P_of_rs and bo_of_rs, then invert rs_of_p by locating rs where P matches
            p_of_rs = _np.asarray(self.pressures)
            rs_vals = _np.asarray(self.rs)
            bo_of_rs = _np.asarray(self.bo_vals) if self.bo_vals is not None else None
            # For a given P, find rs by interpolation on p_of_rs vs rs_vals
            rs_at_P = _np.interp(P_arr, p_of_rs, rs_vals, left=rs_vals[0], right=rs_vals[-1])
            if bo_of_rs is None:
                return None
            return _np.interp(rs_at_P, rs_vals, bo_of_rs, left=bo_of_rs[0], right=bo_of_rs[-1])
        else:
            if self.bo_vals is None:
                return None
            return _np.interp(P_arr, self.pressures, self.bo_vals, left=self.bo_vals[0], right=self.bo_vals[-1])

    def mu_o_of_p(self, P):
        if self.raw is None:
            return None
        P_arr = _np.atleast_1d(_np.asarray(P, dtype=float))
        if self.mode == 'rs-indexed':
            p_of_rs = _np.asarray(self.pressures)
            rs_vals = _np.asarray(self.rs)
            mu_of_rs = _np.asarray(self.mu_vals) if self.mu_vals is not None else None
            rs_at_P = _np.interp(P_arr, p_of_rs, rs_vals, left=rs_vals[0], right=rs_vals[-1])
            if mu_of_rs is None:
                return None
            return _np.interp(rs_at_P, rs_vals, mu_of_rs, left=mu_of_rs[0], right=mu_of_rs[-1])
        else:
            if self.mu_vals is None:
                return None
            return _np.interp(P_arr, self.pressures, self.mu_vals, left=self.mu_vals[0], right=self.mu_vals[-1])

    def rs_of_p(self, P):
        if self.raw is None:
            return None
        P_arr = _np.atleast_1d(_np.asarray(P, dtype=float))
        if self.mode == 'rs-indexed':
            p_of_rs = _np.asarray(self.pressures)
            rs_vals = _np.asarray(self.rs)
            # invert p_of_rs -> rs via interp
            return _np.interp(P_arr, p_of_rs, rs_vals, left=rs_vals[0], right=rs_vals[-1])
        else:
            if self.rs is None:
                return None
            return _np.interp(P_arr, self.pressures, self.rs, left=self.rs[0], right=self.rs[-1])


def _as_float_vector(x):
    if x is None:
        return None
    try:
        arr = _np.asarray(x, dtype=float).ravel()
        return arr
    except Exception:
        return None


def _interp_linear_extrap(x, y, xi):
    """MRST ``interpTable`` equivalent: linear interpolation *and* extrapolation."""
    x = _np.asarray(x, dtype=float).ravel()
    y = _np.asarray(y, dtype=float).ravel()
    xi = _np.asarray(xi, dtype=float)
    if x.size == 0:
        return _np.zeros_like(xi, dtype=float)
    if x.size == 1:
        return _np.full_like(xi, y[0], dtype=float)
    order = _np.argsort(x)
    x, y = x[order], y[order]
    out = _np.interp(xi, x, y)
    below = xi < x[0]
    above = xi > x[-1]
    if _np.any(below):
        out[below] = y[0] + (xi[below] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if _np.any(above):
        out[above] = y[-1] + (xi[above] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


def _parse_pvdg(raw):
    """Port of ``assignPVDG.m``'s table extraction, including its extension.

    ``assignPVDG`` prepends a row at p = 0 carrying the first row's values::

        pG  = [0; pvdg(:, 1)];
        bg  = bg([1, 1:end]);
        mug = mug([1, 1:end]);

    so b_G and mu_G are *constant* below the table's first pressure rather
    than linearly extrapolated.  That matters: b_G is steep at the low end,
    and extrapolating it down the first segment's slope sends it through
    zero a little below the first tabulated pressure -- QIEDIE's PVDG starts
    at 10.1325 bar with b_G = 8.6 and a slope of 0.85 per bar, so the
    extrapolation reaches zero at p = 0 and goes negative below it.  A
    negative gas shrinkage factor is not a slow solve, it is a meaningless
    one.  (``assignPVDO`` deliberately has no such extension; only the gas
    branch carries it.)
    """
    arr = _as_float_vector(raw)
    if arr is None or arr.size < 3:
        return None
    n = arr.size // 3
    if n <= 0:
        return None
    mat = arr[:3 * n].reshape(n, 3)
    p, bg, mu = mat[:, 0], mat[:, 1], mat[:, 2]
    if p[0] > 0.0:
        p = _np.r_[0.0, p]
        bg = _np.r_[bg[0], bg]
        mu = _np.r_[mu[0], mu]
    return {
        'p': p,
        'bg': bg,
        'mu': mu,
    }


def _parse_pvdo(raw):
    """Port of ``assignPVDO.m``'s table extraction.

    Same three columns as PVDG -- pressure, FVF, viscosity -- but note
    that ``assignPVDO`` does *not* extend the table towards zero pressure
    the way ``assignPVDG`` does; only the gas branch carries that.
    """
    arr = _as_float_vector(raw)
    if arr is None or arr.size < 3:
        return None
    n = arr.size // 3
    if n <= 0:
        return None
    mat = arr[:3 * n].reshape(n, 3)
    return {
        'p': mat[:, 0],
        'bo': mat[:, 1],
        'mu': mat[:, 2],
    }


def _parse_pvtw(raw):
    arr = _as_float_vector(raw)
    if arr is None or arr.size < 5:
        return None
    # Eclipse PVTW single-region layout: p_ref, bw_ref, cw, muw_ref, cmu.
    return {
        'p_ref': float(arr[0]),
        'bw_ref': float(arr[1]),
        'cw': float(arr[2]),
        'mu_ref': float(arr[3]),
        'cmu': float(arr[4]),
    }


def _parse_pvto_records(raw):
    # ``readMisciblePVTTable``-compatible representation retained by the
    # deck parser: one list of slash-terminated records per PVT region.
    # The evaluator currently selects region one (as do all bundled cases
    # using this path); retaining this branch prevents unit conversion from
    # guessing PVTO record boundaries in a flattened vector.
    if (isinstance(raw, list) and raw and isinstance(raw[0], list) and
            raw[0] and isinstance(raw[0][0], _np.ndarray)):
        recs = []
        for row in raw[0]:
            row = _np.asarray(row, dtype=float).ravel()
            if row.size < 4 or (row.size - 1) % 3:
                raise ValueError('PVTO miscible record has invalid column count')
            data = row[1:].reshape((-1, 3))
            recs.append({
                'rs': float(row[0]),
                'pb': float(data[0, 0]),
                'bo_sat': float(data[0, 1]),
                'mu_sat': float(data[0, 2]),
                'p_u': _np.asarray(data[:, 0], dtype=float),
                'bo_u': _np.asarray(data[:, 1], dtype=float),
                'mu_u': _np.asarray(data[:, 2], dtype=float),
            })
        return _complete_pvto_usat_records(recs)

    arr = _as_float_vector(raw)
    if arr is None or arr.size < 4:
        return []
    recs = []
    i = 0
    n = arr.size
    while i + 3 < n:
        rs = float(arr[i])
        pb = float(arr[i + 1])
        bo0 = float(arr[i + 2])
        mu0 = float(arr[i + 3])
        i += 4

        p_u = [pb]
        bo_u = [bo0]
        mu_u = [mu0]

        # MRST's miscible table structure distinguishes a new Rs key from
        # an undersaturated (p, B, mu) continuation. In converted FIELD
        # tables, a new key has bubble-point pressure (second value) above
        # its Rs key, while a continuation has B as its second value.
        while i + 2 < n and not (float(arr[i + 1]) > float(arr[i])):
            p_u.append(float(arr[i]))
            bo_u.append(float(arr[i + 1]))
            mu_u.append(float(arr[i + 2]))
            i += 3

        recs.append({
            'rs': rs,
            'pb': pb,
            'bo_sat': bo0,
            'mu_sat': mu0,
            'p_u': _np.asarray(p_u, dtype=float),
            'bo_u': _np.asarray(bo_u, dtype=float),
            'mu_u': _np.asarray(mu_u, dtype=float),
        })
    return _complete_pvto_usat_records(recs)


def _parse_pvtg_records(raw):
    """Build the PVTG ``key/pos/data`` data used by ``assignPVTG.m``."""
    if not (isinstance(raw, list) and raw and isinstance(raw[0], list) and
            raw[0] and isinstance(raw[0][0], _np.ndarray)):
        return []
    records = []
    for row in raw[0]:
        row = _np.asarray(row, dtype=float).ravel()
        if row.size < 4 or (row.size - 1) % 3:
            raise ValueError('PVTG miscible record has invalid column count')
        data = row[1:].reshape((-1, 3))
        records.append({
            'p': float(row[0]),
            'rv': _np.asarray(data[:, 0], dtype=float),
            # assignPVTG transforms Eclipse B to shrinkage b before
            # preprocessing and interpolation.
            'b': 1.0 / _np.asarray(data[:, 1], dtype=float),
            'mu': _np.asarray(data[:, 2], dtype=float),
        })
    return records


def _complete_pvto_usat_records(recs):
    """Port ``fill_usat_invlinear.m`` for compressed Eclipse PVTO data.

    MRST's deck reader permits a PVTO Rs record to contain only its
    saturated row.  Before the fluid functions are built,
    ``readMisciblePVTTable`` invokes ``fill_usat_invlinear``: missing
    undersaturated rows are copied from the next available record while
    preserving the source B and viscosity ratios.  This is observable in
    SPE9, whose first ten records are one-row records and whose final
    record provides the two-point undersaturated curve.

    ``fill_usat_invlinear`` is written for an arbitrary number of
    segments.  Its sparse recurrence reduces exactly to the cumulative
    B/mu ratios below, which keeps both compressibility and viscosibility
    of the source table at each copied pressure interval.
    """
    if not recs:
        return recs
    missing = [i for i, rec in enumerate(recs) if rec['p_u'].size == 1]
    sources = [i for i, rec in enumerate(recs) if rec['p_u'].size > 1]
    if not missing:
        return recs
    if not sources:
        raise ValueError('PVTO final record must provide undersaturated data')

    for i in missing:
        # ``src_ix = 1 + sum(dst > src)`` in fill_usat_invlinear.m picks
        # the next source record, falling back to the nearest available
        # source when the table contains a leading/trailing gap.
        later = [j for j in sources if j > i]
        src = later[0] if later else sources[-1]
        ref = recs[src]
        p0 = float(recs[i]['p_u'][0])
        ref_p = ref['p_u']
        ref_bo = ref['bo_u']
        ref_mu = ref['mu_u']
        recs[i]['p_u'] = p0 + (ref_p - ref_p[0])
        recs[i]['bo_u'] = float(recs[i]['bo_u'][0]) * ref_bo / ref_bo[0]
        recs[i]['mu_u'] = float(recs[i]['mu_u'][0]) * ref_mu / ref_mu[0]
    return recs


class DeckBlackOilPVT:
    """First-pass black-oil PVT evaluator for deck-driven simulations.

    This evaluator is intentionally limited to the subset needed by the
    bundled black-oil decks:
    - PVTO for oil Bo/mu and dissolved gas Rs
    - PVDO for dead-oil Bo/mu
    - PVCDO for constant-compressibility oil Bo/mu
    - PVDG/PVTG for gas Bg/mu
    - PVTW for water Bw/mu
    - RV defaults to zero when no volatile-oil table is present.
    """

    def __init__(self, props):
        self.props = props or {}
        miscible = self.props.get('_miscible_pvt_records', {})
        self.pvto = _parse_pvto_records(miscible.get('PVTO', self.props.get('PVTO')))
        self.pvtg = _parse_pvtg_records(miscible.get('PVTG'))
        self.pvtw = _parse_pvtw(self.props.get('PVTW'))
        self.pvdg = _parse_pvdg(self.props.get('PVDG'))
        self.pvdo = _parse_pvdo(self.props.get('PVDO'))
        pvcdo = _np.asarray(self.props.get('PVCDO', []), dtype=float).ravel()
        # MRST assignPVCDO.m reads one five-item record per PVT region:
        # reference pressure, reference FVF, compressibility, reference
        # viscosity and viscosibility.  The bundled EGG deck has one
        # region; retain the first record until regional PVT dispatch is
        # introduced.
        self.pvcdo = pvcdo[:5] if pvcdo.size >= 5 else None

        self._pb = _np.asarray([r['pb'] for r in self.pvto], dtype=float) if self.pvto else _np.asarray([], dtype=float)
        self._rs = _np.asarray([r['rs'] for r in self.pvto], dtype=float) if self.pvto else _np.asarray([], dtype=float)
        self._bo_sat = _np.asarray([r['bo_sat'] for r in self.pvto], dtype=float) if self.pvto else _np.asarray([], dtype=float)
        self._mu_sat = _np.asarray([r['mu_sat'] for r in self.pvto], dtype=float) if self.pvto else _np.asarray([], dtype=float)
        self._pvtg_p = _np.asarray([r['p'] for r in self.pvtg], dtype=float) if self.pvtg else _np.asarray([], dtype=float)
        self._pvtg_rv_sat = (_np.asarray([r['rv'][0] for r in self.pvtg], dtype=float)
                             if self.pvtg else _np.asarray([], dtype=float))

    def _rs_of_p(self, p):
        if self._pb.size == 0:
            return _np.zeros_like(p)
        order = _np.argsort(self._pb)
        pb = self._pb[order]
        rs = self._rs[order]
        # assignPVTO.m prepends the physical point (0, 0) before calling
        # MRST interpTable, rather than clamping below the first bubble point.
        return _interp_linear_extrap(_np.r_[0.0, pb], _np.r_[0.0, rs], p)

    def _sat_oil_props(self, p):
        if self._pb.size == 0:
            one = _np.ones_like(p)
            return one, one
        order = _np.argsort(self._pb)
        pb = self._pb[order]
        b = 1.0 / self._bo_sat[order]
        muob = b / self._mu_sat[order]
        # assignPVTO interpolates b and muob=1/(B*mu) independently,
        # then returns mu=b/muob.  Interpolating mu directly is not
        # equivalent between table points.
        b_p = _interp_linear_extrap(pb, b, p)
        muob_p = _interp_linear_extrap(pb, muob, p)
        return b_p, b_p / muob_p

    def _undersat_oil_props(self, p, rs):
        """MRST ``interp2DPVT_parallel`` for PVTO tables.

        ``initEclipseProblemAD`` selects ``pvtMethodOil = 'parallel'``.
        The interpolation is performed in pressure distance from the
        interpolated bubble-point curve, then linearly blended between the
        two neighbouring Rs tables.  This is a literal numerical port of
        ``interpPVT.m`` rather than a nearest-Rs approximation.
        """
        if not self.pvto:
            return self._sat_oil_props(p)

        p = _np.asarray(p, dtype=float).ravel()
        rs = _np.asarray(rs, dtype=float).ravel()
        if rs.size != p.size:
            rs = _np.resize(rs, p.size)
        keys = _np.asarray([r['rs'] for r in self.pvto], dtype=float)
        nkey = keys.size
        if nkey == 1:
            rec = self.pvto[0]
            dp = p - rec['pb']
            xx = rec['p_u'] - rec['p_u'][0]
            b_values = 1.0 / rec['bo_u']
            muob_values = b_values / rec['mu_u']
            b = _interp_linear_extrap(xx, b_values, dp)
            muob = _interp_linear_extrap(xx, muob_values, dp)
            return b, b / muob

        # getBins in interpPVT.m: retain the endpoint interval for values
        # outside the table, which gives MRST's linear extrapolation.
        lo = _np.searchsorted(keys, rs, side='right') - 1
        lo = _np.clip(lo, 0, nkey - 2)
        hi = lo + 1
        w = (rs - keys[lo]) / (keys[hi] - keys[lo])
        pb = _interp_linear_extrap(keys, self._pb, rs)
        dp = p - pb

        b_left = _np.empty_like(p)
        b_right = _np.empty_like(p)
        muob_left = _np.empty_like(p)
        muob_right = _np.empty_like(p)
        for key_index in range(nkey):
            # In MRST the table indexed ``tn`` is the right boundary for
            # bin ``tn`` and the left boundary for bin ``tn-1``.
            use_left = hi == key_index
            use_right = lo == key_index
            if not (_np.any(use_left) or _np.any(use_right)):
                continue
            rec = self.pvto[key_index]
            shifted_p = rec['p_u'] - rec['p_u'][0]
            b_values = 1.0 / rec['bo_u']
            muob_values = b_values / rec['mu_u']
            if _np.any(use_left):
                b_left[use_left] = _interp_linear_extrap(
                    shifted_p, b_values, dp[use_left]
                )
                muob_left[use_left] = _interp_linear_extrap(
                    shifted_p, muob_values, dp[use_left]
                )
            if _np.any(use_right):
                b_right[use_right] = _interp_linear_extrap(
                    shifted_p, b_values, dp[use_right]
                )
                muob_right[use_right] = _interp_linear_extrap(
                    shifted_p, muob_values, dp[use_right]
                )

        # interp2DPVT_parallel: f = f_l*w + f_r*(1-w).
        b = b_left * w + b_right * (1.0 - w)
        muob = muob_left * w + muob_right * (1.0 - w)
        return b, b / muob

    def rv_sat(self, p):
        """MRST ``assignPVTG`` saturated Rv curve, including (0, 0)."""
        p = _np.asarray(p, dtype=float).ravel()
        if self._pvtg_p.size == 0:
            return _np.zeros_like(p)
        return _interp_linear_extrap(
            _np.r_[0.0, self._pvtg_p], _np.r_[0.0, self._pvtg_rv_sat], p
        )

    def _pvtg_tables(self, field):
        """The PVTG records' Rv tables for one field, flattened once."""
        cache = getattr(self, '_pvtg_flat', None)
        if cache is None:
            cache = {}
            self._pvtg_flat = cache
        tables = cache.get(field)
        if tables is None:
            tables = _RaggedTables([(table['rv'], table[field])
                                    for table in self.pvtg])
            cache[field] = tables
        return tables

    def _undersaturated_tables(self):
        """The PVTO records' undersaturated branches, flattened once.

        Each record interpolates pressure *above its own bubble point* --
        ``p_u - p_u[0]`` -- against the shrinkage factor and viscosity, so
        the tables are built from the shifted pressures the loop used to
        rebuild on every call.
        """
        cached = getattr(self, '_undersat_flat', None)
        if cached is None:
            shifts = [rec['p_u'] - rec['p_u'][0] for rec in self.pvto]
            cached = (
                _RaggedTables([(shift, 1.0 / rec['bo_u'])
                               for shift, rec in zip(shifts, self.pvto)]),
                _RaggedTables([(shift, (1.0 / rec['bo_u']) / rec['mu_u'])
                               for shift, rec in zip(shifts, self.pvto)]),
            )
            self._undersat_flat = cached
        return cached

    def rv_sat_adi(self, p):
        """ADI form of :meth:`rv_sat`, used by ``RvMax.m``."""
        if not is_ad(p):
            raise TypeError('rv_sat_adi requires an AD pressure')
        if self._pvtg_p.size == 0:
            return type(p).constant(_np.zeros(p.val.size), p.nvar)
        return ad_interp_linear(
            _np.r_[0.0, self._pvtg_p], _np.r_[0.0, self._pvtg_rv_sat], p
        )

    def _pvtg_linshift(self, p, rv, field):
        """Numerical port of ``interp2DPVT_linshift`` for PVTG."""
        p = _np.asarray(p, dtype=float).ravel()
        rv = _np.asarray(rv, dtype=float).ravel()
        if rv.size != p.size:
            rv = _np.resize(rv, p.size)
        if self._pvtg_p.size < 2:
            values = self.pvtg[0][field] if self.pvtg else _np.ones_like(p)
            x = self.pvtg[0]['rv'] if self.pvtg else _np.zeros(1)
            return _interp_linear_extrap(x, values, rv)

        # getBins in interpPVT.m clamps the exterior bin but leaves the
        # interpolation weight unconstrained, i.e. linear extrapolation.
        left = _np.searchsorted(self._pvtg_p, p, side='right') - 1
        left = _np.clip(left, 0, self._pvtg_p.size - 2)
        right = left + 1
        w = ((p - self._pvtg_p[left]) /
             (self._pvtg_p[right] - self._pvtg_p[left]))
        dx = self._pvtg_rv_sat[right] - self._pvtg_rv_sat[left]
        x_left = rv - w * dx
        x_right = rv + (1.0 - w) * dx

        f_left = _np.empty_like(p)
        f_right = _np.empty_like(p)
        for table_index, table in enumerate(self.pvtg):
            use_left = left == table_index
            use_right = right == table_index
            if _np.any(use_left):
                f_left[use_left] = _interp_linear_extrap(
                    table['rv'], table[field], x_left[use_left]
                )
            if _np.any(use_right):
                f_right[use_right] = _interp_linear_extrap(
                    table['rv'], table[field], x_right[use_right]
                )
        return f_left * (1.0 - w) + f_right * w

    def gas_props(self, p, rv, saturated=False):
        """MRST PVTG gas shrinkage/viscosity for a specified Rv status."""
        p = _np.asarray(p, dtype=float).ravel()
        rv = _np.asarray(rv, dtype=float).ravel()
        if rv.size != p.size:
            rv = _np.resize(rv, p.size)
        if not self.pvtg:
            if self.pvdg is None:
                return _np.ones_like(p), _np.ones_like(p)
            b_values = 1.0 / self.pvdg['bg']
            b = _interp_linear_extrap(self.pvdg['p'], b_values, p)
            mugbg = _interp_linear_extrap(
                self.pvdg['p'], b_values / self.pvdg['mu'], p)
            return b, b / mugbg
        sat = _np.asarray(saturated, dtype=bool)
        if sat.size == 1:
            sat = _np.full(p.size, bool(sat.ravel()[0]), dtype=bool)
        elif sat.size != p.size:
            sat = _np.resize(sat, p.size)
        # ``assignPVTG`` calls ``interpPVT(bg, rv, pg, flag, ...)``.  The
        # saturated branch of ``interpPVT`` is consequently parametrized
        # by the saturated *Rv* coordinates (``T.sat.x``), not pressure.
        # It also falls back to the undersaturated surface when Rv is below
        # the first saturated point.  Treating this as a p->b curve matched
        # gas-present cells by coincidence but was materially wrong in
        # Norne's oil-only cells.
        sat = sat & (rv >= self._pvtg_rv_sat[0])
        b_sat = _interp_linear_extrap(self._pvtg_rv_sat, self._np_pvtg_field('b'), rv)
        mu_sat = _interp_linear_extrap(self._pvtg_rv_sat, self._np_pvtg_field('mu'), rv)
        b = self._pvtg_linshift(p, rv, 'b')
        mu = self._pvtg_linshift(p, rv, 'mu')
        return _np.where(sat, b_sat, b), _np.where(sat, mu_sat, mu)

    def _np_pvtg_field(self, field):
        return _np.asarray([table[field][0] for table in self.pvtg], dtype=float)

    def eval(self, pressure, rs_override=None, rv_override=None,
             saturated_override=None, oil_saturated_override=None,
             gas_saturated_override=None):
        p = _np.atleast_1d(_np.asarray(pressure, dtype=float))

        rs_sat = self._rs_of_p(p)
        if rs_override is None:
            rs = rs_sat
        else:
            rs = _np.asarray(rs_override, dtype=float).ravel()
            if rs.size != p.size:
                rs = _np.resize(rs, p.size)
        rv = (_np.zeros_like(p) if rv_override is None else
              _np.asarray(rv_override, dtype=float).ravel())
        if rv.size != p.size:
            rv = _np.resize(rv, p.size)

        if self.pvto:
            bo_sat, muo_sat = self._sat_oil_props(p)
            bo, muo = self._undersat_oil_props(p, rs)
            oil_saturated = (oil_saturated_override if oil_saturated_override is not None
                             else saturated_override)
            if oil_saturated is not None:
                saturated = _np.asarray(oil_saturated, dtype=bool).ravel()
                if saturated.size != p.size:
                    saturated = _np.resize(saturated, p.size)
                # interpPVT.m explicitly prevents use of the saturated curve
                # below the first bubble-point pressure.
                saturated = saturated & (p >= float(_np.min(self._pb))) if self._pb.size else saturated
                bo = _np.where(saturated, bo_sat, bo)
                muo = _np.where(saturated, muo_sat, muo)
            bo = _np.where(_np.isfinite(bo), bo, bo_sat)
            muo = _np.where(_np.isfinite(muo), muo, muo_sat)
        elif self.pvdo is not None:
            # assignPVDO.m: bO = interp1d(p, 1./BO, po), muO = interp1d(p, muo, po).
            # Same B -> b inversion before interpolating that the PVDG branch
            # below does.  Falling through to the unit placeholder instead --
            # which is what a PVDO deck used to do -- gives every cell
            # bO = 1 and muO = 1 Pa*s, a thousand times the deck's oil
            # viscosity, and no dead-oil deck can converge on that.
            bo = _interp_linear_extrap(self.pvdo['p'], 1.0 / self.pvdo['bo'], p)
            muo = _interp_linear_extrap(self.pvdo['p'], self.pvdo['mu'], p)
        elif self.pvcdo is not None:
            # Literal port of ad-props/props/assignPVCDO.m:
            # bO = exp(co*(po-por))/bor, muO = muor*exp(vbo*(po-por)).
            por, bor, co, muor, vbo = self.pvcdo
            dp = p - por
            bo = _np.exp(co * dp) / bor
            muo = muor * _np.exp(vbo * dp)
        else:
            bo = _np.ones_like(p)
            muo = _np.ones_like(p)

        if self.pvtg:
            if gas_saturated_override is None:
                # Retain the legacy direct-evaluation fallback.  Deck model
                # calls provide the phase-status flag below, exactly as
                # PVTProps does in MRST.
                gas_sat = (_np.zeros_like(p, dtype=bool) if rv_override is None
                           else rv >= self.rv_sat(p))
            else:
                gas_sat = _np.asarray(gas_saturated_override, dtype=bool).ravel()
                if gas_sat.size != p.size:
                    gas_sat = _np.resize(gas_sat, p.size)
            bg, mug = self.gas_props(p, rv, gas_sat)
        elif self.pvdg is not None:
            # assignPVDG interpolates b and mugbg=1/(Bg*muG), then divides.
            b_values = 1.0 / self.pvdg['bg']
            bg = _interp_linear_extrap(self.pvdg['p'], b_values, p)
            mugbg = _interp_linear_extrap(
                self.pvdg['p'], b_values / self.pvdg['mu'], p)
            mug = bg / mugbg
        else:
            bg = _np.ones_like(p)
            mug = _np.ones_like(p)

        if self.pvtw is not None:
            dp = p - self.pvtw['p_ref']
            # Exact assignPVTW.m formulas.  The deck stores B, while MRST
            # uses b=(1+x+x^2/2)/B_ref in the equations.
            x = self.pvtw['cw'] * dp
            bw = (1.0 + x + 0.5 * x * x) / self.pvtw['bw_ref']
            y = -self.pvtw['cmu'] * dp
            muw = self.pvtw['mu_ref'] / (1.0 + y + 0.5 * y * y)
        else:
            bw = _np.ones_like(p)
            muw = _np.ones_like(p)

        return {
            'bw': _np.maximum(bw, 1e-30),
            'bo': _np.maximum(bo, 1e-30),
            'bg': _np.maximum(bg, 1e-30),
            'muw': muw,
            'muo': muo,
            'mug': mug,
            'rs': rs,
            'rv': rv,
        }

    def eval_adi(self, pressure, rs_override=None, rv_override=None,
                 saturated_override=None, oil_saturated_override=None,
                 gas_saturated_override=None):
        """Sparse-ADI counterpart of :meth:`eval`.

        Each interpolation applies the same piecewise-linear table slope as
        MRST's ADI ``interpTable`` operation.  Branch selectors (PVTO
        table bins and saturated-oil choice) are evaluated from values,
        exactly as MRST's state-function/upwind selectors do.
        """
        if not is_ad(pressure):
            raise TypeError('eval_adi requires an AD pressure')
        p = pressure
        # Every AD value built here follows the representation of the
        # pressure that was handed in, so the whole table evaluation stays
        # in whichever one the model's backend seeded.  Naming SparseADI
        # instead kept the diagonal backend out of the PVT chain -- the
        # longest elementwise stretch in the assembly, and the one it
        # exists to make cheap.
        AD = type(p)
        n = p.val.size
        zero = AD.constant(_np.zeros(n), p.nvar)
        one = AD.constant(_np.ones(n), p.nvar)

        if self._pb.size:
            order = _np.argsort(self._pb)
            pb = self._pb[order]
            rs_keys = self._rs[order]
            rs_sat = ad_interp_linear(_np.r_[0.0, pb], _np.r_[0.0, rs_keys], p)
        else:
            rs_sat = zero
        rs = rs_sat if rs_override is None else rs_override
        if not is_ad(rs):
            rs = AD.constant(rs, p.nvar)._broadcast(n)

        rv = zero if rv_override is None else rv_override
        if not is_ad(rv):
            rv = AD.constant(rv, p.nvar)._broadcast(n)

        if self.pvto:
            order = _np.argsort(self._pb)
            pb = self._pb[order]
            keys = self._rs[order]
            bo_sat = ad_interp_linear(pb, 1.0 / self._bo_sat[order], p)
            muob_sat = ad_interp_linear(
                pb, (1.0 / self._bo_sat[order]) / self._mu_sat[order], p)
            muo_sat = bo_sat / muob_sat
            if len(self.pvto) == 1:
                rec = self.pvto[0]
                dp = p - rec['pb']
                shift = rec['p_u'] - rec['p_u'][0]
                bo = ad_interp_linear(shift, 1.0 / rec['bo_u'], dp)
                muob = ad_interp_linear(
                    shift, (1.0 / rec['bo_u']) / rec['mu_u'], dp)
                muo = bo / muob
            else:
                lo = _np.searchsorted(keys, rs.val, side='right') - 1
                lo = _np.clip(lo, 0, len(self.pvto) - 2)
                hi = lo + 1
                w = (rs - keys[lo]) / (keys[hi] - keys[lo])
                pb_rs = ad_interp_linear(keys, pb, rs)
                dp = p - pb_rs
                # Every cell interpolates in two of the undersaturated
                # records -- the one below its Rs and the one above -- and
                # blends them by ``w``.  Done record by record that is a
                # scatter and a grid-sized addition per record per property;
                # done through the flattened tables it is four expressions,
                # whatever the deck's record count.
                b_table, muob_table = self._undersaturated_tables()
                b_left = _interp_ragged(b_table, hi, dp)
                muob_left = _interp_ragged(muob_table, hi, dp)
                b_right = _interp_ragged(b_table, lo, dp)
                muob_right = _interp_ragged(muob_table, lo, dp)
                bo = b_left * w + b_right * (1.0 - w)
                muob = muob_left * w + muob_right * (1.0 - w)
                muo = bo / muob
            oil_saturated = (oil_saturated_override if oil_saturated_override is not None
                             else saturated_override)
            if oil_saturated is not None:
                saturated = _np.asarray(oil_saturated, dtype=bool).reshape(-1)
                saturated = saturated & (p.val >= float(_np.min(self._pb)))
                bo = ad_select(saturated, bo_sat, bo)
                muo = ad_select(saturated, muo_sat, muo)
        elif self.pvdo is not None:
            bo = ad_interp_linear(self.pvdo['p'], 1.0 / self.pvdo['bo'], p)
            muo = ad_interp_linear(self.pvdo['p'], self.pvdo['mu'], p)
        elif self.pvcdo is not None:
            por, bor, co, muor, vbo = self.pvcdo
            dp = p - por
            bo = (dp * co).exp() / bor
            muo = (dp * vbo).exp() * muor
        else:
            bo = one
            muo = one

        if self.pvtg:
            # Direct port of assignPVTG.m -> interpPVT.m's ``linshift``
            # branch.  The PVTG tables interpolate in (Rv, p), and its
            # saturated branch is parametrized by Rv (the first data point
            # in each pressure table).  Bin choices are
            # values-only, as in MRST's ADI interpolation; both p and Rv
            # remain AD variables inside the selected linear piece.
            p_bins = _np.searchsorted(self._pvtg_p, p.val, side='right') - 1
            p_bins = _np.clip(p_bins, 0, len(self.pvtg) - 2)
            p_right = p_bins + 1
            dp = self._pvtg_p[p_right] - self._pvtg_p[p_bins]
            w = (p - self._pvtg_p[p_bins]) / dp
            drv = self._pvtg_rv_sat[p_right] - self._pvtg_rv_sat[p_bins]
            rv_left = rv - w * drv
            rv_right = rv + (1.0 - w) * drv

            def linshift(field):
                # Each cell interpolates the pressure table below it and the
                # one above, in the shifted Rv each of those implies, then
                # blends.  The per-table loop this replaces built two
                # grid-sized AD values for every table in the deck -- forty-
                # one of them on Norne -- to fill in the handful of rows that
                # chose it.
                tables = self._pvtg_tables(field)
                left_value = _interp_ragged(tables, p_bins, rv_left)
                right_value = _interp_ragged(tables, p_right, rv_right)
                return left_value * (1.0 - w) + right_value * w

            bg_usat, mug_usat = linshift('b'), linshift('mu')
            bg_sat = ad_interp_linear(self._pvtg_rv_sat, self._np_pvtg_field('b'), rv)
            mug_sat = ad_interp_linear(self._pvtg_rv_sat, self._np_pvtg_field('mu'), rv)
            if gas_saturated_override is None:
                gas_saturated = rv.val >= self.rv_sat(p.val)
            else:
                gas_saturated = _np.asarray(gas_saturated_override, dtype=bool).reshape(-1)
                if gas_saturated.size != n:
                    gas_saturated = _np.resize(gas_saturated, n)
            # interpPVT.m routes saturated requests below the first Rv point
            # to the undersaturated branch rather than extrapolating sat data.
            gas_saturated = gas_saturated & (rv.val >= self._pvtg_rv_sat[0])
            bg = ad_select(gas_saturated, bg_sat, bg_usat)
            mug = ad_select(gas_saturated, mug_sat, mug_usat)
        elif self.pvdg is not None:
            b_values = 1.0 / self.pvdg['bg']
            bg = ad_interp_linear(self.pvdg['p'], b_values, p)
            mugbg = ad_interp_linear(
                self.pvdg['p'], b_values / self.pvdg['mu'], p)
            mug = bg / mugbg
        else:
            bg = one
            mug = one

        if self.pvtw is not None:
            dp = p - self.pvtw['p_ref']
            x = dp * self.pvtw['cw']
            bw = (one + x + 0.5 * x * x) / self.pvtw['bw_ref']
            y = dp * (-self.pvtw['cmu'])
            muw = self.pvtw['mu_ref'] / (one + y + 0.5 * y * y)
        else:
            bw = one
            muw = one

        return {
            'bw': ad_maximum(bw, 1.0e-30),
            'bo': ad_maximum(bo, 1.0e-30),
            'bg': ad_maximum(bg, 1.0e-30),
            'muw': muw, 'muo': muo, 'mug': mug,
            'rs': rs, 'rv': rv,
        }
