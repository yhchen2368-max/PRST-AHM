"""Port of MRST ``updateDeckSchedule.m`` (mrst-2026a/hm/utils/evaluate).

Writes an MRST schedule's wells back into the deck's SCHEDULE keywords,
so a calibrated case can be exported as a runnable ECLIPSE deck.

Each well's perforation list becomes its COMPDAT block (one record per
cell, with the logical indices, open/shut flag, well index and radius),
and its control becomes the matching column of WCONINJE or WCONPROD.

Production targets are written as ``val * sign``, i.e. negated back to
ECLIPSE's positive-rate convention; bhp and thp are written unsigned.
"""

import copy as _copy

import numpy as _np

# 0-based WCONINJE columns per control type.
_INJ_COLUMN = {'rate': 4, 'resv': 5, 'bhp': 6, 'thp': 7}
# 0-based WCONPROD columns per control type.
_PROD_COLUMN = {'orat': 3, 'wrat': 4, 'grat': 5, 'lrat': 6, 'resv': 7,
                'bhp': 8, 'thp': 9}
# Which of those are pressures, and so not multiplied by the well sign.
_PRESSURE_TYPES = ('bhp', 'thp')


def updateDeckSchedule(deck, G, schedule):
    """Return the deck SCHEDULE with its well keywords refreshed."""
    CTRL = _copy.deepcopy(deck['SCHEDULE']['control'])
    ctrl = schedule['control']
    assert len(CTRL) == len(ctrl), \
        'deck and schedule must have the same number of controls'

    for cno, control in enumerate(CTRL):
        names = [row[0] for row in control.get('WELSPECS', []) if row]
        welspecs, compdat, wconinje, wconprod = [], [], [], []

        for name in names:
            w = _find_well(ctrl[cno]['W'], name)
            if w is None:
                continue

            cells = _np.atleast_1d(_np.asarray(w['cells'], dtype=int)).ravel()
            I, J, K = _grid_logical_indices(G, cells)

            row = _row_for(control.get('WELSPECS'), name)
            if row is not None:
                row = list(row)
                row[2], row[3] = int(I[0]), int(J[0])
                welspecs.append(row)

            template = _row_for(control.get('COMPDAT'), name)
            if template is not None:
                cstatus = _np.atleast_1d(
                    _np.asarray(w.get('cstatus', _np.ones(cells.size)),
                                dtype=bool)).ravel()
                wi = _np.atleast_1d(_np.asarray(w['WI'], dtype=float)).ravel()
                r = _np.atleast_1d(_np.asarray(w['r'], dtype=float)).ravel()
                direction = _np.atleast_1d(_np.asarray(w['dir'])).ravel()
                for k in range(cells.size):
                    record = list(template)
                    record[1:5] = [int(I[k]), int(J[k]), int(K[k]), int(K[k])]
                    record[5] = 'OPEN' if _at(cstatus, k) else 'SHUT'
                    record[7] = float(_at(wi, k))
                    record[8] = float(_at(r, k))
                    record[9] = -1        # a defaulted Kh
                    record[12] = _at(direction, k)
                    compdat.append(record)

            wtype = str(w.get('type', '')).lower()
            val = float(w.get('val', 0.0))
            sign = float(w.get('sign', 0.0))
            status = bool(w.get('status', True))

            row = _row_for(control.get('WCONINJE'), name)
            if row is not None:
                row = list(row)
                row[2] = 'OPEN' if status else 'SHUT'
                row[3] = wtype.upper()
                column = _INJ_COLUMN.get(wtype)
                if column is not None:
                    row[column] = val if wtype in _PRESSURE_TYPES else val * sign
                wconinje.append(row)

            row = _row_for(control.get('WCONPROD'), name)
            if row is not None:
                row = list(row)
                row[1] = 'OPEN' if status else 'SHUT'
                row[2] = wtype.upper()
                column = _PROD_COLUMN.get(wtype)
                if column is not None:
                    row[column] = val if wtype in _PRESSURE_TYPES else val * sign
                wconprod.append(row)

        control['WELSPECS'] = welspecs
        control['COMPDAT'] = compdat
        control['WCONINJE'] = wconinje
        control['WCONPROD'] = wconprod

    SCHEDULE = dict(deck['SCHEDULE'])
    SCHEDULE['control'] = CTRL
    return SCHEDULE


def _find_well(W, name):
    lowered = str(name).lower()
    for w in W:
        if str(w.get('name', '')).lower() == lowered:
            return w
    return None


def _row_for(records, name):
    """The first record naming ``name``."""
    lowered = str(name).lower()
    for row in records or []:
        if row and str(row[0]).lower() == lowered:
            return row
    return None


def _at(values, k):
    return values[k] if k < values.size else values[-1]


def _grid_logical_indices(G, cells):
    """Port of ``gridLogicalIndices``: per-cell (I, J, K), one-based."""
    dims = _np.asarray(G['cartDims'], dtype=int).ravel()
    nx, ny = int(dims[0]), int(dims[1])
    index_map = G['cells'].get('indexMap')
    index_map = (_np.arange(int(G['cells']['num']), dtype=int)
                 if index_map is None
                 else _np.asarray(index_map, dtype=int).ravel())
    linear = index_map[cells]
    return (linear % nx + 1, (linear // nx) % ny + 1, linear // (nx * ny) + 1)
