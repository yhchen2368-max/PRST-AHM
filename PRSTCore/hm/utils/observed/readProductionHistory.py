"""Port of MRST ``readProductionHistory.m`` (mrst-2026a/hm/utils/observed).

Reads a production-history spreadsheet into one table per well.

Expected columns (header names may be Chinese or English -- see
:data:`POSSIBLE_KEYS`)::

    name  date  time  water  oil  gas  thp  chp  bhp

with units: water/oil rate m^3/day, gas rate 10^4 m^3/day, pressures MPa.

Conditioning applied, matching the MATLAB:

* the date column is parsed from ``yyyyMM``, ``yyyyMMdd`` or ``yyyy-MM-dd``
  when it is not already a date;
* leading rows with zero total rate (the well not yet on production) are
  dropped;
* missing rates become zero;
* missing or zero pressures become atmospheric (0.101325 MPa), since a
  zero reading means "not measured", not "no pressure".
"""

import numpy as _np

from ._tables import (group_by_well, parse_dates, read_sheets,
                      solve_key_similarities)

ATMOSPHERIC_MPA = 0.101325

# Header synonyms, in the MATLAB's order.
POSSIBLE_KEYS = (
    ('name', ('井号', '井名', 'wellname', 'name')),
    ('date', ('日期', '生产日期', '年月', 'date')),
    ('time', ('时间', '生产时间', '生产天数', 'time')),
    ('water', ('日产水量', '日产水', 'water')),
    ('oil', ('日产油量', '日产油', 'oil')),
    ('gas', ('日产气量', '日产气', 'gas')),
    ('thp', ('油压', 'thp')),
    ('chp', ('套压', 'chp')),
    ('bhp', ('流压', 'bhp')),
)

_RATE_COLUMNS = ('water', 'oil', 'gas')
_PRESSURE_COLUMNS = ('bhp', 'chp', 'thp')

def readProductionHistory(fn):
    """Return ``[(well_name, table), ...]``, one entry per well per sheet."""
    out = []
    for sheet in read_sheets(fn):
        table = solveKeySimilarities(sheet)
        if 'name' not in table:
            continue
        if _np.asarray(table['name']).size == 0:
            continue
        table['date'] = parse_dates(table['date'])
        table = _drop_leading_idle_rows(table)
        _fill_missing(table)
        out.extend(group_by_well(table))
    return out


def solveKeySimilarities(sheet):
    """Port of ``solveKeySimilarities``: map header synonyms to canonical names."""
    return solve_key_similarities(sheet, POSSIBLE_KEYS)


def _drop_leading_idle_rows(table):
    """Drop rows before the first with a nonzero total rate."""
    present = [c for c in _RATE_COLUMNS if c in table]
    if not present:
        return table
    total = _np.zeros(len(table[present[0]]))
    for column in present:
        # MATLAB performs this sum before replacing NaNs, so a NaN in one
        # phase makes that row's total NaN rather than treating it as zero.
        total = total + _np.asarray(table[column], dtype=float)
    nonzero = _np.flatnonzero(_np.abs(total) > 0)
    if nonzero.size == 0 or nonzero[0] == 0:
        return table
    start = int(nonzero[0])
    return {k: _np.asarray(v)[start:] for k, v in table.items()}


def _fill_missing(table):
    """Missing rates -> 0; missing or zero pressures -> atmospheric."""
    for column in _RATE_COLUMNS:
        if column in table:
            values = _np.asarray(table[column], dtype=float)
            values[_np.isnan(values)] = 0.0
            table[column] = values
    for column in _PRESSURE_COLUMNS:
        if column in table:
            values = _np.asarray(table[column], dtype=float)
            values[_np.isnan(values) | (values == 0.0)] = ATMOSPHERIC_MPA
            table[column] = values
    return table

