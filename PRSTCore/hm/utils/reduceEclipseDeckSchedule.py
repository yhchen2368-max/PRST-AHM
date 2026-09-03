"""Port of MRST ``reduceEclipseDeckSchedule.m`` (mrst-2026a/hm/utils).

Writes ``deck['reducedSCHEDULE']``: a copy of the schedule in which the
records of WELSPECS/COMPDAT/GINJGAS/GRUPTREE that simply repeat the
previous control are removed, so each control carries only what actually
changed. The original SCHEDULE is left untouched.

MATLAB routes the comparison through ``cell2table``/``setdiff(..., 'rows',
'stable')``; the port compares the records directly as tuples, which is the
same set difference with the same ordering.
"""

import copy as _copy

import numpy as _np

FIELDS = ('WELSPECS', 'COMPDAT', 'GINJGAS', 'GRUPTREE')


def reduceEclipseDeckSchedule(deck):
    """Attach ``deck['reducedSCHEDULE']``."""
    schedule = deck.get('SCHEDULE')
    if not schedule:
        return deck
    reduced = _copy.deepcopy(schedule)
    controls = reduced.get('control', [])
    for field in FIELDS:
        _reduceControlField(controls, field)
    reduced['control'] = controls
    deck['reducedSCHEDULE'] = reduced
    return deck


def _reduceControlField(controls, field):
    """Port of ``reduceControlField`` for one keyword."""
    first = next((i for i, c in enumerate(controls) if c.get(field)), None)
    if first is None:
        return controls

    for c in controls[first:]:
        if c.get(field):
            c[field] = _replace_nan(c[field])

    last = [tuple(row) for row in controls[first][field]]
    for c in controls[first + 1:]:
        records = c.get(field)
        if not records:
            continue
        # setdiff(..., 'rows', 'stable'): drop rows already in `last`,
        # keeping the surviving rows in their original order.
        seen = set(last)
        new = [row for row in records if tuple(row) not in seen]
        c[field] = new
        if new:
            last = last + [tuple(row) for row in new]
    return controls


def _replace_nan(records):
    """Port of ``replace_nan``: a NaN entry becomes ``-1``.

    ECLIPSE defaults arrive as NaN; the comparison above needs a value that
    compares equal to itself, which NaN does not.
    """
    out = []
    for row in records:
        new_row = []
        for value in row:
            if isinstance(value, float) and _np.isnan(value):
                new_row.append(-1)
            elif isinstance(value, (int, float, _np.floating, _np.integer)) \
                    and _np.isnan(_np.asarray(value, dtype=float)):
                new_row.append(-1)
            else:
                new_row.append(value)
        out.append(new_row)
    return out
