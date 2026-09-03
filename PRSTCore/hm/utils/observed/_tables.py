"""Shared sheet reading and header mapping for the ``observed`` readers.

Every ``read*Test``/``readProductionHistory`` in MRST repeats the same
``solveKeySimilarities`` shape: a list of header synonyms per canonical
column, matched case-insensitively by substring against the sheet's
headers, with non-name/date columns coerced to float. Only the synonym
tables differ, so the mechanism lives here once and each reader supplies
its own table.
"""

import datetime as _dt

import numpy as _np

DATE_FORMATS = {6: '%Y%m', 8: '%Y%m%d', 10: '%Y-%m-%d'}


def solve_key_similarities(sheet, possible_keys, text_columns=('name', 'date')):
    """Map a sheet's headers onto canonical names.

    ``possible_keys`` is a sequence of ``(canonical, synonyms)`` pairs, in
    priority order. Columns not in ``text_columns`` are coerced to float,
    as MATLAB does with ``str2double``.
    """
    lowered = {str(k).lower(): k for k in sheet}
    out = {}
    for canonical, synonyms in possible_keys:
        match = None
        for header_l, header in lowered.items():
            if any(s.lower() in header_l for s in synonyms):
                match = header
                break
        if match is None:
            continue
        values = sheet[match]
        if canonical not in text_columns:
            values = to_float(values)
        out[canonical] = _np.asarray(values)
    return out


def parse_dates(values, forward_fill=False):
    """Parse a date column from the three fixed-width ECLIPSE-ish formats.

    ``forward_fill`` carries the previous date into a blank cell, which is
    how the profile sheets mark "same survey, next interval".
    """
    values = _np.asarray(values, dtype=object)
    if values.size and isinstance(values[0], (_dt.date, _dt.datetime)):
        return values

    text = []
    previous = None
    for v in values:
        item = '' if v is None else str(v).strip()
        if item in ('', 'nan', 'NaN', 'None'):
            if forward_fill and previous is not None:
                item = previous
            else:
                raise ValueError('Blank date with nothing to carry forward')
        # A numeric date (20200101.0) loses its integer form via float.
        if item.endswith('.0'):
            item = item[:-2]
        text.append(item)
        previous = item

    out = []
    for item in text:
        fmt = DATE_FORMATS.get(len(item))
        if fmt is None:
            raise ValueError('Unrecognised date format: %r' % item)
        out.append(_dt.datetime.strptime(item, fmt).date())
    return _np.asarray(out, dtype=object)


def split_depth_interval(values):
    """``'1200 - 1250'`` -> ``(top, bottom)`` arrays.

    Whitespace is stripped before splitting on the hyphen, matching the
    MATLAB's ``regexprep(x, '\\s+', '')`` then ``strsplit(x, '-')``.
    """
    top, bottom = [], []
    for item in _np.asarray(values, dtype=object):
        text = ''.join(str(item).split())
        parts = text.split('-')
        if len(parts) < 2:
            raise ValueError('Cannot read a depth interval from %r' % item)
        top.append(float(parts[0]))
        bottom.append(float(parts[1]))
    return _np.asarray(top), _np.asarray(bottom)


def read_sheets(fn):
    """Every sheet of ``fn`` as ``{header: column}``.

    CSV is read natively; .xls/.xlsx needs pandas.
    """
    name = str(fn)
    if name.lower().endswith('.csv'):
        return [read_csv(name)]
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            'Reading %s requires pandas (CSV files do not)' % name) from exc
    book = pd.read_excel(name, sheet_name=None)
    return [{c: frame[c].to_numpy() for c in frame.columns}
            for frame in book.values()]


def read_csv(name):
    import csv
    with open(name, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {}
    headers = [h.strip() for h in rows[0]]
    columns = {h: [] for h in headers}
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        for i, h in enumerate(headers):
            columns[h].append(row[i] if i < len(row) else '')
    return {h: _np.asarray(v, dtype=object) for h, v in columns.items()}


def to_float(values):
    out = _np.empty(len(values), dtype=float)
    for i, v in enumerate(values):
        try:
            out[i] = float(str(v).strip())
        except (TypeError, ValueError):
            out[i] = _np.nan
    return out


def group_by_well(table, name_column='name'):
    """``[(well, table), ...]`` preserving first-appearance order."""
    names = _np.asarray(table[name_column], dtype=object)
    out = []
    seen = set()
    for well in names:
        if well in seen:
            continue
        seen.add(well)
        ix = names == well
        out.append((well, {k: _np.asarray(v)[ix] for k, v in table.items()}))
    return out


def fill_missing_with(table, columns, value):
    """NaN entries of the named columns become ``value``."""
    for column in columns:
        if column in table:
            values = _np.asarray(table[column], dtype=float)
            values[_np.isnan(values)] = value
            table[column] = values
    return table
