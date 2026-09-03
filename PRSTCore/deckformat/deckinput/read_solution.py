"""Read the ECLIPSE ``SOLUTION`` section.

The representation deliberately mirrors the part of MRST's ``readSOLUTION``
that is consumed by ``initStateDeck``: per-cell vectors are expanded from
ECLIPSE repeat syntax and tabular keywords retain one row per terminating
slash.  In particular, this must not leave the section as raw text: MRST
uses ``PRESSURE``, ``SWAT``, ``SGAS`` and ``RS`` directly when constructing a
deck initial state.
"""

from __future__ import annotations

import re

import numpy as np


_KEYWORD = re.compile(r"[A-Z][A-Z0-9_]*$")
_REPEAT = re.compile(
    r"([+-]?\d+)\*([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[EeDd][+-]?\d+)?)$"
)


def _parse_token(token: str):
    """Expand one ECLIPSE numeric/repeat token.

    A bare ``n*`` is ECLIPSE's default marker.  Keeping it as ``nan`` lets
    callers retain positional columns without pretending that it is zero.
    """
    token = token.strip().rstrip(",;")
    if not token or token == "/":
        return []
    repeated = _REPEAT.fullmatch(token)
    if repeated:
        return [float(repeated.group(2).replace("D", "E").replace("d", "e"))] * int(repeated.group(1))
    if token.endswith("*") and token[:-1].lstrip("+-").isdigit():
        return [np.nan] * int(token[:-1])
    try:
        return [float(token.replace("D", "E").replace("d", "e"))]
    except ValueError:
        return [token.strip("'\"")]


def _finish_record(records, record):
    if record:
        records.append(record)


def read_solution(block):
    """Parse a SOLUTION block into MRST-like keyword values.

    Vector-valued fields are returned as one-dimensional arrays.  Keywords
    with multiple slash-delimited records (e.g. EQUIL/RSVD) are returned as
    a two-dimensional array when rows are rectangular, otherwise as a list
    of arrays.
    """
    records_by_keyword = {}
    tables_by_keyword = {}
    current = None
    record = []
    table_values = []
    # readSOLUTION.m calls readRecordString once for each EQL region.  A
    # slash therefore terminates a *whole* two-column depth table, not one
    # row.  MRST stores the result as a cell array indexed by region.
    table_keywords = {'PBVD', 'PDVD', 'RSVD', 'RVVD', 'RTEMPVD'}

    def flush_current():
        nonlocal record, table_values
        if current is None:
            return
        if current in table_keywords:
            if table_values:
                tables_by_keyword.setdefault(current, []).append(table_values)
            table_values = []
        else:
            _finish_record(records_by_keyword.setdefault(current, []), record)
            record = []

    for raw_line in block.splitlines():
        line = raw_line.split('--', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        head = parts[0].upper()

        # The section header itself is part of the block supplied by
        # read_eclipse_deck and is not a SOLUTION keyword.
        if head == "SOLUTION":
            continue

        if _KEYWORD.fullmatch(head):
            flush_current()
            current = head
            parts = parts[1:]

        if current is None:
            continue

        for part in parts:
            if part == "/":
                if current in table_keywords:
                    # Ignore trailing explanatory text after the slash.  It
                    # is legal ECLIPSE syntax and is used on Norne's EQUIL
                    # cards.
                    if table_values:
                        tables_by_keyword.setdefault(current, []).append(table_values)
                    table_values = []
                else:
                    _finish_record(records_by_keyword.setdefault(current, []), record)
                    record = []
                break
            values = _parse_token(part)
            if current in table_keywords:
                table_values.extend(values)
            else:
                record.extend(values)

    flush_current()

    out = {}
    vector_keywords = {
        "PBUB", "PRESSURE", "RS", "RV", "SGAS", "SOIL", "SWAT",
        "TEMPI", "DATUM", "RPTSOL", "RPTRST", "OUTSOL",
    }
    for keyword, tables in tables_by_keyword.items():
        parsed = []
        for values in tables:
            if (len(values) % 2 != 0 or
                    not all(isinstance(value, (float, int, np.floating)) for value in values)):
                raise ValueError('%s must contain numeric two-column depth tables' % keyword)
            parsed.append(np.asarray(values, dtype=float).reshape((-1, 2)))
        # Preserve the existing single-region representation while exposing
        # MRST's cell-array behavior when EQUIL defines multiple regions.
        out[keyword] = parsed[0] if len(parsed) == 1 else parsed

    for keyword, rows in records_by_keyword.items():
        if keyword in vector_keywords:
            values = [value for row in rows for value in row]
            if values and all(isinstance(value, (float, int, np.floating)) for value in values):
                out[keyword] = np.asarray(values, dtype=float)
            else:
                out[keyword] = values
            continue

        if not rows:
            out[keyword] = np.empty((0,), dtype=float)
            continue
        if all(all(isinstance(value, (float, int, np.floating)) for value in row) for row in rows):
            widths = {len(row) for row in rows}
            if len(widths) == 1:
                out[keyword] = np.asarray(rows, dtype=float)
            else:
                out[keyword] = [np.asarray(row, dtype=float) for row in rows]
        else:
            out[keyword] = rows
    return out
