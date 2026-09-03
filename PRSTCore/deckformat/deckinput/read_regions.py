"""Read the ECLIPSE ``REGIONS`` section.

This is the array part of MRST's ``readREGIONS.m``.  Region arrays use the
same GRID-box grammar as GRID properties, so the implementation deliberately
reuses the corresponding token expansion and operator application routines
from :mod:`read_grid`.
"""

from __future__ import annotations

import numpy as np

from .read_grid import _apply_operators, _flatten_tokens


# ``readREGIONS.m`` accepts these grid-box arrays.  Keeping the list here
# explicit prevents a following, unknown keyword from silently being folded
# into the preceding region array.
_ARRAY_KEYWORDS = {
    'EQLNUM', 'FIPNUM', 'IMBNUM', 'PVTNUM', 'SATNUM', 'SURFNUM',
    'ENDNUM', 'ROCKNUM', 'FIPFAC', 'EOSNUM',
}
_OPERATORS = {'ADD', 'COPY', 'EQUALS', 'MAXVALUE', 'MINVALUE', 'MULTIPLY'}
_SECTION_CONTROLS = {'REGIONS', 'SOLUTION', 'SCHEDULE', 'SUMMARY', 'END',
                     'ECHO', 'NOECHO'}


def read_regions(block, cart_dims=None):
    """Parse region arrays and their source-order GRID-box operations.

    ``readREGIONS`` invokes ``readGridBoxArray`` for each named array and
    ``applyOperator`` for EQUALS/ADD/... records.  The deck reader has already
    expanded INCLUDE files, therefore this parser only needs to preserve the
    slash-delimited records and run the same operations afterwards.
    """
    data = {}
    collected = {}
    operators = []
    current = None
    lines = [line.split('--', 1)[0].strip() for line in block.splitlines()]
    line_no = 0

    while line_no < len(lines):
        line = lines[line_no]
        line_no += 1
        if not line:
            continue
        parts = line.split()
        head = parts[0].upper()

        if head in _SECTION_CONTROLS:
            current = None
            continue

        if head in _OPERATORS:
            records = []
            remainder = parts[1:]
            if remainder:
                records.append(remainder[:remainder.index('/') + 1]
                               if '/' in remainder else remainder)
            # A slash-only row terminates the operator keyword.  This is
            # the record loop in MRST applyOperator/readREGIONS.
            while line_no < len(lines):
                record_line = lines[line_no]
                line_no += 1
                if not record_line:
                    continue
                record = record_line.split()
                if record == ['/']:
                    break
                records.append(record[:record.index('/') + 1]
                               if '/' in record else record)
            operators.append((head, records))
            current = None
            continue

        if head in _ARRAY_KEYWORDS:
            current = head
            rest = parts[1:]
            collected.setdefault(current, []).extend(rest)
            if '/' in rest:
                current = None
            continue

        if parts == ['/']:
            current = None
            continue
        if current is not None:
            collected.setdefault(current, []).extend(parts)

    for keyword, tokens in collected.items():
        values = _flatten_tokens(tokens)
        if not values:
            continue
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError('REGIONS %s contains a non-numeric array value' % keyword)
        data[keyword] = np.asarray(values, dtype=float)

    return _apply_operators(data, operators, cart_dims)
