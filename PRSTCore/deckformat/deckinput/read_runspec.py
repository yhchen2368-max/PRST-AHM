"""Read RUNSPEC section from an ECLIPSE deck."""

import math
import re


# Local MRST readRUNSPEC.m readDefaultedRecord templates. A default is an
# occupied item, never a deletion: TABDIMS item 13 selects ROCKNUM tables.
_DIMENSION_DEFAULTS = {
    'EQLDIMS': [1, 100, 50, 1, 50],
    'TABDIMS': [1, 1, 20, 20, 1, 20, 20, 1, 1, math.nan, 10, 1, 1,
                0, 0, math.nan, 10, 10, 10, math.nan, 5, 5, 5, 0, math.nan],
    'WELLDIMS': [0, 0, 0, 0, 5, 10, 5, 4, 3, 0, 1, 1, 10, 201],
    'AQUDIMS': [1, 1, 1, 36, 1, 1, 0, 0],
    'REGDIMS': [1, 1, 0, 0, 0, 1, 0, 0, 0],
    'VFPIDIMS': [0, 0, 0],
    'VFPPDIMS': [0, 0, 0, 0, 0, 0],
}


def _record_tokens(lines, start, initial):
    """Read one slash-terminated ECLIPSE record, like MRST readRecordString."""
    tokens = []
    i = start - 1
    record_line = ' '.join(initial)
    while True:
        for token in re.findall(r"'[^']*'|/|[^\s/]+", record_line):
            if token == '/':
                return tokens, i
            tokens.append(token.strip("'"))
        i += 1
        if i >= len(lines):
            break
        record_line = lines[i].split('--', 1)[0]
    return tokens, i


def _integer_tokens(tokens, defaults=None):
    values = [] if defaults is None else list(defaults)
    position = 0
    for token in tokens:
        # readDefaultedRecord's replaceEmpty: quoted '' occupies one slot.
        token = token or '1*'
        count, value = 1, token
        if '*' in token:
            repeat, _, value = token.partition('*')
            count = int(repeat)
            if not value:
                if defaults is None:
                    raise ValueError('Defaulted integer item requires a keyword template')
                position += count
                continue
        number = float(value.replace('D', 'E').replace('d', 'e'))
        number = int(number) if math.isfinite(number) else number
        for _ in range(count):
            if defaults is None:
                values.append(number)
            else:
                if position >= len(values):
                    raise ValueError('Too many items in RUNSPEC dimension record')
                values[position] = number
            position += 1
    return values


def read_runspec(block):
    """Parse the MRST ``readRUNSPEC`` subset needed by deck simulations.

    ECLIPSE records normally follow a keyword on the next line.  The old
    line-by-line reader therefore lost DIMENS and all defaulted dimension
    records.  This follows the record boundary used by MRST's
    ``readRecordString`` for those keyword forms.
    """
    data = {}
    lines = [line.split('--', 1)[0] for line in block.split('\n')]
    i = 0
    flags = {"METRIC", "FIELD", "LAB", "PVT_M", "PVT-M", "SI",
             "OIL", "WATER", "GAS", "DISGAS", "VAPOIL", "BLACKOIL",
             "POLYMER", "SURFACT", "BRINE", "TEMP", "THERMAL", "MECH"}
    dimensions = _DIMENSION_DEFAULTS
    while i < len(lines):
        parts = lines[i].strip().split()
        if not parts:
            i += 1
            continue
        kw = parts[0].upper()
        if kw == "RUNSPEC":
            i += 1
            continue
        if kw in flags:
            data[kw.replace('-', '_')] = True
        elif kw == "DIMENS":
            record, i = _record_tokens(lines, i + 1, parts[1:])
            values = _integer_tokens(record)
            if len(values) >= 3:
                data["cartDims"] = values[:3]
                data["DIMENS"] = values[:3]
        elif kw in dimensions:
            record, i = _record_tokens(lines, i + 1, parts[1:])
            data[kw] = _integer_tokens(record, dimensions[kw])
            if kw == 'WELLDIMS' and not math.isfinite(data[kw][1]):
                data[kw][1] = data['cartDims'][2]
        elif kw == "ENDSCALE":
            # Keep the presence of ENDSCALE and its record.  MRST's
            # FlowPropertyFunctions enables endpoint scaling from this
            # keyword; NODIR/REVERS are consumed by the deck reader here
            # but do not alter the scalar saturation mapping itself.
            record, i = _record_tokens(lines, i + 1, parts[1:])
            data[kw] = record
        elif kw in ("TITLE", "START"):
            if len(parts) > 1:
                value = " ".join(parts[1:])
            elif i + 1 < len(lines):
                i += 1
                value = lines[i].strip()
            else:
                value = ""
            data[kw] = value.rstrip('/').strip().strip("'\"")
        i += 1
    return data
