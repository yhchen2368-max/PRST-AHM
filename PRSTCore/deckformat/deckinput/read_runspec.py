"""Read RUNSPEC section from an ECLIPSE deck."""


def _record_tokens(lines, start, initial):
    """Read one slash-terminated ECLIPSE record, like MRST readRecordString."""
    tokens = list(initial)
    i = start
    while i < len(lines):
        for token in lines[i].strip().split():
            if token == '/':
                return tokens, i
            if token.endswith('/'):
                tokens.append(token[:-1])
                return tokens, i
            tokens.append(token)
        i += 1
    return tokens, i


def _integer_tokens(tokens):
    values = []
    for token in tokens:
        # DIMENS and the dimensions records used here are integer records.
        # Preserve MRST's defaulted-item convention by simply omitting an
        # unspecified ``n*`` item; downstream code only needs explicit
        # cartesian dimensions.
        if '*' in token:
            count, _, value = token.partition('*')
            if not value:
                continue
            try:
                values.extend([int(float(value))] * int(count))
            except ValueError:
                continue
            continue
        try:
            values.append(int(float(token)))
        except ValueError:
            continue
    return values


def read_runspec(block):
    """Parse the MRST ``readRUNSPEC`` subset needed by deck simulations.

    ECLIPSE records normally follow a keyword on the next line.  The old
    line-by-line reader therefore lost DIMENS and all defaulted dimension
    records.  This follows the record boundary used by MRST's
    ``readRecordString`` for those keyword forms.
    """
    data = {}
    lines = block.split("\n")
    i = 0
    flags = {"METRIC", "FIELD", "LAB", "PVT_M", "PVT-M", "SI",
             "OIL", "WATER", "GAS", "DISGAS", "VAPOIL", "BLACKOIL",
             "POLYMER", "SURFACT", "BRINE", "TEMP", "THERMAL", "MECH"}
    dimensions = {"TABDIMS", "WELLDIMS", "AQUDIMS", "EQLDIMS", "REGDIMS",
                  "VFPIDIMS", "VFPPDIMS"}
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
            data[kw] = _integer_tokens(record)
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
