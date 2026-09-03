"""Read PROPS section from an ECLIPSE deck.

This implements a pragmatic parser that groups PROPS records by keyword
and attempts to parse numeric values into arrays. It retains non-numeric
tokens as strings for downstream processing.
"""

import re
from typing import List
import numpy as np

from .read_grid import _apply_operators


def _try_float(tok: str):
    try:
        return float(tok.replace('D', 'E').replace('d', 'e'))
    except Exception:
        return None


def _expand_repeat_token(tok: str):
    m = re.fullmatch(r'([+-]?\d+)\*([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eEdD][+-]?\d+)?)', tok.strip())
    if not m:
        return None
    n = int(m.group(1))
    if n < 0:
        return None
    v = float(m.group(2).replace('D', 'E').replace('d', 'e'))
    return [v] * n


def _flatten_tokens(tokens: List[str]):
    vals = []
    for t in tokens:
        for part in re.split('[,;]', t):
            if part == '' or part == '/':
                continue
            rep = _expand_repeat_token(part)
            if rep is not None:
                vals.extend(rep)
                continue
            v = _try_float(part)
            if v is None:
                vals.append(part)
            else:
                vals.append(v)
    return vals


def _parse_miscible_pvt_records(lines, keyword):
    """Preserve the ``key/pos/data`` grammar of MRST miscible PVT tables.

    PVTO/PVTG use slash-terminated key records, with a slash-only line
    ending one PVT region.  A flat numeric vector cannot recover that
    boundary and therefore cannot be converted or interpolated faithfully.
    """
    regions = []
    region = []
    record = []
    for parts in lines:
        has_slash = '/' in parts
        values = _flatten_tokens([part for part in parts if part != '/'])
        if values:
            if not all(isinstance(value, (int, float)) for value in values):
                raise ValueError('%s contains a non-numeric PVT record' % keyword)
            record.extend(float(value) for value in values)
        if has_slash:
            if record:
                region.append(np.asarray(record, dtype=float))
                record = []
            elif region:
                regions.append(region)
                region = []
    if record:
        raise ValueError('%s ended before its slash-terminated PVT record' % keyword)
    if region:
        regions.append(region)
    return regions


#: PROPS keywords that ECLIPSE reads as a fixed number of items per record,
#: with the trailing items defaultable.  MRST reads these through
#: ``readFixedNumRecords(fid, tmpl, ntpvt)`` with an explicit template, so a
#: short record comes back padded rather than short --
#: ``readPROPS.m`` case 'PVTW' uses ``repmat({'0.0'}, [1, 5])``.
#:
#: Without the padding a defaulted trailing item silently truncates the
#: record, and every consumer that slices it in fixed-width groups then sees
#: nothing: QIEDIE's ``PVTW 1.01325 1 3.9e-5 0.3 /`` defaults the fifth item
#: (viscosibility), and the four numbers that survive make
#: ``(size // 5) * 5`` zero -- so the unit conversion is skipped and the
#: table is dropped, leaving water at the 1 Pa*s placeholder instead of the
#: deck's 0.3 cP.
_FIXED_RECORD_PROPS = {
    'PVTW': (5, 0.0),
}


def _pad_fixed_records(tokens, width, default):
    """Port of ``readFixedNumRecords``: pad each record out to ``width``.

    ``tokens`` still carries ECLIPSE's ``/`` record terminators, so the
    record boundaries a short record would otherwise hide are still visible
    here.
    """
    out, record = [], []
    for tok in tokens:
        if tok != '/':
            record.append(tok)
            continue
        if record or out or True:
            while len(record) < width:
                record.append(repr(default))
            out.extend(record)
        record = []
    if record:
        while len(record) < width:
            record.append(repr(default))
        out.extend(record)
    return out


def read_props(block, cart_dims=None):
    lines = [ln.split('--', 1)[0].strip() for ln in block.splitlines()]
    records = {}
    miscible_lines = {}
    operators = []
    current = None

    line_no = 0
    while line_no < len(lines):
        line = lines[line_no]
        line_no += 1
        if not line:
            continue
        parts = line.split()
        # A slash terminates an ECLIPSE record even when a legacy deck has
        # an explanatory comment on the same line (Norne's DENSITY data is
        # one such case).  MRST's readDefaultedKW never consumes tokens to
        # the right of this terminator.
        if '/' in parts:
            parts = parts[:parts.index('/') + 1]
        if not parts:
            continue
        head = parts[0].upper()
        if head == 'SCALECRS':
            # SCALECRS is a one-record character keyword.  Without this
            # explicit branch, a following ``YES /`` line is mistaken for
            # a new keyword and MRST's three-point endpoint scaling is
            # silently disabled.
            value_parts = parts[1:]
            if not value_parts:
                while line_no < len(lines):
                    value_line = lines[line_no]
                    line_no += 1
                    if not value_line:
                        continue
                    value_parts = value_line.split()
                    if '/' in value_parts:
                        value_parts = value_parts[:value_parts.index('/')]
                    break
            else:
                value_parts = value_parts[:value_parts.index('/')] if '/' in value_parts else value_parts
            if value_parts:
                records['SCALECRS'] = value_parts
            current = None
            continue
        if head in {'ADD', 'COPY', 'EQUALS', 'MAXVALUE', 'MINVALUE', 'MULTIPLY'}:
            op_records = []
            remainder = parts[1:]
            if remainder:
                op_records.append(remainder)
            # Operators are a sequence of slash-terminated records followed
            # by one slash-only record, exactly as applyOperator.m reads.
            while line_no < len(lines):
                op_line = lines[line_no]
                line_no += 1
                if not op_line:
                    continue
                op_parts = op_line.split()
                if '/' in op_parts:
                    op_parts = op_parts[:op_parts.index('/') + 1]
                if op_parts == ['/']:
                    break
                if op_parts:
                    op_records.append(op_parts)
            operators.append((head, op_records))
            current = None
            continue
        # Heuristic: keywords in PROPS are uppercase words (e.g., ROCK, ROCKTAB)
        if re.fullmatch(r'[A-Z][A-Z0-9_]*', head):
            current = head
            records.setdefault(current, []).extend(parts[1:])
            if current in {'PVTO', 'PVTG'} and parts[1:]:
                miscible_lines.setdefault(current, []).append(parts[1:])
            if '/' in parts:
                current = None
            continue

        # continuation
        if current is not None:
            records.setdefault(current, []).extend(parts)
            if current in {'PVTO', 'PVTG'}:
                miscible_lines.setdefault(current, []).append(parts)
                # PVTO/PVTG have one slash per PVT-region record, with the
                # keyword itself only closed by the next real keyword (or a
                # lone-slash record); an inline slash here is a record
                # separator, not the keyword terminator.
            # A table keyword (SWOF, SGOF, SURFADS, SURFROCK, SURFCAPD, ...)
            # may carry several directly-concatenated `/`-terminated blocks,
            # one per NTSFUN/NSURFNUM region, with no re-mention of the
            # keyword name between them -- ``current`` must stay set across
            # each inline `/` so later blocks are not misfiled as
            # 'UNKNOWN'.  The keyword ends only when a genuine new KEYWORD
            # line is recognised (top branch, above) or the section does;
            # ECLIPSE table-data rows never start with a bare uppercase
            # token that could be confused with one.
        else:
            records.setdefault('UNKNOWN', []).extend(parts)

    out = {}
    for k, toks in records.items():
        if k == 'COPY':
            out[k] = toks
            continue
        # Special handling for ROCKTAB: split into tables separated by '/'
        if k == 'ROCKTAB':
            tables = []
            cur = []
            for tok in toks:
                if tok == '/':
                    if cur:
                        vals = _flatten_tokens(cur)
                        if vals:
                            tables.append(np.asarray(vals, dtype=float))
                        cur = []
                    continue
                cur.append(tok)
            if cur:
                vals = _flatten_tokens(cur)
                if vals:
                    tables.append(np.asarray(vals, dtype=float))
            out[k] = tables
            continue

        fixed = _FIXED_RECORD_PROPS.get(k)
        if fixed is not None:
            toks = _pad_fixed_records(toks, fixed[0], fixed[1])

        vals = _flatten_tokens(toks)
        if not vals:
            continue
        if all(isinstance(v, (int, float)) for v in vals):
            out[k] = np.asarray(vals, dtype=float)
        else:
            out[k] = vals

    if miscible_lines:
        out['_miscible_pvt_records'] = {
            keyword: _parse_miscible_pvt_records(lines, keyword)
            for keyword, lines in miscible_lines.items()
        }

    # Same source-order semantics as readPROPS/applyOperator.  This covers
    # Norne ENDSCALE endpoints (SWL/SWCR/SGU/...): the target arrays are
    # full Cartesian vectors before initEclipseRock indexes active cells.
    out = _apply_operators(out, operators, cart_dims)

    return out
