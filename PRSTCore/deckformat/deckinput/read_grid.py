"""Read GRID section from an ECLIPSE deck.

This parser is a pragmatic translation of MRST's readGRID: it collects
keyworded arrays and scalar values appearing in the GRID block. It does
not implement the full MRST semantics (e.g. MAPAXES transformation,
NNC processing), but returns a dict with parsed numeric arrays where
possible to ease downstream conversion/processing.
"""

import re
from typing import List
import numpy as np


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
        # skip separators
        if t == '/':
            continue
        # split comma/semicolon separated
        for part in re.split('[,;]', t):
            if part == '':
                continue
            # In ECLIPSE a terminator can be attached to the final repeat
            # token (for example ``3600*4024/``).  MRST's token reader
            # recognizes the repeat before consuming the slash.
            if part != '/' and part.endswith('/'):
                part = part[:-1]
                if not part:
                    continue
            rep = _expand_repeat_token(part)
            if rep is not None:
                vals.extend(rep)
                continue
            v = _try_float(part)
            if v is None:
                # keep literal strings
                vals.append(part)
            else:
                vals.append(v)
    return vals


def _operator_indices(record, dims, size):
    """Port ``applyOperator.box_indices`` for an ECLIPSE I/J/K box."""
    if dims is None or len(dims) < 3:
        return np.arange(size, dtype=int)
    nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
    bounds = [1, nx, 1, ny, 1, nz]
    for i, token in enumerate(record[2:8]):
        try:
            bounds[i] = int(float(str(token).strip("'\"")))
        except (TypeError, ValueError):
            pass
    i1, i2, j1, j2, k1, k2 = bounds
    i1, i2 = max(1, i1), min(nx, i2)
    j1, j2 = max(1, j1), min(ny, j2)
    k1, k2 = max(1, k1), min(nz, k2)
    if i1 > i2 or j1 > j2 or k1 > k2:
        return np.empty(0, dtype=int)
    ii, jj, kk = np.meshgrid(np.arange(i1 - 1, i2),
                             np.arange(j1 - 1, j2),
                             np.arange(k1 - 1, k2), indexing='ij')
    return (ii + nx * jj + nx * ny * kk).ravel(order='F')


def _apply_operators(data, operators, dims):
    """Apply MRST ``applyOperator`` records to GRID arrays.

    This implements the ECLIPSE array operators used by the included SPE9
    and EGG decks.  Records are executed in source order, with MATLAB's
    column-major I/J/K box indexing.
    """
    for name, records in operators:
        for raw in records:
            rec = [str(item).strip("'\"") for item in raw if str(item) != '/']
            if len(rec) < 2:
                continue
            target = rec[0].upper()
            source_or_value = rec[1]
            source = np.asarray(data.get(target, []), dtype=float).ravel()
            if source.size == 0:
                if name == 'COPY':
                    raise ValueError("COPY source array %r is undefined" % target)
                size = int(np.prod(dims)) if dims is not None and len(dims) >= 3 else 1
                source = np.ones(size, dtype=float) if name == 'MULTIPLY' else np.full(size, np.nan)
                data[target] = source
            ix = _operator_indices(rec, dims, source.size)
            if name == 'COPY':
                dest = source_or_value.upper()
                target_values = np.asarray(data.get(dest, np.zeros(source.size)), dtype=float).ravel()
                if target_values.size < source.size:
                    expanded = np.zeros(source.size, dtype=float)
                    expanded[:target_values.size] = target_values
                    target_values = expanded
                target_values[ix] = source[ix]
                data[dest] = target_values
                continue
            try:
                value = float(source_or_value.replace('D', 'E').replace('d', 'e'))
            except ValueError:
                raise ValueError('Invalid %s value %r' % (name, source_or_value))
            if name == 'ADD':
                source[ix] += value
            elif name == 'EQUALS':
                source[ix] = value
            elif name == 'MULTIPLY':
                source[ix] *= value
            elif name == 'MAXVALUE':
                source[ix] = np.minimum(source[ix], value)
                source[ix[np.isnan(source[ix])]] = value
            elif name == 'MINVALUE':
                source[ix] = np.maximum(source[ix], value)
                source[ix[np.isnan(source[ix])]] = value
            data[target] = source
    return data


def read_grid(block, existing=None, cart_dims=None):
    if existing is None:
        existing = {}
    data = dict(existing)

    lines = [ln.split('--', 1)[0].strip() for ln in block.splitlines()]
    current = None
    collected = {}
    operators = []

    # Known array/keyword names commonly found in GRID
    kw_names = set([
        'GRID', 'GRIDFILE', 'DXV', 'DYV', 'DZV', 'DEPTHZ', 'COORD', 'ZCORN',
        'TOPS', 'DX', 'DY', 'DZ', 'PERMX', 'PERMY', 'PERMZ', 'PORO', 'NTG',
        'ACTNUM', 'MULTX', 'MULTY', 'MULTZ', 'MULTFLT', 'FAULTS', 'INCLUDE', 'MAPAXES', 'MAPUNITS',
        'MINPV', 'MINPVV', 'PORV', 'NNC', 'TRANX', 'TRANY', 'TRANZ', 'THCONR',
        'YMODULE', 'PINCH', 'PINCHREG', 'SIGMAV', 'SIGMA', 'DZMTRXV', 'DZMTRX',
        'JFUNC', 'cartDims', 'TOPS'
    ])
    operator_names = {'ADD', 'COPY', 'EQUALS', 'MAXVALUE', 'MINVALUE', 'MULTIPLY'}
    section_controls = {'ECHO', 'NOECHO', 'INIT', 'EDIT', 'PROPS', 'REGIONS',
                        'SOLUTION', 'SCHEDULE', 'SUMMARY', 'END'}

    line_no = 0
    while line_no < len(lines):
        line = lines[line_no]
        line_no += 1
        if not line:
            continue
        parts = line.split()
        head = parts[0].upper()

        if head in section_controls:
            current = None
            continue

        if head in operator_names:
            records = []
            remainder = parts[1:]
            if remainder:
                records.append(remainder[:remainder.index('/') + 1] if '/' in remainder else remainder)
            # Operator records are slash terminated, and a slash-only row
            # terminates the whole keyword.  This is exactly the grammar
            # consumed by MRST ``applyOperator``.
            while line_no < len(lines):
                op_line = lines[line_no]
                line_no += 1
                if not op_line:
                    continue
                op_parts = op_line.split()
                if op_parts == ['/']:
                    break
                if op_parts[0].startswith('--'):
                    continue
                records.append(op_parts[:op_parts.index('/') + 1] if '/' in op_parts else op_parts)
            operators.append((head, records))
            current = None
            continue

        if head in kw_names:
            # start a new keyword collect
            current = head
            rest = parts[1:]
            if current == 'INCLUDE':
                data['INCLUDE_grid'] = ' '.join(rest).strip("'\"")
                current = None
                continue
            if current == 'COPY':
                # COPY blocks express keyword copy operations and are not
                # raw numeric arrays. Keep raw text for future handling.
                collected.setdefault('COPY', []).append(' '.join(rest))
                if '/' in rest:
                    current = None
                continue
            # ECLIPSE records are slash-terminated.  Some of the Norne
            # include files put informal prose after that slash on the same
            # line (e.g. ``/ matand 0.05``).  It is a comment, not part of
            # the following record, and MRST's deck parser discards it.
            if '/' in rest:
                rest = rest[:rest.index('/') + 1]
            collected.setdefault(current, []).extend(rest)
            # single-line terminator '/' may appear
            if '/' in rest:
                current = None
            continue

        # terminator
        if parts[0] == '/':
            current = None
            continue

        # continuation line for previous keyword
        if current is not None:
            if current == 'COPY':
                collected.setdefault('COPY', []).append(' '.join(parts))
                if '/' in parts:
                    current = None
                continue
            if '/' in parts:
                parts = parts[:parts.index('/') + 1]
                collected.setdefault(current, []).extend(parts)
                current = None
                continue
            collected.setdefault(current, []).extend(parts)
        else:
            # stray values - ignore or keep under 'UNKNOWN'
            collected.setdefault('UNKNOWN', []).extend(parts)

    # Convert collected tokens into numeric arrays where sensible
    for k, toks in collected.items():
        if k in {'FAULTS', 'MULTFLT'}:
            records, current_record = [], []
            for token in toks:
                if token == '/':
                    if current_record:
                        records.append(current_record)
                        current_record = []
                    continue
                current_record.append(token.strip("'\""))
            if current_record:
                records.append(current_record)
            parsed = []
            for record in records:
                if not record:
                    continue
                row = [record[0]]
                for item in record[1:]:
                    value = _try_float(item)
                    row.append(value if value is not None else item.strip("'\""))
                parsed.append(row)
            data[k] = parsed
            continue
        # Special handling for NNC: groups of 7 numeric fields per row,
        # possibly separated by '/'.
        if k == 'NNC':
            rows = []
            cur = []
            for tok in toks:
                if tok == '/':
                    if cur:
                        rows.append([_try_float(t) for t in cur])
                        cur = []
                    continue
                cur.append(tok)
            if cur:
                rows.append([_try_float(t) for t in cur])

            # Filter and convert
            nnc_rows = []
            for r in rows:
                nums = [x for x in r if x is not None]
                if nums:
                    nnc_rows.append(nums)
            if nnc_rows:
                try:
                    arr = np.asarray(nnc_rows, dtype=float)
                    data[k] = arr.reshape(-1, arr.shape[1])
                except Exception:
                    data[k] = np.asarray(nnc_rows, dtype=float)
            continue

        # Special handling for JFUNC: keep records split by '/'
        if k == 'JFUNC':
            records = []
            cur = []
            for tok in toks:
                if tok == '/':
                    if cur:
                        records.append(_flatten_tokens(cur))
                        cur = []
                    continue
                cur.append(tok)
            if cur:
                records.append(_flatten_tokens(cur))
            data[k] = records
            continue

        vals = _flatten_tokens(toks)
        if not vals:
            continue
        # If all values are numeric, store as numpy array
        if all(isinstance(v, (int, float)) for v in vals):
            arr = np.asarray(vals, dtype=float)
            data[k] = arr
        else:
            data[k] = vals

    dims = cart_dims if cart_dims is not None else data.get('cartDims')
    return _apply_operators(data, operators, dims)
