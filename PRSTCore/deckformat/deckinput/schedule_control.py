"""Port of the control-splitting machinery in MRST's ``readSCHEDULE.m``.

MRST does not keep the SCHEDULE section as a flat bag of keywords. It
walks the section keeping one *current* control -- the running set of
WELSPECS/COMPDAT/WCON* tables -- and snapshots it every time a DATES or
TSTEP record advances the clock::

    case {well keywords}
       if ~def_ctrl, def_ctrl = true; cno = cno + 1;
                     ctrl = defaultControl(ctrl);   end
       ctrl = readWellKW(fid, ctrl, kw, ncomp);

    case {'DATES', 'TSTEP'}
       if def_ctrl, def_ctrl = false; schd.control = [schd.control; ctrl]; end
       schd.step.control = [schd.step.control; repmat(cno, numel(data), 1)];
       schd.step.val     = [schd.step.val; data];

so a deck with 63 WCONHIST blocks separated by DATES produces 63
controls, each carrying forward the wells and completions declared
earlier and replacing only the rate targets that the block restates.
``writeSchedule`` -- and by extension ``writeDeck``, which the history
match calls once per iteration to hand a fresh deck to ECLIPSE -- reads
exactly that structure and writes nothing at all without it.

This is ported from MRST-0's copy rather than 2026a's: that is the tree
whose ``writeSchedule`` PRSTCore already follows, and it carries the
``% edited by zhang`` changes (WEFAC as a control keyword, well names
truncated to eight characters, COMPDAT item 12 read as a number and item
13 truncated to a single direction letter, empty record entries treated
as ``1*``).

**Additive by design.** :func:`~PRSTCore.deckformat.deckinput.read_schedule.read_schedule`
still returns every flat keyword list it always did; this module only
adds ``control`` and ``step`` beside them, so no existing consumer of the
deck changes behaviour.

Two deliberate departures from the MATLAB, both in the direction of not
losing a deck PRSTCore could otherwise read:

* an unsupported well keyword is recorded in ``missing`` rather than
  raising, because this parser runs on every deck in the suite while
  MRST's runs only where the caller already committed to the section;
* a numeric item whose default is a non-numeric template string (COMPDAT
  K1/K2, WELSPECS I/J) becomes ``None`` rather than MATLAB's ``[]``,
  which ``writeSchedule`` writes back as ``1*``.
"""

import re
from datetime import date as _date

__all__ = ['read_schedule_control', 'parse_eclipse_date', 'default_control',
           'CONTROL_KEYWORDS']


# ------------------------------------------------------------- tokens --

_COMMENT = re.compile(r'(--|#)')

#: ``getEclipseKeyword``'s test: up to eight upper-case alphanumerics at
#: the very start of a line, optionally followed by the terminator.
_KEYWORD = re.compile(r'^[A-Z][A-Z0-9]{0,7}(|/)')
_KEYWORD_HEAD = re.compile(r'^([A-Z][A-Z0-9]*)')

#: ``readDefaultedRecord``'s default marker.  Note that MRST recognises
#: only ``n*`` -- a bare ``*`` is read as a literal value.
_DEFAULT_MARKER = re.compile(r'^(\d+)\*$')

#: What MATLAB's ``sscanf(v, '%f')`` accepts: a leading number, stopping
#: at the first character that cannot continue one.  ``'1*'`` therefore
#: reads as ``1``, and ``'Default'`` as nothing at all.
_NUMBER = re.compile(r'\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?'
                     r'|[-+]?[Ii][Nn][Ff]|[-+]?[Nn][Aa][Nn])')

_MONTHS = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
           'JUL': 7, 'JLY': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11,
           'DEC': 12}


def _split_quoted(s):
    """Port of ``splitQuotedString``.

    Returns the quoted runs (quotes included) and the unquoted runs
    between them, so that a ``--`` or ``/`` inside a quoted well name is
    not mistaken for a comment or a record terminator.
    """
    s = s.strip()
    marks = [i for i, c in enumerate(s) if c == "'"]
    if not marks:
        return [''], [s]
    if len(marks) % 2 != 0:
        raise ValueError('Non-terminated quoted string in input: %r' % s)

    quoted = [s[b:e + 1] for b, e in zip(marks[0::2], marks[1::2])]

    # The unquoted runs are the gaps: before the first quote, between
    # each closing and the next opening quote, and after the last.
    bounds = [-1] + marks + [len(s)]
    unquoted = [s[b + 1:e] for b, e in zip(bounds[0::2], bounds[1::2])]
    return quoted, unquoted


def _assemble(quoted, unquoted):
    """Inverse of :func:`_split_quoted`, as ``assembleString`` is."""
    out = []
    for i, u in enumerate(unquoted):
        out.append(u)
        if i < len(quoted) and quoted[i]:
            out.append(quoted[i])
    return ''.join(out)


def _tokenize_record(data):
    """Port of ``tokenizeRecord``: split on whitespace, keeping quoted
    runs whole."""
    quoted, unquoted = _split_quoted(data)
    if len(unquoted) > 1:
        tokens = []
        for i, u in enumerate(unquoted[:-1]):
            tokens.extend(u.split())
            tokens.append(quoted[i])
        tokens.extend(unquoted[-1].split())
        return [t for t in tokens if t != '']
    return unquoted[0].strip().split()


def _remove_quotes(token):
    """Port of ``removeQuotes``/``dequote``."""
    return token.replace("'", '').strip()


class _Cursor(object):
    """The ``fid`` MRST passes around, over an already-included section.

    MRST reads the SCHEDULE section straight from the file, alternating
    between ``getEclipseKeyword`` and the per-keyword record readers.
    Here the section text has already had its INCLUDEs expanded by
    ``read_eclipse_deck``, so the same alternation runs over a list of
    lines instead.
    """

    def __init__(self, lines):
        self._lines = list(lines)
        self._pos = 0

    def _next_line(self):
        if self._pos >= len(self._lines):
            return None
        line = self._lines[self._pos]
        self._pos += 1
        return line

    @property
    def at_end(self):
        return self._pos >= len(self._lines)

    def get_keyword(self):
        """Port of ``getEclipseKeyword``."""
        line = self._next_line()
        while line is not None and not _KEYWORD.match(line):
            line = self._next_line()
        if line is None:
            return None
        if line.endswith('/'):
            line = line[:-1]
        match = _KEYWORD_HEAD.match(line)
        return match.group(1) if match else None

    def read_record_string(self):
        """Port of ``readRecordString``.

        Accumulates lines until a ``/`` appears outside quotes, dropping
        ``--``/``#`` comments as it goes, and returns the text up to but
        not including that terminator.
        """
        data = ''
        while True:
            line = self._next_line()
            if line is None:
                break

            if re.match(r'^\s*(--|#)', line):
                # Whole-line comment: skip without terminating the record.
                continue

            quoted, unquoted = _split_quoted(line)

            # First unquoted run holding a comment truncates the line.
            for i, u in enumerate(unquoted):
                found = _COMMENT.search(u)
                if found:
                    quoted = quoted[:i]
                    unquoted = unquoted[:i + 1]
                    unquoted[-1] = u[:found.start()]
                    break

            # First unquoted run holding '/' terminates the record.
            done = False
            for i, u in enumerate(unquoted):
                if '/' in u:
                    quoted = quoted[:i]
                    unquoted = unquoted[:i + 1]
                    unquoted[-1] = u[:u.index('/') + 1]
                    done = True
                    break

            data = data + ' ' + _assemble(quoted, unquoted)
            if done:
                break

        cut = data.rfind('/')
        if cut >= 0:
            data = data[:cut]
        return data

    def read_vector(self):
        """Port of ``readVector(fid, kw, inf)``: numbers to the ``/``,
        with ``n*v`` repeat counts expanded."""
        text = self.read_record_string()
        values = []
        for token in text.split():
            token = _remove_quotes(token)
            if not token:
                continue
            match = re.match(r'^(\d+)\*(.*)$', token)
            if match:
                count = int(match.group(1))
                value = _to_float(match.group(2)) if match.group(2) else 0.0
                values.extend([value] * count)
            else:
                value = _to_float(token)
                if value is not None:
                    values.append(value)
        return values


def _to_float(text):
    try:
        return float(str(text).replace('D', 'E').replace('d', 'e'))
    except (TypeError, ValueError):
        return None


# -------------------------------------------------- defaulted records --

def read_defaulted_record(cursor, template):
    """Port of ``readDefaultedRecord``.

    Returns a copy of ``template`` with the record's entries placed at
    the right positions, ``n*`` skipping that many. A blank record gives
    back the template unchanged, which is how ``readDefaultedKW`` knows
    the keyword ended.
    """
    rec = list(template)
    data = cursor.read_record_string()
    if not data or data.isspace():
        return rec

    tokens = [_remove_quotes(t) for t in _tokenize_record(data.strip())]
    # ``replaceEmpty`` (% edited by zhang): an empty entry means default.
    tokens = [t if t != '' else '1*' for t in tokens]

    marks = [_DEFAULT_MARKER.match(t) for t in tokens]
    if all(m is not None for m in marks) and marks:
        return rec

    # ``add`` counts the slots skipped before each supplied entry.
    skipped = 0
    slot = 0
    for token, mark in zip(tokens, marks):
        if mark is not None:
            skipped += int(mark.group(1))
            continue
        index = slot + skipped
        if index < len(rec):
            rec[index] = token
        slot += 1
    return rec


def read_defaulted_kw(cursor, template, nrec=None):
    """Port of ``readDefaultedKW``: records until one equals the template."""
    data = []
    count = 1
    rec = read_defaulted_record(cursor, template)
    while rec != list(template) and (nrec is None or count < nrec):
        data.append(rec)
        rec = read_defaulted_record(cursor, template)
        count += 1
    if rec != list(template):
        data.append(rec)
    return data


def _to_double(rows, numeric):
    """Port of ``toDouble``: convert the listed columns in place.

    MATLAB's ``sscanf(v, '%f')`` reads the leading number and stops, so
    ``'1*'`` becomes ``1`` and a non-numeric template default such as
    ``'Default'`` yields an empty matrix -- represented here as ``None``,
    which ``writeSchedule`` writes back as ``1*``.
    """
    for row in rows:
        for col in numeric:
            if col < len(row) and isinstance(row[col], str):
                match = _NUMBER.match(row[col])
                if match is None:
                    row[col] = None
                    continue
                token = match.group(1).replace('D', 'E').replace('d', 'e')
                row[col] = float(token)
    return rows


def _replace_well_name(rows, col=0, warn=False):
    """Port of ``relpaceWellName`` (% edited by zhang): ECLIPSE well names
    are at most eight characters, so a longer one is chopped."""
    for row in rows:
        name = row[col]
        if isinstance(name, str) and len(name) > 8:
            if warn:
                import warnings
                warnings.warn("The length of well name: '%s' is great than "
                              '8. We have chopped it.' % name, RuntimeWarning)
            row[col] = name[:8]
    return rows


# ------------------------------------------------------- the templates --

#: ``(template, numeric columns)`` for each keyword, straight out of
#: MRST-0's ``readWellKW.m``.  Column indices are 0-based here; the
#: MATLAB's are 1-based.
_TEMPLATES = {
    'WELSPECS': (['Default', 'Default', 'Default', 'Default', 'NaN',
                  'Default', '0.0', 'STD', 'SHUT', 'YES', '0', 'SEG', '0',
                  'FrontSim', 'FrontSim', 'STD'],
                 [2, 3, 4, 6, 10, 12]),
    'COMPDAT': (['Default', '-1', '-1', 'Default', 'Default', 'OPEN', '-1',
                 '-1.0', '0.0', '-1.0', '0.0', '-1', 'Z', '-1'],
                [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13]),
    'WCONHIST': (['Default', 'OPEN', 'Default', '0.0', '0.0', '0.0', '0',
                  'Default', '0.0', '0.0', '0.0'],
                 [3, 4, 5, 6, 8, 9, 10]),
    'WCONINJH': (['Default', 'Default', 'OPEN', '0.0', '0.0', '0.0', '0',
                  '0.0', '0', '0.0', '0.0'],
                 [3, 4, 5, 6, 7, 8, 9, 10]),
    'WCONINJE': (['Default', 'Default', 'OPEN', 'Default', 'inf', 'inf',
                  'NaN', 'inf', '0', '0.0', '0.0', '0', '0', '0'],
                 [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    'WCONINJ': (['Default', 'Default', 'OPEN', 'Default', 'inf', 'inf',
                 '0.0', 'NONE', '-1', 'inf', '0.0', '0.0'],
                [4, 5, 6, 8, 9, 10, 11]),
    'WCONPROD': (['Default', 'OPEN', 'Default', 'inf', 'inf', 'inf', 'inf',
                  'inf', 'NaN', '0.0', '0', '0.0'],
                 [3, 4, 5, 6, 7, 8, 9, 10, 11]),
    'WELTARG': (['Default', 'Default', 'NaN'], [2]),
    'WELCNTL': (['Default', 'Default', 'NaN'], [2]),
    'WEFAC': (['Default', '1.0', 'YES'], [1]),
    'WPIMULT': (['Default', '1.0', '-1', '-1', '-1', '-1', '-1'],
                [1, 2, 3, 4, 5, 6]),
    'WELOPEN': (['Default', 'OPEN', '-1', '-1', '-1', '-1', '-1'],
                [2, 3, 4, 5, 6]),
    'WTEMP': (['Default', '0.0'], [1]),
    'WSOLVENT': (['Default', '0.0'], [1]),
    'WSURFACT': (['Default', '0.0'], [1]),
    'WPOLYMER': (['Default', '0.0', '0.0', 'Default', 'Default'], [1, 2]),
    'WINJGAS': (['Default', 'Default', 'Default', 'Default', '0.0'], [4]),
    'GINJGAS': (['Default', 'Default', 'Default', 'Default', '0.0'], [4]),
    'WGRUPCON': (['Default', 'YES', '-1.0', '', '1.0'], [2, 4]),
    'GRUPTREE': (['Default', 'Default'], []),
    'GRUPNET': (['Default', '-1.0', '0', '0', 'NO', 'NO', 'NONE'], [1, 2, 3]),
    'GECON': (['Default', '0.0', '0.0', '0.0', '0.0', '0.0', 'NONE', 'NO',
               '0'],
              [1, 2, 3, 4, 5, 8]),
    'GCONINJE': (['Default', 'Default', 'NONE', 'Inf', 'Inf', 'Inf', 'Inf',
                  'YES', 'NaN', '', 'NaN', 'NaN', 'Inf'],
                 [3, 4, 5, 6, 8, 10, 11, 12]),
    'GCONPROD': (['Default', 'NONE', 'inf', 'inf', 'inf', 'inf', 'NONE',
                  'YES', 'inf', '', 'NONE', 'NONE', 'NONE', 'inf', 'inf',
                  'inf', 'inf', 'inf', 'inf', 'inf', 'NONE'],
                 [2, 3, 4, 5, 8, 13, 14, 15, 16, 17, 18, 19]),
}

#: The keywords whose first entry after a DATES/TSTEP opens a new
#: control.  Taken verbatim from MRST-0's ``readSCHEDULE`` case list --
#: the set matters even for keywords whose records this module stores
#: without interpreting, because it is what makes the control *numbering*
#: agree with MRST's.
CONTROL_KEYWORDS = frozenset({
    'COMPDAT', 'COMPSEGS', 'GCONINJE', 'GCONPROD', 'GECON', 'GINJGAS',
    'GRUPTREE', 'GRUPNET', 'RPTSCHED', 'RPTRST', 'WELLSTRE', 'WINJGAS',
    'WCONHIST', 'WCONINJ', 'WCONINJE', 'WCONINJH', 'WCONPROD', 'WELSEGS',
    'WELSPECS', 'WELOPEN', 'WELLOPEN', 'WELTARG', 'WELLTARG', 'WELCNTL',
    'WPIMULT', 'WGRUPCON', 'WPOLYMER', 'WSURFACT', 'WSOLVENT', 'WTEMP',
    'WEFAC', 'BCPROP',
})

#: ``WELLOPEN``/``WELLTARG`` are ECLIPSE aliases MRST accepts alongside
#: the six-character spellings.
_ALIASES = {'WELLOPEN': 'WELOPEN', 'WELLTARG': 'WELTARG'}

#: ``defaultControl`` accumulates these across controls.
_ACCUMULATED = ('WELSPECS', 'COMPDAT', 'WELLSTRE', 'WINJGAS')

#: ...and copies these forward whole, so that a control which restates
#: nothing keeps the previous targets (this is what honours WELTARG).
_CARRIED = ('WCONINJE', 'WCONINJH', 'WCONINJ', 'WCONPROD', 'WCONHIST',
            'GCONINJE', 'GCONPROD', 'GECON', 'GRUPTREE', 'GRUPNET',
            'GINJGAS', 'COMPSEGS', 'WELSEGS', 'RPTSCHED', 'RPTRST',
            'VFPINJ', 'VFPPROD')

#: Every field ``defaultControl`` declares, so a control always has the
#: same shape whether or not the deck used the keyword.
_FIELDS = ('WELSPECS', 'COMPDAT', 'COMPSEGS', 'WELLSTRE', 'WINJGAS',
           'WCONINJ', 'WCONINJE', 'WCONINJH', 'WCONPROD', 'WCONHIST',
           'GCONINJE', 'GCONPROD', 'GECON', 'GINJGAS', 'GRUPTREE',
           'GRUPNET', 'RPTSCHED', 'RPTRST', 'WELSEGS', 'WGRUPCON',
           'WPOLYMER', 'WSURFACT', 'WSOLVENT', 'VFPINJ', 'VFPPROD',
           'WPIMULT', 'WEFAC', 'WTEMP', 'WELTARG', 'WELCNTL', 'BOX',
           'MULTPV', 'MULTX', 'MULTX_', 'MULTY', 'MULTY_', 'MULTZ',
           'MULTZ_', 'BCPROP', 'TUNING', 'TUNINGDP')


def _copy_table(rows):
    """A control's tables are MATLAB cell arrays, which copy by value.

    Python lists do not, and several readers -- WELOPEN's status rewrite
    most visibly -- edit a record in place.  Without this the edit would
    reach back through every control that carried the record forward and
    silently rewrite history.
    """
    return [list(row) if isinstance(row, list) else row for row in rows]


def default_control(previous=None):
    """Port of ``defaultControl``.

    A fresh control with everything empty, or -- given the outgoing
    control -- one that has inherited the wells and completions declared
    so far and whichever rate targets were last in force.
    """
    control = {name: [] for name in _FIELDS}
    control['VAPPARS'] = [0.0, 0.0]
    control['DRSDT'] = [float('inf'), 'ALL']
    if previous is None:
        return control

    for name in _ACCUMULATED:
        control[name] = _copy_table(previous.get(name) or [])
    control['DRSDT'] = list(previous.get('DRSDT', [float('inf'), 'ALL']))
    for name in _CARRIED:
        if previous.get(name):
            control[name] = _copy_table(previous[name])
    return control


# ------------------------------------------------- record bookkeeping --

def _unique_well_records(data):
    """Port of ``unique_well_records``: last record wins per well."""
    last = {}
    for i, row in enumerate(data):
        last[row[0]] = i
    keep = set(last.values())
    return [row for i, row in enumerate(data) if i in keep]


def _matches(name, pattern):
    return re.search(pattern, name) is not None


def _append_spec(table, data, wells):
    """Port of ``appendSpec``.

    Expands ``NAME*`` templates against the declared wells, drops any
    record for a well that was never declared, and -- crucially -- drops
    the carried-forward record for a well this keyword now restates.
    """
    table = list(table or [])
    data = _unique_well_records(list(data))

    is_wc = [bool(re.search(r'\w+\*\s*$', row[0])) for row in data]
    for i, wc in enumerate(is_wc):
        if not wc:
            continue
        patt = data[i][0].replace('*', '.*')
        table = [row for row in table if not _matches(row[0], patt)]

        matched = [w for w in wells if _matches(w, patt)]
        if not matched:
            import warnings
            warnings.warn("Well control wildcard '%s' does not match any "
                          'previously defined wells.  Ignored.' % data[i][0],
                          RuntimeWarning)
        for well in matched:
            row = list(data[i])
            row[0] = well
            table.append(row)

    data = [row for row, wc in zip(data, is_wc) if not wc]

    if data and not all(row[0] in wells for row in data):
        unknown = [row[0] for row in data if row[0] not in wells]
        import warnings
        warnings.warn('Well control specified in undeclared wells:%s.\n'
                      'Check input to verify that well was declared using '
                      'WELSPECS.' % ''.join(" '%s'" % u for u in unknown),
                      RuntimeWarning)
        return table

    names = {row[0] for row in data}
    table = [row for row in table if row[0] not in names]
    return table + data


def _exclude_set(kw):
    """Port of ``excludeSet``: the other four rate keywords."""
    every = ('WCONINJ', 'WCONINJE', 'WCONINJH', 'WCONPROD', 'WCONHIST')
    return tuple(k for k in every if k.upper() != kw.upper())


def _assign_control_records(control, data, kw, wells):
    """Port of ``assignControlRecords``.

    Placing a well under one rate keyword removes it from the other four,
    which is what stops a producer that becomes an injector from carrying
    its old WCONHIST record forward.
    """
    control[kw] = _append_spec(control.get(kw), data, wells)
    names = {row[0] for row in data}
    for other in _exclude_set(kw):
        if control.get(other):
            control[other] = [row for row in control[other]
                              if row[0] not in names]
    return control


def _well_names(control):
    return [row[0] for row in (control.get('WELSPECS') or [])]


# ----------------------------------------------------- keyword readers --

def _read_welspecs(cursor, control):
    """Port of ``readWellSpec``: restated wells replace their earlier
    record in place; new ones are appended."""
    template, numeric = _TEMPLATES['WELSPECS']
    spec = read_defaulted_kw(cursor, template)
    if not spec:
        return control
    _replace_well_name(spec, 0, warn=True)
    _to_double(spec, numeric)

    existing = control.get('WELSPECS') or []
    if existing:
        index = {row[0]: i for i, row in enumerate(existing)}
        merged = [list(row) for row in existing]
        fresh = []
        for row in spec:
            if row[0] in index:
                merged[index[row[0]]] = row
            else:
                fresh.append(row)
        spec = merged + fresh
    control['WELSPECS'] = spec
    return control


def _read_compdat(cursor, control):
    """Port of ``readCompDat``, including the two `% edited by zhang`
    changes: item 12 is numeric, and a multi-character direction that is
    not FX/FY/FZ keeps only its first letter."""
    template, numeric = _TEMPLATES['COMPDAT']
    compdat = read_defaulted_kw(cursor, template)
    if not compdat:
        return control
    _replace_well_name(compdat, 0)
    _to_double(compdat, numeric)

    for row in compdat:                                  # edited by zhang
        direction = row[12]
        if (isinstance(direction, str) and len(direction) > 1
                and direction.upper() not in ('FX', 'FY', 'FZ')):
            row[12] = direction[0]

    wells = _well_names(control)
    expanded = []
    for row in compdat:
        if '*' in row[0]:
            patt = row[0].replace('*', '.*')
            matched = [w for w in wells if _matches(w, patt)]
            for well in matched:
                clone = list(row)
                clone[0] = well
                expanded.append(clone)
        else:
            expanded.append(row)
    compdat = expanded

    existing = control.get('COMPDAT') or []
    if not existing:
        control['COMPDAT'] = compdat
        return control

    ext_wells = {row[0] for row in existing}
    new_wells = {row[0] for row in compdat}
    for name in new_wells:
        if name.startswith('*') or name.endswith('*'):
            raise ValueError('MRST does not support well lists or templates '
                             'in COMPDAT')

    if not (ext_wells & new_wells):
        control['COMPDAT'] = existing + compdat
        return control

    control['COMPDAT'] = _handle_overlap_compdat(control, existing, compdat)
    return control


def _compdat_expand(rows, ij):
    """Port of ``expand``: fill a defaulted I/J from WELSPECS and split a
    K-range record into one record per completion."""
    out = []
    for row in rows:
        row = list(row)
        if _is_number(row[1]) and row[1] < 1:
            row[1] = ij[0]
        if _is_number(row[2]) and row[2] < 1:
            row[2] = ij[1]
        lo, hi = row[3], row[4]
        if _is_number(lo) and _is_number(hi) and hi > lo:
            for k in range(int(lo), int(hi) + 1):
                clone = list(row)
                clone[3] = float(k)
                clone[4] = float(k)
                out.append(clone)
        else:
            out.append(row)
    return out


def _handle_overlap_compdat(control, existing, compdat):
    """Port of ``handleOverlapCompdat``.

    A COMPDAT block that names a well already completed may be adding
    completions or re-specifying existing ones (a new well index, say).
    Both tables are first expanded to one record per connection, then
    every pair of records addressing the same (I, J, K1, K2) is matched
    and the existing record replaced by the new one::

        k = all(oloc == nloc, 2);
        if any(k), ocd(i(k), :) = ncd(j(k), :); end
        append = [ append ; ocd ; ncd(unique(j(~k)), :) ];

    Note what the last line does. ``[i, j]`` is the full cross product of
    the two tables, so a new record that matched one existing record is
    still paired with every *other* existing record, and those pairings
    put it back into ``j(~k)``.  A matched record is therefore written
    twice: once in place of the record it replaced, and once more
    appended at the end.  Norne relies on this -- its control 11 has
    B-4H's three reopened connections both in place and repeated at the
    end of the table -- so the duplication is reproduced rather than
    tidied away.
    """
    welspecs = {row[0]: row for row in (control.get('WELSPECS') or [])}
    overlap = {row[0] for row in existing} & {row[0] for row in compdat}

    affected = [False] * len(existing)
    handled = [False] * len(compdat)
    append = []

    for well in sorted(overlap):
        spec = welspecs.get(well)
        ij = (spec[2], spec[3]) if spec is not None else (None, None)

        old_ix = [i for i, row in enumerate(existing) if row[0] == well]
        new_ix = [i for i, row in enumerate(compdat) if row[0] == well]
        for i in old_ix:
            affected[i] = True
        for i in new_ix:
            handled[i] = True

        ocd = _compdat_expand([existing[i] for i in old_ix], ij)
        ncd = _compdat_expand([compdat[i] for i in new_ix], ij)

        matched = [(oi, ni)
                   for oi, orow in enumerate(ocd)
                   for ni, nrow in enumerate(ncd)
                   if all(orow[c] == nrow[c] for c in (1, 2, 3, 4))]
        for oi, ni in matched:
            ocd[oi] = ncd[ni]

        pairs = {(oi, ni) for oi in range(len(ocd)) for ni in range(len(ncd))}
        unmatched = sorted({ni for oi, ni in pairs - set(matched)})

        append.extend(ocd)
        append.extend(ncd[ni] for ni in unmatched)

    return ([row for row, a in zip(existing, affected) if not a]
            + [row for row, h in zip(compdat, handled) if not h]
            + append)


def _read_welopen(cursor, control):
    """Port of ``readWelOpen``.

    With every perforation index defaulted the record opens or shuts the
    *well*, so it rewrites the status column of whichever rate keyword
    holds it; otherwise it rewrites the status of the matching COMPDAT
    connections.
    """
    template, numeric = _TEMPLATES['WELOPEN']
    data = read_defaulted_kw(cursor, template)
    if not data:
        return control
    _replace_well_name(data, 0)
    _to_double(data, numeric)

    change_well = all(_is_number(v) and v < 0
                      for row in data for v in row[2:])

    if change_well:
        for kw, status_col in (('WCONINJ', 2), ('WCONINJE', 2),
                               ('WCONINJH', 2), ('WCONPROD', 1),
                               ('WCONHIST', 1)):
            rows = control.get(kw)
            if not rows or not data:
                continue
            status = {row[0]: row[1] for row in data}
            used = set()
            for row in rows:
                if row[0] in status:
                    row[status_col] = status[row[0]]
                    used.add(row[0])
            data = [row for row in data if row[0] not in used]
        if data:
            import warnings
            warnings.warn('Unused WELOPEN Specifications for Well%s\n%s'
                          % ('s' if len(data) != 1 else '',
                             ''.join('  * %s\n' % row[0] for row in data)),
                          RuntimeWarning)
        return control

    compdat = control.get('COMPDAT') or []
    for row in data:
        rec = [c for c in compdat if c[0] == row[0]]
        active = [True] * len(rec)
        for i, col in enumerate((1, 2, 3)):
            limit = row[col + 1]
            for k, conn in enumerate(rec):
                if not (conn[col] == limit or limit == 0):
                    active[k] = False
        low = row[5]
        high = row[6]
        if _is_number(high) and high < 0:
            high = float('inf')
        for k, conn in enumerate(rec):
            if active[k] and low <= k + 1 <= high:
                conn[5] = row[1]
    return control


def _read_simple(cursor, control, kw, target=None, mode='append',
                 well_names=False):
    """Everything whose reader is template, name-chop, to-double, store."""
    template, numeric = _TEMPLATES[kw]
    data = read_defaulted_kw(cursor, template)
    if not data:
        return control
    if well_names:
        _replace_well_name(data, 0)
    _to_double(data, numeric)

    field = target or kw
    if mode == 'assign':
        return _assign_control_records(control, data, field,
                                       _well_names(control))
    if mode == 'spec':
        control[field] = _append_spec(control.get(field), data,
                                      _well_names(control))
        return control
    if mode == 'replace':
        control[field] = data
        return control
    if mode == 'cell':
        control.setdefault(field, [])
        control[field] = list(control[field]) + [data]
        return control
    if mode == 'merge_by_name':
        # GECON/GRUPNET: a restated group replaces its earlier record.
        existing = [list(row) for row in (control.get(field) or [])]
        index = {row[0]: i for i, row in enumerate(existing)}
        fresh = []
        for row in data:
            if row[0] in index:
                existing[index[row[0]]] = row
            else:
                fresh.append(row)
        control[field] = existing + fresh
        return control
    if mode == 'merge_by_pair':
        # GCONINJE keys on (group, phase).
        existing = [list(row) for row in (control.get(field) or [])]
        index = {(row[0], row[1]): i for i, row in enumerate(existing)}
        fresh = []
        for row in data:
            key = (row[0], row[1])
            if key in index:
                existing[index[key]] = row
            else:
                fresh.append(row)
        control[field] = existing + fresh
        return control

    control[field] = list(control.get(field) or []) + data
    return control


def _read_report_control(cursor, control, kw):
    """Port of ``readReportControl``: the record kept as bare tokens."""
    text = cursor.read_record_string()
    text = re.sub(r'\s*=\s*', '=', text).replace("'", '')
    control[kw] = text.split()
    return control


#: How each supported keyword updates the control.  ``mode`` mirrors what
#: the corresponding ``readWellKW`` subfunction does with its records.
_SIMPLE = {
    'WCONHIST': dict(mode='assign', well_names=True),
    'WCONINJH': dict(mode='assign', well_names=True),
    'WCONINJE': dict(mode='assign', well_names=True),
    'WCONINJ': dict(mode='assign', well_names=True),
    'WCONPROD': dict(mode='assign', well_names=True),
    'WELTARG': dict(mode='spec', well_names=True),
    'WELCNTL': dict(mode='spec', well_names=True),
    'WTEMP': dict(mode='spec', well_names=True),
    'WSOLVENT': dict(mode='spec', well_names=True),
    'WSURFACT': dict(mode='spec', well_names=True),
    'WPOLYMER': dict(mode='spec', well_names=True),
    'WEFAC': dict(mode='replace', well_names=True),
    'WPIMULT': dict(mode='cell', well_names=True),
    'WINJGAS': dict(mode='append', well_names=True),
    'WGRUPCON': dict(mode='append', well_names=True),
    'GINJGAS': dict(mode='append'),
    'GRUPTREE': dict(mode='append'),
    'GCONPROD': dict(mode='append'),
    'GRUPNET': dict(mode='merge_by_name'),
    'GECON': dict(mode='merge_by_name'),
    'GCONINJE': dict(mode='merge_by_pair'),
}


def _read_well_kw(cursor, control, kw):
    """Port of ``readWellKW``'s dispatch."""
    kw = _ALIASES.get(kw, kw)
    if kw == 'WELSPECS':
        return _read_welspecs(cursor, control)
    if kw == 'COMPDAT':
        return _read_compdat(cursor, control)
    if kw == 'WELOPEN':
        return _read_welopen(cursor, control)
    if kw in ('RPTSCHED', 'RPTRST', 'OUTSOL'):
        return _read_report_control(cursor, control, kw)
    if kw in _SIMPLE:
        return _read_simple(cursor, control, kw, **_SIMPLE[kw])

    # In MRST's position an unsupported well keyword is a hard error.
    # Here the section is parsed for every deck the suite touches, so the
    # records are consumed and the keyword reported instead -- see the
    # module docstring.
    _skip_keyword(cursor)
    return None


def _skip_keyword(cursor):
    """Consume records up to the keyword's terminating blank record."""
    while True:
        text = cursor.read_record_string()
        if cursor.at_end:
            return
        if not text or text.isspace():
            return


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --------------------------------------- keywords outside readWellKW --

def _read_tuning(cursor, control, nlines):
    """Port of readSCHEDULE's TUNING/TUNINGDP branches.

    MRST keeps the records as raw strings and ``writeSchedule`` writes
    them straight back out, so nothing here needs to understand them.
    """
    control['TUNING' if nlines == 3 else 'TUNINGDP'] = [
        cursor.read_record_string().strip() for _ in range(nlines)]
    return control


def _read_drsdt(cursor, control):
    """Port of the DRSDT branch."""
    rec = read_defaulted_record(cursor, ['NaN', 'ALL'])
    rec = _to_double([rec], [0])[0]
    control['DRSDT'] = rec
    return control


def _read_vappars(cursor, control):
    """Port of the VAPPARS branch."""
    rec = read_defaulted_record(cursor, ['0', '0'])
    control['VAPPARS'] = _to_double([rec], [0, 1])[0]
    return control


def _read_vfpprod(cursor, control):
    """Port of ``readVFPPROD``.

    A VFP table has no terminating blank record -- its length follows from
    the axis vectors -- so it has to be read properly rather than skipped,
    or the records would run on into the following keywords.
    """
    header = read_defaulted_record(
        cursor, ['0', '0.0', 'FLO', 'WFR', 'GFR', 'THP', 'ALQ', 'USYS',
                 'BHP'])
    header = _to_double([header], [0, 1])[0]
    table = {'tid': header[0], 'depth': header[1], 'FLOID': header[2],
             'WFRID': header[3], 'GFRID': header[4], 'THPID': header[5],
             'ALQID': header[6], 'USYS': header[7], 'QID': header[8]}
    for axis in ('FLO', 'THP', 'WFR', 'GFR', 'ALQ'):
        table[axis] = cursor.read_vector()

    nflo = len(table['FLO'])
    nrec = 1
    for axis in ('THP', 'WFR', 'GFR', 'ALQ'):
        nrec *= max(len(table[axis]), 1)

    records = []
    for _ in range(nrec):
        values = cursor.read_vector()
        if len(values) < 4:                              # edited by zhang
            continue
        records.append((values[:4], values[4:4 + nflo]))
    table['Q'] = records

    control.setdefault('VFPPROD', [])
    control['VFPPROD'] = list(control['VFPPROD']) + [table]
    return control


def _read_vfpinj(cursor, control):
    """Port of ``readVFPINJ``."""
    header = read_defaulted_record(
        cursor, ['0', '0.0', 'OIL', 'THP', 'USYS', 'BHP'])
    header = _to_double([header], [0, 1])[0]
    table = {'tid': header[0], 'depth': header[1], 'FLOID': header[2],
             'THPID': header[3], 'USYS': header[4], 'BHPID': header[5]}
    table['FLO'] = cursor.read_vector()
    table['THP'] = cursor.read_vector()

    nflo = len(table['FLO'])
    records = []
    for _ in range(len(table['THP'])):
        values = cursor.read_vector()
        if not values:
            continue
        records.append((values[0], values[1:1 + nflo]))
    table['BHP'] = records

    control.setdefault('VFPINJ', [])
    control['VFPINJ'] = list(control['VFPINJ']) + [table]
    return control


#: Keywords outside ``readWellKW`` that nevertheless open a control, and
#: the reader each uses.  Leaving one out does not merely lose its data:
#: it shifts every later control index, because MRST counts a control for
#: each of these too.
_OTHER_CONTROL_READERS = {
    'TUNING': lambda c, k: _read_tuning(c, k, 3),
    'TUNINGDP': lambda c, k: _read_tuning(c, k, 1),
    'DRSDT': _read_drsdt,
    'VAPPARS': _read_vappars,
    'VFPPROD': _read_vfpprod,
    'VFPINJ': _read_vfpinj,
}


# --------------------------------------------------------- time steps --

def parse_eclipse_date(text):
    """Parse ``01 AUG 2025`` (with an optional ``HH:MM:SS``) to a float
    day number, the way MATLAB's ``datenum`` does for MRST."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    tokens = str(text).replace("'", ' ').replace('/', ' ').split()
    if len(tokens) < 3:
        return None
    try:
        day = int(float(tokens[0]))
        month = _MONTHS[tokens[1][:3].upper()]
        year = int(float(tokens[2]))
    except (KeyError, ValueError):
        return None

    value = float(_date(year, month, day).toordinal())
    if len(tokens) > 3 and ':' in tokens[3]:
        parts = tokens[3].split(':')
        seconds = 0.0
        for i, part in enumerate(parts[:3]):
            seconds += float(part) * (3600.0, 60.0, 1.0)[i]
        value += seconds / 86400.0
    return value


def _read_dates(cursor, start, elapsed):
    """Port of ``readDATES``: absolute dates become step lengths in days."""
    dates = []
    while True:
        text = _remove_quotes(cursor.read_record_string()).strip()
        if not text:
            break
        dates.append(text)
        if cursor.at_end:
            break
    if not dates:
        return []

    values = [parse_eclipse_date(d) for d in dates]
    if any(v is None for v in values):
        bad = [d for d, v in zip(dates, values) if v is None]
        raise ValueError('Unparseable DATES record(s): %s' % ', '.join(bad))
    if start is None:
        raise ValueError('DATES in SCHEDULE but no START in RUNSPEC')

    previous = float(start) + float(elapsed)
    out = []
    for value in values:
        out.append(value - previous)
        previous = value
    if any(v <= 0 for v in out):
        raise ValueError('DATES record does not advance the simulation clock')
    return out


# ------------------------------------------------------------- driver --

def read_schedule_control(text, start=None):
    """Split a SCHEDULE section into MRST's ``control``/``step`` structure.

    Parameters
    ----------
    text : str
        The SCHEDULE section, INCLUDEs already expanded.
    start : str or float, optional
        ``RUNSPEC.START``.  Needed only if the deck uses DATES rather
        than TSTEP.

    Returns
    -------
    control : list of dict
        One entry per control, in deck order.
    step : dict
        ``{'val': [days], 'control': [index]}``.  The indices are 0-based
        -- MRST's are 1-based -- to match how ``write_schedule`` and the
        rest of PRSTCore address controls.  A deck carrying SKIPRESTART
        adds ``'SKIPRESTART': True`` here; the caller lifts it onto the
        SCHEDULE itself, which is where MRST keeps it and where
        ``writeSchedule`` looks for it.
    missing : list of str
        Keywords seen in the section but not interpreted.
    """
    start = parse_eclipse_date(start) if isinstance(start, str) else start

    cursor = _Cursor(str(text).splitlines())
    control = []
    step_val = []
    step_control = []
    missing = []

    ctrl = default_control()
    def_ctrl = False
    cno = -1                     # MRST starts at 0 and writes 1-based.
    skiprestart = False

    kw = cursor.get_keyword()
    while kw is not None:
        if kw == 'SCHEDULE':
            # The section keyword itself; MRST has already consumed it.
            kw = cursor.get_keyword()
            continue

        if kw in CONTROL_KEYWORDS or kw in _OTHER_CONTROL_READERS:
            if not def_ctrl:
                def_ctrl = True
                cno += 1
                ctrl = default_control(ctrl)
            if kw in _OTHER_CONTROL_READERS:
                ctrl = _OTHER_CONTROL_READERS[kw](cursor, ctrl)
            else:
                updated = _read_well_kw(cursor, ctrl, kw)
                if updated is None:
                    missing.append(kw)
                else:
                    ctrl = updated

        elif kw in ('DATES', 'TSTEP'):
            if def_ctrl:
                def_ctrl = False
                control.append(ctrl)
            if kw == 'DATES':
                data = _read_dates(cursor, start, sum(step_val))
            else:
                data = cursor.read_vector()
            step_control.extend([cno] * len(data))
            step_val.extend(data)

        elif kw == 'SKIPRESTART':
            # MRST-0 sets a flag on the schedule, not on a control.
            skiprestart = True

        elif kw == 'END':
            break

        elif kw in ('ECHO', 'NOECHO'):
            pass

        else:
            missing.append(kw)

        kw = cursor.get_keyword()

    if def_ctrl:
        control.append(ctrl)

    step = {'val': step_val, 'control': step_control}
    if skiprestart:
        step['SKIPRESTART'] = True
    return control, step, missing
