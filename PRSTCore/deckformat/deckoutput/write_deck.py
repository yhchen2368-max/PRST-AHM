"""Port of MRST ``writeDeck.m`` (model-io/deckformat/deckoutput).

**Which writeDeck.** Two versions exist in the trees here. mrst-2026a
ships a 617-line one whose only option is ``unit``; MRST-0 carries a
1131-line one with ``filename`` and ``NOSIM``. Every hm caller --
evaluateMatchFromEclipseRun, evaluateObjectiveFromEclipseRun, FAHM_M and
HistoryMatching -- passes ``filename``, and two pass ``NOSIM``, so the
2026a version cannot serve the module that calls it. This ports the
MRST-0 version, which is the one hm was written against.

The MRST-0 version also **generates its own SUMMARY section** (SUM.INC:
the standard field and well vectors, each also in its ``*H`` history
form) rather than echoing whatever the reader captured. That matters for
history matching: a deck round-tripped through the 2026a writer loses
every summary vector, including the ``*H`` observations a match is scored
against, whereas this one writes a complete set back out.

Bulk arrays go to ``.INC`` files beside the ``.DATA``, as MRST-0 does
(the 2026a version used ``.TXT``).
"""

import os as _os

import numpy as _np

from .write_schedule import DEFAULT_FORMATS, writeSchedule

_DASHED = '-' * 57

_HEADER = """\
---------------------------------------------------------
---                                                   ---
---           M   M   RRRR     SSSS   TTTTT           ---
---           MM MM   R   R   S         T             ---
---           M M M   RRRR     SSS      T             ---
---           M   M   R R         S     T             ---
---           M   M   R  RR   SSSS      T             ---
---                                                   ---
--------------------------------- www.sintef.no/mrst  ---

-- Generated deck from MRST function writeDeck

"""

#: MRST-0's formats. Narrower than 2026a's, and six values per line.
MRST0_FORMATS = dict(DEFAULT_FORMATS)
MRST0_FORMATS.update(int='%6d', string='%8s', double='%12.6f', sci='%12.6e')

_PER_LINE = 6

_RUNSPEC_DIMS = ('DIMENS', 'EQLDIMS', 'FAULTDIM', 'REGDIMS', 'AQUDIMS')
_RUNSPEC_FLAGS = ('OIL', 'WATER', 'GAS', 'DISGAS', 'VAPOIL', 'CO2STOR',
                  'METRIC', 'NOGRAV', 'FIELD', 'NONNC', 'TEMP', 'THERMAL',
                  'MECH')

_GRID_ARRAYS = ('PERMX', 'PERMY', 'PERMZ', 'PORO', 'ACTNUM', 'NTG', 'MULTPV',
                'MULTX', 'MULTY', 'MULTZ', 'PRATIO', 'YMODULE', 'BIOTCOEF',
                'POELCOEF', 'THELCOEF', 'THERMEXR', 'THCONR')

_EDIT_ARRAYS = ('DEPTH', 'PORV', 'TRANX', 'TRANY', 'TRANZ')

#: MESSAGES limits, verbatim from MRST-0.
_MESSAGES = (1000000, 1000000, 1000000, 50000, 50000, 50000,
             1000000, 1000000, 1000000, 50000, 5000, 5000)

#: The per-well and per-field quantities SUM.INC requests.
_SUMMARY_VARS = ('WPR', 'WPT', 'OPR', 'OPT', 'GPR', 'GPT', 'LPR', 'LPT',
                 'WIR', 'WIT', 'OIR', 'OIT', 'GIR', 'GIT', 'WCT', 'GOR')


def write_deck(deck, directory, filename=None, unit='metric', formats=None,
               NOSIM=False):
    """Write ``deck`` into ``directory``. Returns the path written.

    ``filename`` names the ``.DATA`` file; without it the file is named
    after the directory. ``NOSIM`` adds the NOSIM keyword so the
    simulator initialises and writes INIT/EGRID/UNRST without running the
    schedule -- how ``HistoryMatching`` builds G, rock and state0 from
    the simulator's own output.
    """
    from PRSTCore.deckformat.deckinput.convert_deck_units import \
        convert_deck_units

    f = dict(MRST0_FORMATS if formats is None else formats)
    deck = convert_deck_units(deck, output_unit=unit)

    _os.makedirs(directory, exist_ok=True)
    name = filename or _os.path.basename(_os.path.abspath(directory))
    path = _os.path.join(directory, '%s.DATA' % name.upper())

    runspec = deck.get('RUNSPEC') or {}
    props = deck.get('PROPS') or {}

    with open(path, 'w') as fid:
        fid.write(_HEADER)

        _section(fid, 'RUNSPEC')
        if 'TITLE' in runspec:
            fid.write('TITLE\n%s\n\n' % runspec['TITLE'])
        if 'CPR' in ((deck.get('UnhandledKeywords') or {}).get('RUNSPEC')
                     or ()):
            fid.write('CPR\n/\n\n')
        if NOSIM:
            fid.write('NOSIM\n\n')
        _dump_runspec(fid, directory, deck, f)

        _section(fid, 'GRID')
        fid.write('INIT\n\n')
        fid.write('GRIDFILE\n2 1 \n/\n\n')
        _dump_grid(fid, directory, deck, f)

        _section(fid, 'EDIT')
        _dump_edit(fid, directory, deck, f)

        _section(fid, 'PROPS')
        if 'ROCK' in props:
            # Trimmed to two columns for OPM/E300 compatibility.
            rock = _np.atleast_2d(_np.asarray(props['ROCK'], dtype=float))
            props = dict(props)
            props['ROCK'] = rock[:, :2]
        _dump_tables(fid, directory, props, f)

        _section(fid, 'REGIONS')
        _dump_arrays(fid, directory, deck.get('REGIONS') or {},
                     sorted(deck.get('REGIONS') or {}), f, 'int')

        _section(fid, 'SOLUTION')
        _dump_solution(fid, directory, deck, f)

        _section(fid, 'SUMMARY')
        _dump_summary(fid, directory, deck)

        _section(fid, 'SCHEDULE')
        writeSchedule(fid, directory, deck.get('SCHEDULE') or {},
                      writeInclude=True, includeName='SCHEDULE.INC',
                      formats=f, start=runspec.get('START'))
    return path


# ------------------------------------------------------------ sections --

def _dump_runspec(fid, dirname, deck, f):
    runspec = deck.get('RUNSPEC') or {}

    for field in _RUNSPEC_DIMS:
        if field in runspec:
            _dump_vector(fid, dirname, field, runspec[field], f, 'int',
                         newFile=False)
    if 'TABDIMS' in runspec:
        v = _np.atleast_1d(_np.asarray(runspec['TABDIMS'],
                                       dtype=float)).ravel()[:13]
        v = v.copy()
        if v.size >= 10 and not _np.isfinite(v[9]):
            v[9] = 1                      # MRST-0 substitutes 1 for NaN here
        _dump_vector(fid, dirname, 'TABDIMS', v, f, 'int', newFile=False,
                     per_line=len(v))
    if 'WELLDIMS' in runspec:
        v = _np.atleast_1d(_np.asarray(runspec['WELLDIMS'])).ravel()[:4]
        _dump_vector(fid, dirname, 'WELLDIMS', v, f, 'int', newFile=False,
                     per_line=len(v))

    for flag in _RUNSPEC_FLAGS:
        if runspec.get(flag) == 1 or runspec.get(flag) is True:
            fid.write('%s\n\n' % flag)

    if 'GRIDOPTS' in runspec:
        _dump_mixed(fid, 'GRIDOPTS', runspec['GRIDOPTS'][:3], ('s', 'i', 'i'),
                    f)
    if 'PARALLEL' in runspec:
        _dump_mixed(fid, 'PARALLEL', runspec['PARALLEL'][:2], ('i', 's'), f)
    if 'ENDSCALE' in runspec:
        _dump_mixed(fid, 'ENDSCALE', runspec['ENDSCALE'][:4],
                    ('s', 's', 'i', 'i'), f)
    if 'START' in runspec:
        fid.write('START \n %s\n/\n\n' % _start_date(runspec['START']))
    if 'ROCKCOMP' in runspec:
        _dump_mixed(fid, 'ROCKCOMP', runspec['ROCKCOMP'][:3], ('s', 'i', 's'),
                    f)

    fid.write('UNIFOUT\n\n')
    _dump_vector(fid, dirname, 'MESSAGES', _MESSAGES, f, 'int',
                 newFile=False, per_line=len(_MESSAGES))
    _dump_vector(fid, dirname, 'SMRYDIMS', [10000000], f, 'int', per_line=1,
                 newFile=False)


def _dump_grid(fid, dirname, deck, f):
    grid = deck.get('GRID', {}) or {}
    dims = grid.get('cartDims') or (deck.get('RUNSPEC') or {}).get('cartDims')

    if 'COORD' in grid and 'ZCORN' in grid:
        if dims is not None:
            fid.write('SPECGRID\n%s 1 F\n/\n\n'
                      % ' '.join('%d' % d for d in dims))
        _dump_vector(fid, dirname, 'COORD', grid['COORD'], f, 'double')
        _dump_vector(fid, dirname, 'ZCORN', grid['ZCORN'], f, 'double')
    elif all(k in grid for k in ('DXV', 'DYV', 'DZV')):
        for k in ('DXV', 'DYV', 'DZV', 'TOPS'):
            if k in grid:
                _dump_vector(fid, dirname, k, grid[k], f, 'sci')
    elif all(k in grid for k in ('DX', 'DY', 'DZ')):
        if dims is not None:
            fid.write('SPECGRID\n%s 1 F\n/\n\n'
                      % ' '.join('%d' % d for d in dims))
        for k in ('DX', 'DY', 'DZ', 'TOPS'):
            if k in grid:
                _dump_vector(fid, dirname, k, grid[k], f, 'sci')

    for k in _GRID_ARRAYS:
        if k in grid:
            _dump_vector(fid, dirname, k, grid[k], f,
                         'int' if k == 'ACTNUM' else 'double')


def _dump_edit(fid, dirname, deck, f):
    """Port of ``dump_edit``.

    An EDIT section with nothing in it still gets an empty PORV.INC and
    an INCLUDE for it, so the deck's structure does not change depending
    on whether the section happened to be populated.
    """
    edit = deck.get('EDIT') or {}
    if not edit:
        open(_os.path.join(dirname, 'PORV.INC'), 'w').close()
        fid.write("INCLUDE\n'PORV.INC'\n/\n\n")
        return
    for k in _EDIT_ARRAYS:
        if k in edit:
            _dump_vector(fid, dirname, k, edit[k], f, 'double')


def _dump_solution(fid, dirname, deck, f):
    """Port of ``dump_solution``, with the restart-output requests.

    RPTSOL/RPTRST are dropped from the deck and rewritten, so the output
    always carries the RESTART=2 / BASIC=2 requests the workflow needs to
    read states back -- and PCOW as well when SWATINIT is present, since
    the capillary scaling cannot be recovered without it.
    """
    solution = dict(deck.get('SOLUTION') or {})
    solution.pop('RPTSOL', None)
    solution.pop('RPTRST', None)

    runspec = deck.get('RUNSPEC') or {}
    props = deck.get('PROPS') or {}

    if 'RESTART' in solution:
        _dump_mixed(fid, 'RESTART', list(solution.pop('RESTART'))[:4],
                    ('s', 'i', 's', 's'), f)

    v = [' RESTART=2']
    if 'COMPS' in runspec:
        v += ['TEMP', 'XMF', 'YMF', 'ZMF']
    fid.write('RPTSOL\n%s /\n\n' % ' '.join(v))

    v = [' BASIC=2 SWAT SOIL SGAS PRESSURE']
    if 'SWATINIT' in props:
        v += ['PCOW']
    if 'COMPS' in runspec:
        v += ['TEMP', 'XMF', 'YMF', 'ZMF']
    fid.write('RPTRST\n%s /\n\n' % ' '.join(v))

    # ``EQUIL`` is written explicitly: nine columns, the first six as
    # reals and the last three as integers, because those three are table
    # numbers. A deck may state fewer and let ECLIPSE default the rest --
    # QIEDIE writes seven and terminates with ``/`` -- but the writer has
    # to emit the full record: a short one comes back as "CANNOT READ
    # EQUILIBRATION DATA FOR REGION 1".
    if 'EQUIL' in solution:
        _dump_equil(fid, solution.pop('EQUIL'), f)

    _dump_tables(fid, dirname, solution, f)


#: ``EQUIL``'s nine columns and the defaults ECLIPSE applies to the ones
#: a deck omits: the three table numbers default to zero, and the
#: capillary-transition depths to the contact depths already given.
_EQUIL_WIDTH = 9


def _dump_equil(fid, values, f):
    """Port of ``writeDeck``'s EQUIL branch."""
    rows = _np.atleast_2d(_np.asarray(values, dtype=float))
    lines = ['EQUIL']
    for row in rows:
        row = _np.where(_np.isfinite(row), row, 0.0)
        if row.size < _EQUIL_WIDTH:
            row = _np.concatenate([row, _np.zeros(_EQUIL_WIDTH - row.size)])
        row = row[:_EQUIL_WIDTH]
        cells = [f['double'] % v for v in row[:6]]
        cells += [f['int'] % int(v) for v in row[6:9]]
        lines.append(' '.join(cells) + ' /')
    fid.write('\n'.join(lines) + '\n\n')


def _dump_summary(fid, directory, deck):
    """Port of the SUMMARY block: write SUM.INC and include it.

    The vector list is generated rather than read from the deck, so a
    written deck always requests the full set -- both the simulated
    vectors and their ``*H`` history counterparts.
    """
    fid.write("INCLUDE\n'SUM.INC'\n/\n\n")

    lines = []
    for kw in ('RPTONLY', 'DATE', 'RUNSUM', 'TCPU', 'NEWTON'):
        lines.append('%s\n' % kw)

    field = ['F' + v for v in _SUMMARY_VARS + ('PR',)]
    well = ['W' + v for v in _SUMMARY_VARS + ('BHP', 'THP')]
    flds = field + well
    flds = flds + [k + 'H' for k in flds]
    for kw in flds:
        # A well vector needs a well-name record; a field vector does not.
        lines.append('%s\n/\n' % kw if kw[0].upper() == 'W' else '%s\n' % kw)

    for kw in ('CPR', 'CDRD'):
        lines.append("%s\n'*' /\n/\n" % kw)
    for kw in ('WSTAT', 'WMCTL'):
        lines.append('%s\n/\n' % kw)

    if 'COMPS' in (deck.get('RUNSPEC') or {}):
        for prefix in 'FGW':
            for kw in ('XMF', 'YMF', 'ZMF'):
                name = prefix + kw
                lines.append('%s\n/\n' % name if prefix in 'GRSW'
                             else '%s\n' % name)

    with open(_os.path.join(directory, 'SUM.INC'), 'w') as out:
        out.write('\n'.join(lines) + '\n')


def _scaler_names():
    """``classifyPropsKeywords``'s ``'scalers'`` case, expanded.

    Every saturation and relative-permeability endpoint that ECLIPSE
    takes as one value per cell, in each of the four families the MATLAB
    lists literally: the drainage endpoint, its per-direction variants
    (``X``/``Y``/``Z`` and their ``-`` reverse forms), and the imbibition
    ``I`` prefix on both.
    """
    names = set()

    def family(base, directional=True):
        forms = [base]
        if directional:
            for axis in 'XYZ':
                forms += [base + axis, base + axis + '-']
        for form in list(forms):
            names.add(form)
            names.add('I' + form)

    for base in ('SWL', 'SWCR', 'SWU', 'SGL', 'SGCR', 'SGU',
                 'SOL', 'SOCR', 'SOU',
                 'KRW', 'KRG', 'KRO'):
        family(base)
    # The residual-endpoint permeabilities put the axis *inside* the name
    # (KRWXR) but keep the reverse form outside it (KRWRX-).
    for base in ('KRWR', 'KRGR'):
        names.update({base, 'I' + base})
        for axis in 'XYZ':
            names.update({base[:-1] + axis + 'R', base + axis + '-',
                          'I' + base[:-1] + axis + 'R', 'I' + base + axis + '-'})
    for base in ('KRORW', 'KRORG'):
        family(base)
    names.update({'SWLPC', 'ISWLPC', 'SGLPC', 'ISGLPC'})
    names.update({'SOWCR', 'SOGCR', 'ISOWCR', 'ISOGCR'})
    names.update({'SWATINIT'})
    names.update({'PCW', 'IPCW', 'PCG', 'IPCG'})
    return frozenset(names)


_SCALERS = _scaler_names()


def _dump_arrays(fid, dirname, section, keys, f, kind):
    for k in keys:
        if k in section:
            _dump_vector(fid, dirname, k, section[k], f, kind)


def _dump_tables(fid, dirname, section, f):
    """Write a PROPS/SOLUTION section: one INCLUDE per keyword."""
    for key in sorted(section):
        if key.startswith('_'):
            # Internal bookkeeping such as ``_miscible_pvt_records``, not
            # a deck keyword.
            continue
        values = section[key]
        if key.upper() in _SCALERS:
            # ``dump_props``'s ``case 'scalers'``: a per-cell endpoint
            # array, written six to a line and skipped entirely if any
            # entry is non-finite.  It is *not* a table -- writing it
            # through the table path puts all 54080 values on one line,
            # which ECLIPSE rejects with "ERROR ENCOUNTERED WHILE READING
            # SECTION 1 OF KEYWORD KRO".
            arr = _np.atleast_1d(_np.asarray(values, dtype=float)).ravel()
            if _np.all(_np.isfinite(arr)):
                _dump_vector(fid, dirname, key, arr, f, 'double', per_line=6)
        elif key in ('PVTO', 'PVTG'):
            records = _miscible_records(section, key)
            _dump_pvt(fid, dirname, key, records or values, f)
        elif isinstance(values, str):
            fid.write('%s\n%s\n/\n\n' % (key, values))
        elif key in _TABLE_COLUMNS:
            _dump_multiple(fid, dirname, key, values, f,
                           ncol=_TABLE_COLUMNS[key])
        elif _is_text(values):
            # ``writeDeck``'s ``iscellstr(values)`` branch: a keyword
            # whose records are words, not numbers -- SCALECRS 'NO',
            # ENDSCALE 'NODIR' 'REVERS'. It writes them with the string
            # format instead of the scientific one, which is why the
            # commented-out SCALECRS special case above it is redundant.
            _dump_text(fid, dirname, key, values)
        else:
            _dump_multiple(fid, dirname, key, values, f)


# ------------------------------------------------------------- writers --

def _dump_vector(fid, dirname, field, values, f, kind, newFile=True,
                 per_line=None):
    """Port of ``dump_vector``, with ``getFmtStr``'s two jobs.

    ``getFmtStr(fmt, n)`` appends a space to the per-value format *and*
    repeats it ``n`` times before the newline. Both matter: without the
    space the values run together into one enormous number -- MESSAGES'
    twelve limits became the single 21-digit ``100000010000001000000``,
    which ECLIPSE rejects outright -- and without the repeat count the
    line wraps somewhere the keyword does not expect.

    MRST calls this with ``n = numel(v)`` for the small header keywords,
    so they occupy one line; a bulk array keeps the six-per-line default.
    """
    values = _np.atleast_1d(_np.asarray(values)).ravel()
    fmt = f['int'] if kind == 'int' else f.get(kind, f['double'])
    width = int(per_line or _PER_LINE)

    cells = [(fmt % (int(v) if kind == 'int' else float(v))) for v in values]
    rows = [' '.join(cells[i:i + width])
            for i in range(0, len(cells), width)]
    body = '%s\n%s\n/\n\n' % (field.upper(), '\n'.join(rows))

    if not newFile:
        fid.write(body)
        return
    name = '%s.INC' % field.upper()
    with open(_os.path.join(dirname, name), 'w') as out:
        out.write(body)
    fid.write("INCLUDE\n'%s'\n/\n\n" % name)


def _dump_multiple(fid, dirname, field, values, f, ncol=None):
    """Port of ``dump_multiple``: one ``/``-terminated record per region.

    MRST builds the format string as ``getFmtStr(f.sci, size(v{1}, 2))``
    -- the per-value format repeated *once per column* and closed with a
    newline -- so ``fprintf`` breaks the line every ``ncol`` values. The
    column count is the whole point: a saturation or PVT table is a flat
    run of numbers on disk, and only the wrap tells ECLIPSE where one
    row ends. Written as a single line it reports "NOT ENOUGH ROWS OF
    DATA ( 0)" and refuses the deck.

    NaN becomes 0, as the MATLAB does before writing.
    """
    regions = _rows(values, ncol)
    lines = [field.upper()]
    for region in regions:
        region = _np.atleast_2d(region)
        region = _np.where(_np.isfinite(region), region, 0.0)
        for row in region:
            lines.append(' '.join(f['sci'] % v for v in row))
        lines.append('/')
    name = '%s.INC' % field.upper()
    with open(_os.path.join(dirname, name), 'w') as out:
        out.write('\n'.join(lines) + '\n')
    fid.write("INCLUDE\n'%s'\n/\n\n" % name)


#: How many columns each tabular keyword has. MRST reads this off
#: ``size(values{1}, 2)`` because its reader keeps the table
#: two-dimensional; PRSTCore's flattens some of them, so the width has to
#: be stated. Getting it wrong writes a table ECLIPSE parses as a
#: different number of rows -- or, unwrapped entirely, as none.
_TABLE_COLUMNS = {
    'SWOF': 4, 'SGOF': 4, 'SLGOF': 4, 'SWFN': 3, 'SGFN': 3,
    'SOF2': 2, 'SOF3': 3,
    'PVDG': 3, 'PVDO': 3, 'PVTW': 4, 'PVCDO': 5,
    'ROCK': 2, 'DENSITY': 3, 'GRAVITY': 3,
    'RSVD': 2, 'RVVD': 2, 'PBVD': 2, 'PDVD': 2,
    'EQUIL': 11, 'AQUFETP': 8, 'AQUCT': 12,
}


def _is_text(values):
    """Whether a keyword's records are words rather than numbers.

    ``writeDeck``'s ``iscellstr(values)`` test. A deck states some
    keywords as tokens -- ``SCALECRS 'NO'``, ``ENDSCALE 'NODIR'
    'REVERS'`` -- and pushing those through the numeric formatter turns
    a valid deck into one ECLIPSE rejects.
    """
    if isinstance(values, str):
        return True
    if isinstance(values, (list, tuple)):
        flat = []
        for item in values:
            flat.extend(item if isinstance(item, (list, tuple)) else [item])
        return any(isinstance(v, str) for v in flat)
    return getattr(values, 'dtype', None) is not None \
        and values.dtype.kind in 'US'


def _dump_text(fid, dirname, field, values):
    """A token-valued keyword, written inline as MRST's string-format
    ``dump_vector`` does."""
    if isinstance(values, str):
        records = [[values]]
    else:
        records = [item if isinstance(item, (list, tuple)) else [item]
                   for item in values]
    lines = [field.upper()]
    for record in records:
        lines.append(' '.join(str(v) for v in record) + ' /')
    fid.write('\n'.join(lines) + '\n\n')


def _miscible_records(props, name):
    """PVTO/PVTG as ``readMisciblePVTTable`` leaves them.

    ``props[name]`` is the flattened array the interpolators use; the
    record structure MRST's ``dump_pvt`` writes -- one saturated key per
    block, its undersaturated rows beneath -- survives separately in
    ``props['_miscible_pvt_records']``. Reading the flat array instead
    loses the block boundaries, and there is no way to recover them from
    the numbers alone.
    """
    records = (props or {}).get('_miscible_pvt_records')
    if not isinstance(records, dict):
        return None
    regions = records.get(name.upper())
    if not regions:
        return None

    tables = []
    for region in regions:
        key, pos, rows = [], [0], []
        for record in region:
            record = _np.asarray(record, dtype=float).ravel()
            key.append(record[0])
            block = record[1:].reshape((-1, 3))
            rows.append(block)
            pos.append(pos[-1] + block.shape[0])
        tables.append({'key': _np.asarray(key, dtype=float),
                       'pos': _np.asarray(pos, dtype=int),
                       'data': _np.vstack(rows) if rows
                       else _np.zeros((0, 3))})
    return tables


def _dump_pvt(fid, dirname, name, values, f):
    """Port of ``dump_pvt``: PVTO/PVTG's key/pos/data layout."""
    name = name.upper()
    tables = values if isinstance(values, (list, tuple)) else [values]
    lines = [name]
    for table in tables:
        key = _np.atleast_1d(_np.asarray(_get(table, 'key'), dtype=float))
        pos = _np.atleast_1d(_np.asarray(_get(table, 'pos'), dtype=int))
        data = _np.atleast_2d(_np.asarray(_get(table, 'data'), dtype=float))
        assert key.size + 1 == pos.size, 'PVT key/pos length mismatch'
        for r in range(pos.size - 1):
            lines.append(f['sci'] % key[r])
            for row in data[pos[r]:pos[r + 1], :]:
                lines.append(' '.join(f['sci'] % v for v in row))
            lines.append('/')
        lines.append('/')
    filename = '%s.INC' % name
    with open(_os.path.join(dirname, filename), 'w') as out:
        out.write('\n'.join(lines) + '\n')
    fid.write("INCLUDE\n'%s'\n/\n\n" % filename)


def _dump_mixed(fid, field, values, spec, f):
    """A short record of mixed strings and numbers, written inline."""
    out = []
    for value, kind in zip(values, spec):
        if isinstance(value, str):
            out.append(f['string'] % value)
        elif value is None or not _np.isfinite(float(value)):
            out.append(f['string'] % '1*')
        elif kind == 'i':
            out.append(f['int'] % int(value))
        else:
            out.append(f['double'] % float(value))
    fid.write('%s\n%s\n/\n\n' % (field.upper(), ' '.join(out)))


# ------------------------------------------------------------ support --

def _section(fid, name):
    fid.write('%s\n%s\n%s\n' % (_DASHED, name, _DASHED))


def _rows(values, ncol=None):
    """Port of ``convertValuesToCell``: one 2-D block per region.

    A keyword the reader flattened comes back as a bare run of numbers;
    ``ncol`` restores its shape. Without it a table keyword writes as one
    very long line and ECLIPSE counts zero rows.
    """
    def shape(v):
        v = _np.asarray(v, dtype=float)
        if v.ndim >= 2:
            return v
        v = v.ravel()
        if ncol and v.size % ncol == 0:
            return v.reshape((-1, ncol))
        return v.reshape((1, -1))

    if isinstance(values, (list, tuple)):
        return [shape(v) for v in values]
    return [shape(values)]


def _start_date(value):
    """``START`` as ECLIPSE wants it: ``DD 'MON' YYYY``."""
    import datetime as _dt
    if isinstance(value, _dt.date):
        return "%d '%s' %d" % (value.day, value.strftime('%b').upper(),
                               value.year)
    values = _np.atleast_1d(_np.asarray(value)).ravel()
    if values.size >= 3:
        day, month, year = int(values[0]), int(values[1]), int(values[2])
        names = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG',
                 'SEP', 'OCT', 'NOV', 'DEC')
        return "%d '%s' %d" % (day, names[month - 1], year)
    return str(value)


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)
