"""Port of MRST ``writeSchedule.m`` (model-io/deckformat/deckoutput).

Ports MRST-0's version, the one ``writeDeck`` in the same directory
calls -- see :mod:`write_deck` for why that tree rather than 2026a's.
It covers far more of the SCHEDULE section than the 2026a version:
besides the well keywords it writes group control (GRUPTREE, GINJGAS),
per-well injection composition (WINJGAS, WPOLYMER, WSOLVENT, WSURFACT,
WTEMP), retargeting (WELTARG, WELCNTL), connection multipliers
(WPIMULT), solver TUNING, and BOX/ENDBOX array edits.

**Per-column formats.** ECLIPSE writes an omitted entry as ``1*``, which
is text where the column is otherwise a number. So a row is not written
with one fixed format: three passes over it -- :func:`replace_default`,
:func:`replace_nan`, :func:`replace_negative` -- each return a mask of
the columns they turned into ``1*``, and those columns switch to the
string format for that row only. Two rows of the same keyword can
therefore have different formats, which is exactly what the MATLAB does.
"""

import os as _os

import numpy as _np

#: MRST-0's default formats. Note the three string widths -- which one a
#: column uses depends on what it holds.
DEFAULT_FORMATS = {
    'case': str.upper,
    'int': '%4d',
    'string': '%8s',
    'string4': '%4s',
    'string6': '%6s',
    'string8': '%8s',
    'double': '%12.4f',
    'sci': '%12.3e',
    'doubleRange': (0.1, 100.0),
}

# Per-keyword column formats, as MRST-0 lays them out. 'i' int, 'd'
# double, 'e' sci, 's4'/'s6'/'s8' the three string widths.
_SPECS = {
    'WELSPECS': ('s8', 's8', 'i', 'i', 'd', 's6', 'd', 's4', 's6', 's4',
                 'i', 's4', 'i'),
    'COMPDAT': ('s8', 'i', 'i', 'i', 'i', 's6', 'i', 'd', 'd', 'd', 'd',
                'd', 's4', 'd'),
    'WPIMULT': ('s8', 'd', 'i', 'i', 'i', 'i', 'i'),
    'WCONINJE': ('s8', 's6', 's6', 's6', 'd', 'd', 'd', 'd', 'i', 'd', 'd',
                 'd', 'd', 'd'),
    'WCONINJH': ('s8', 's6', 's6', 'd', 'd', 'd', 'i', 'd', 'd', 'd', 'd'),
    'WCONPROD': ('s8', 's6', 's6', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'i',
                 'i'),
    'WCONHIST': ('s8', 's6', 's6', 'd', 'd', 'd', 'i', 'i', 'd', 'd'),
    'WELTARG': ('s8', 's6', 'd'),
    'WELCNTL': ('s8', 's6', 'd'),
    'WEFAC': ('s8', 'd', 's4'),
    'WTEMP': ('s8', 'd'),
    'WINJGAS': ('s8', 's6', 's8', 'i'),
    'WPOLYMER': ('s8', 'd', 'd', 's8', 's8'),
    'WSOLVENT': ('s8', 'd'),
    'WSURFACT': ('s8', 'd'),
    'GRUPTREE': ('s8', 's8'),
    'GINJGAS': ('s8', 's8', 's8', 's8', 'i'),
    'BCPROP': ('i', 's8', 's8', 'e', 'e', 'e', 's8', 'i', 'i', 'i', 'e',
               'e', 'e', 'e', 'e', 'e'),
}

#: Which blanking passes each keyword runs, in order. Not uniform: the
#: rate keywords blank infinities (an unlimited target), while WELSPECS,
#: COMPDAT and WPIMULT blank negatives (a defaulted index or factor).
#: Applying the wrong one writes a real value where ECLIPSE expects 1*.
_PASSES = {
    'GRUPTREE': ('default',),
    'GINJGAS': ('default',),
    'WELSPECS': ('default', 'nan', 'negative'),
    'COMPDAT': ('default', 'nan', 'negative'),
    'WPIMULT': ('default', 'negative'),
    'WCONINJE': ('default', 'nan', 'inf'),
    'WCONINJH': ('default', 'nan', 'inf'),
    'WCONPROD': ('default', 'nan', 'inf'),
    'WCONHIST': ('default', 'nan', 'inf'),
    'WINJGAS': ('default', 'nan', 'inf'),
    'WPOLYMER': ('default', 'nan', 'inf'),
    'WELCNTL': ('nan',),
    'WEFAC': ('default',),
}

#: Anything not listed above gets the common three.
_DEFAULT_PASSES = ('default', 'nan', 'inf')

#: Written in this order, matching MRST-0's sequence within a control.
_ORDER = ('GRUPTREE', 'GINJGAS', 'WELSPECS', 'COMPDAT', 'WPIMULT',
          'WCONINJE', 'WCONINJH', 'WCONPROD', 'WCONHIST', 'WELTARG',
          'WELCNTL', 'WEFAC', 'WTEMP', 'WINJGAS', 'WPOLYMER', 'WSOLVENT',
          'WSURFACT', 'BCPROP')

#: Keywords `fields` can select, and the two that default to off/on.
_SELECTABLE = _ORDER + ('TSTEP', 'BOX')


def writeSchedule(fn, dirname, SCHEDULE, writeInclude=False, includeName='',
                  onlyWells=False, fields=(), formats=None, start=None,
                  writeWEFAC=False, writeWPIMULT=True):
    """Write SCHEDULE to an open handle or a named file.

    ``fn`` may be an open file object or a file name relative to
    ``dirname``; naming a file forces ``writeInclude`` off, as in the
    MATLAB. ``fields`` restricts output to the named keywords and implies
    ``onlyWells``.

    ``start`` switches the time output from relative TSTEP steps to
    absolute DATES records counted from that start date. writeDeck passes
    ``deck.RUNSPEC.START``, so a written deck keeps the same calendar as
    the one it came from.
    """
    f = dict(DEFAULT_FORMATS if formats is None else formats)
    fncase = f.get('case', str.upper)

    write = {k: True for k in _SELECTABLE}
    write['WEFAC'] = writeWEFAC
    write['WPIMULT'] = writeWPIMULT
    if fields:
        onlyWells = True
        write = {k: k in fields for k in _SELECTABLE}

    doCloseFile = False
    handle = fn
    if isinstance(fn, str):
        doCloseFile = True
        handle = open(_os.path.join(dirname, fn), 'w')
        writeInclude = False

    if writeInclude:
        includeName = includeName or fncase('schedule.inc')
        inc = open(_os.path.join(dirname, includeName), 'w')
    else:
        inc = handle

    try:
        if 'RPTSCHED' in SCHEDULE and not onlyWells:
            handle.write('RPTSCHED\nPRES SWAT SOIL SGAS RS WELLS\n/\n')
            handle.write('RPTRST\nBASIC=1\n/\n')
        if SCHEDULE.get('SKIPRESTART'):
            handle.write('SKIPRESTART\n\n')

        controls = SCHEDULE.get('control')
        if controls is None:
            return
        if writeInclude:
            handle.write("INCLUDE\n'%s'\n/\n" % includeName)

        step = SCHEDULE.get('step', {})
        for cstep, control in enumerate(controls):
            _tuning(inc, control, 'TUNING', 3)
            _tuning(inc, control, 'TUNINGDP', 1)
            for keyword in _ORDER:
                if write.get(keyword, True):
                    _block(inc, control, keyword, f)
            if write['BOX']:
                _box(inc, control, f)
            if write['TSTEP']:
                _time(inc, step, cstep, f, start)
    finally:
        if writeInclude:
            inc.close()
        if doCloseFile:
            handle.close()


# ------------------------------------------------------------- blocks --

def _block(out, control, keyword, f):
    """Write one keyword's rows, each with its own per-column formats."""
    rows = _get(control, keyword)
    if not rows:
        return
    spec = _SPECS.get(keyword)
    if spec is None:
        return

    passes = _PASSES.get(keyword, _DEFAULT_PASSES)
    out.write('%s\n' % keyword.upper())
    for row in rows:
        cells = list(row)[:len(spec)]
        if not cells:
            continue
        out.write(_row(cells, spec, f, passes) + '/\n')
    out.write('/\n\n')


def _tuning(out, control, keyword, nlines):
    """TUNING/TUNINGDP are stored as ready-made lines, not fields."""
    lines = _get(control, keyword)
    if not lines:
        return
    out.write('%s\n' % keyword.upper())
    for line in list(lines)[:nlines]:
        out.write('%s /\n' % line)
    out.write('\n')


def _box(out, control, f):
    """Port of the BOX branch: run-length encoded array edits.

    The values are written as ECLIPSE repeat counts (``12*0.3``) rather
    than one per cell, which is what makes a box edit over a large region
    readable at all.
    """
    boxes = _get(control, 'BOX')
    if not boxes:
        return
    for entry in boxes:
        out.write('BOX\n')
        out.write(''.join((f['int'] % int(v)) + ' '
                          for v in _get(entry, 'box')[:6]))
        out.write('/\n')
        out.write('%s\n' % str(_get(entry, 'name')).upper())

        values = _np.atleast_1d(_np.asarray(_get(entry, 'values'))).ravel()
        counts, vals = _runs(values)
        for n, (count, value) in enumerate(zip(counts, vals), start=1):
            out.write(f['string8'] % ('%g*%g' % (count, value)))
            if n % 6 == 0:
                out.write('\n')
        out.write(' /\n')
        out.write('ENDBOX\n\n')


def _time(out, step, cstep, f, start):
    """TSTEP, or DATES when a start date is known."""
    if not step:
        return
    ctrl = _np.asarray(step.get('control', []), dtype=int).ravel()
    val = _np.asarray(step.get('val', []), dtype=float).ravel()

    if start is None:
        sel = val[ctrl == cstep] if ctrl.size else val
        if sel.size:
            out.write('TSTEP\n')
            for v in sel:
                out.write((f['double'] % v) + '\n')
            out.write('/\n\n')
        return

    ind = _np.flatnonzero(ctrl == cstep) if ctrl.size else _np.arange(val.size)
    for s in ind:
        # Elapsed time to the end of this step, counted from the very
        # first one -- not from this control's own start.
        out.write('DATES\n%s /\n/\n\n'
                  % _date(start, float(_np.sum(val[:s + 1]))))


# ------------------------------------------------------------ formats --

_PASS_FUNCTIONS = {}          # filled in below, once the passes exist


def _row(cells, spec, f, passes=_DEFAULT_PASSES):
    """Format one row, switching a column to text where it is defaulted.

    Runs the blanking passes this keyword uses, in order; each marks the
    columns it turned into ``1*`` so they print as text rather than as a
    number.
    """
    cells = list(cells)
    text = [False] * len(cells)
    for name in passes:
        cells, ix = _PASS_FUNCTIONS[name](cells)
        text = [a or b for a, b in zip(text, ix)]

    out = []
    for value, kind, as_text in zip(cells, spec, text):
        if as_text or isinstance(value, str):
            out.append(f['string4' if as_text else _string_key(kind, f)]
                       % str(value))
        elif kind == 'i':
            out.append(f['int'] % int(value))
        elif kind == 'e':
            out.append(f['sci'] % float(value))
        else:
            out.append(f['double'] % float(value))
    # ``getFmtStr`` appends a space to *every* per-value format before
    # concatenating them.  Without it two adjacent values that each fill
    # their field width run together -- '7576.760000' and '63566.450000'
    # in a %12.6f column become one 24-character number.
    return ''.join(cell + ' ' for cell in out)


def _string_key(kind, f):
    key = 'string%s' % kind[1:] if kind.startswith('s') else 'string'
    return key if key in f else 'string'


def replace_default(row):
    """Port of ``replace_default``: 'Default' becomes ``1*``.

    Whitespace is stripped from every text cell first, so a padded
    ``' Default '`` is recognised too.
    """
    out, mask = [], []
    for value in row:
        if isinstance(value, str):
            stripped = ''.join(value.split())
            if stripped.lower() == 'default':
                stripped = '1*'
            out.append(stripped)
            mask.append(stripped == '1*')
        else:
            out.append(value)
            mask.append(False)
    return out, mask


def replace_nan(row):
    """Port of ``replace_nan``: NaN becomes ``1*``."""
    out, mask = [], []
    for value in row:
        bad = (not isinstance(value, str) and value is not None
               and _isnan(value))
        out.append('1*' if bad else value)
        mask.append(bad)
    return out, mask


def replace_negative(row):
    """Port of ``replace_negative``: a negative number becomes ``1*``.

    A negative entry in these records means 'defaulted' rather than a
    real value, so writing the number through would change the meaning.
    """
    out, mask = [], []
    for value in row:
        bad = (not isinstance(value, str) and value is not None
               and _isnumber(value) and float(value) < 0)
        out.append('1*' if bad else value)
        mask.append(bad)
    return out, mask


def replace_inf(row):
    """Port of ``replace_inf``: infinity becomes ``1*``."""
    out, mask = [], []
    for value in row:
        bad = (not isinstance(value, str) and value is not None
               and _isnumber(value) and _np.isinf(float(value)))
        out.append('1*' if bad else value)
        mask.append(bad)
    return out, mask


_PASS_FUNCTIONS.update(default=replace_default, nan=replace_nan,
                       negative=replace_negative, inf=replace_inf)


# ------------------------------------------------------------ support --

def _runs(values):
    """Run-length encode: counts and the value each run holds."""
    if values.size == 0:
        return [], []
    starts = _np.flatnonzero(_np.concatenate([[True], _np.diff(values) != 0]))
    counts = _np.diff(_np.concatenate([starts, [values.size]]))
    return list(counts), list(values[starts])


def _base_date(start):
    """RUNSPEC.START in whichever form the deck carried it.

    MRST's readRUNSPEC stores a ``datenum``; PRSTCore's read_runspec keeps
    the record verbatim (``'01 AUG 2025'``), and a caller may hand over a
    ``date`` or a ``[day, month, year]`` record instead. All four resolve
    here so the writer does not care which reader produced the deck.
    """
    import datetime as _dt

    if isinstance(start, _dt.datetime):
        return start.date()
    if isinstance(start, _dt.date):
        return start
    if isinstance(start, str):
        from PRSTCore.deckformat.deckinput.schedule_control import \
            parse_eclipse_date
        ordinal = parse_eclipse_date(start)
        if ordinal is None:
            raise ValueError('Unparseable START record: %r' % start)
        return _dt.date.fromordinal(int(ordinal))

    values = _np.atleast_1d(_np.asarray(start, dtype=float)).ravel()
    if values.size >= 3:
        return _dt.date(int(values[2]), int(values[1]), int(values[0]))
    # A single number is a day count, as MATLAB's datenum is.
    return _dt.date.fromordinal(int(values[0]))


def _date(start, elapsed_days):
    """``DD MON YYYY`` for ``start`` plus that many days."""
    import datetime as _dt

    when = _base_date(start) + _dt.timedelta(days=float(elapsed_days))
    return '%2d %s %d' % (when.day, when.strftime('%b').upper(), when.year)


def _isnan(value):
    try:
        return bool(_np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _isnumber(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _get(container, key):
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)
