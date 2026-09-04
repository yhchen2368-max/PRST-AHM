"""Port of MRST ``readTracerTest.m`` (mrst-2026a/hm/utils/observed).

Reads an interwell tracer test description -- a keyword-per-line text
format, one record per slug, terminated by a ``/`` line::

    注入井号     I1                 injector
    注入层位     1200-1250 ...      injection intervals (top-bottom)
    注剂时间     20200101           injection date
    示踪剂类型   T1                 tracer name
    示踪剂用量   500                dosage
    示踪剂观测                      breakthrough block, opened by
    日期井号     P1 P2              the producer list, then one row per
    20200201     0.1 0.2            sample: date followed by one
    20200301     0.3 0.4            concentration per producer
    /

Every keyword must appear exactly once per record; the MATLAB enforces
that with ``recordCheck``/``recordCheckSingle``, and so does this port --
a missing or duplicated field means the file is malformed, and reading it
anyway would silently produce a wrong test setup.
"""

import numpy as _np

from ._tables import parse_dates

KEYWORDS = {
    '注入井号': 'injector',
    '注入层位': 'depth',
    '注剂时间': 'date',
    '示踪剂类型': 'name',
    '示踪剂用量': 'dosage',
}
OUTPUT_KEYWORD = '示踪剂观测'
PRODUCER_KEYWORDS = ('日期井号', '日期井名')
TERMINATORS = ('/', '//', '///', '////')

_REQUIRED = ('injector', 'depth', 'date', 'name', 'dosage',
             'producer', 'output')


def readTracerTest(fn):
    """Return a list of slug records."""
    with open(str(fn), 'rt', encoding='utf-8-sig') as handle:
        lines = handle.read().splitlines()

    records = []
    current = {}
    i = 0
    while i < len(lines):
        parts = _split(lines[i])
        i += 1
        if not parts:
            continue
        head = parts[0]

        if head in KEYWORDS:
            field = KEYWORDS[head]
            _check_single(current, field)
            if field == 'depth':
                current['depth'] = _parse_depths(parts[1:])
            elif field == 'dosage':
                current['dosage'] = float(parts[1])
            elif field == 'date':
                # MATLAB stores the raw string and calls datenum at each
                # comparison site; parsing once here makes the record
                # directly comparable with the simulation dates.
                current['date'] = parse_dates([parts[1]])[0]
            else:
                current[field] = parts[1]

        elif head == OUTPUT_KEYWORD:
            _check_single(current, 'producer')
            producer, output, i = _read_output_block(lines, i)
            current['producer'] = producer
            current['output'] = output

        elif head in TERMINATORS:
            missing = [f for f in _REQUIRED if f not in current]
            if missing:
                raise ValueError(
                    'Tracer test record %d is missing: %s'
                    % (len(records) + 1, ', '.join(missing)))
            records.append(current)
            current = {}

        else:
            raise ValueError("Unsupported keyword '%s' in tracer test data file"
                             % head)

    if current:
        raise ValueError('Tracer test record is missing its / terminator')

    return records


def _read_output_block(lines, i):
    """Port of ``readOutputRecordString``.

    Consumes the producer-name line and then every sample row until a line
    that is not sample data (a terminator or the next keyword), which is
    left for the caller -- the MATLAB rewinds the file pointer for the
    same reason.
    """
    producer = None
    while i < len(lines):
        parts = _split(lines[i])
        i += 1
        if _is_comment(lines[i - 1]):
            continue
        if parts and parts[0] in PRODUCER_KEYWORDS:
            producer = parts[1:]
            break
    if not producer:
        raise ValueError('Missing producing well name of the tracer test.')

    ncol = len(producer) + 1
    output = []
    while i < len(lines):
        if _is_comment(lines[i]):
            i += 1
            continue
        parts = _split(lines[i])
        if not parts:
            i += 1
            continue
        if parts[0] in TERMINATORS or parts[0] in KEYWORDS \
                or parts[0] == OUTPUT_KEYWORD:
            break
        if len(parts) != ncol:
            raise ValueError(
                'Tracer sample row has %d entries, expected %d (one date plus '
                'one concentration per producer)' % (len(parts), ncol))
        output.append([parse_dates([parts[0]])[0]]
                      + [float(v) for v in parts[1:]])
        i += 1

    return producer, _np.asarray(output, dtype=object), i


def _parse_depths(items):
    """``'1200-1250' '1300-1360'`` -> an ``(n, 2)`` top/bottom array."""
    values = []
    for item in items:
        text = ''.join(str(item).split())
        parts = text.split('-')
        if len(parts) < 2:
            raise ValueError('Cannot read an injection interval from %r' % item)
        values.append([float(parts[0]), float(parts[1])])
    return _np.asarray(values, dtype=float).reshape(-1, 2)


def _check_single(current, field):
    """Port of ``recordCheck``: a keyword may appear once per record."""
    if field in current:
        raise ValueError("Duplicated '%s' entry in a tracer test record" % field)


def _split(line):
    """Port of ``splitString``: split on runs of whitespace."""
    return str(line).strip().split()


def _is_comment(line):
    text = str(line).lstrip()
    return text.startswith('#') or text.startswith('--')
