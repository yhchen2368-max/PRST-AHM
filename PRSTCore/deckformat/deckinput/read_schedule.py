"""Read SCHEDULE section from an ECLIPSE deck.

This parser groups SCHEDULE entries by leading keyword (e.g. WELSPECS,
TSTEP, CONTROL). It preserves order within lists and converts numeric
tokens to numbers where appropriate.

Beside those flat lists it attaches ``control`` and ``step``, MRST's own
representation of the section: one control per stretch of well keywords
between two DATES/TSTEP records, and a step vector saying which control
each report step runs under.  See :mod:`schedule_control` -- ``writeDeck``
and everything downstream of it addresses the schedule that way, and the
flat lists cannot express where one control ends and the next begins.
"""

import re
from typing import List
import numpy as np

from .schedule_control import read_schedule_control


def _try_float(tok: str):
    try:
        return float(tok.replace('D', 'E').replace('d', 'e'))
    except Exception:
        return None


def _flatten_tokens(tokens: List[str]):
    vals = []
    for t in tokens:
        for part in re.split('[,;]', t):
            if part == '' or part == '/':
                continue
            v = _try_float(part)
            if v is None:
                vals.append(part)
            else:
                vals.append(v)
    return vals


def _terminates_record(line):
    """Whether a data line closes the record it belongs to.

    Only a ``/`` outside quotes counts; a well name may legitimately
    contain one.
    """
    quoted = False
    for ch in line:
        if ch == "'":
            quoted = not quoted
        elif ch == '/' and not quoted:
            return True
    return False


def read_schedule(block, start=None):
    """Parse the SCHEDULE section.

    ``start`` is ``RUNSPEC.START``; it is needed only to turn DATES into
    step lengths, and its absence leaves ``control``/``step`` off rather
    than failing the whole read.
    """
    lines = [ln.strip() for ln in block.splitlines()]
    schedule = {}
    order = []
    current = None
    pending = []

    for line in lines:
        if not line:
            continue

        # A record runs to its ``/``, and ECLIPSE lets it wrap. While one
        # is still open no line can start a keyword -- QIEDIE writes its
        # WCONINJH records over two lines, and reading the continuation
        # (``RATE /``) as a keyword swallowed three of the four injectors
        # into a phantom ``SCHEDULE['RATE']``, leaving them shut for the
        # whole run.
        if pending:
            pending.append(line)
            if _terminates_record(line):
                schedule.setdefault(current, []).append(
                    ' '.join(pending).split())
                pending = []
            continue

        parts = line.split()
        if not parts:
            continue

        # An empty record -- a lone ``/`` -- closes a multi-record keyword
        # block.  This is the block terminator of the ECLIPSE input
        # format, the one MRST's ``readDefaultedKW`` stops on (its record
        # reader returns the all-default template for such a record);
        # WELSPECS, COMPDAT, WCONHIST and WCONINJH all end this way.
        if _terminates_record(line) and line.replace('/', '').strip() == '':
            current = None
            continue

        head = parts[0].upper()

        # Inside a keyword block every line is a record whose first token
        # is a well name or datum -- never a keyword.  Treating an
        # all-uppercase well name (T142X9, T769, W128XC302, ...) as a
        # keyword split WELSPECS/COMPDAT/WCONINJH blocks apart and
        # dropped whole injection controls.  A lone keyword line
        # (keyword plus at most a trailing ``/``) is the one exception:
        # it opens the next block, which is how single-record keywords
        # like TSTEP hand control back to the reader.
        if current is not None:
            lone = len(parts) == 1 or (len(parts) == 2 and parts[1] == '/')
            if lone and re.fullmatch(r'[A-Z][A-Z0-9_]*', head):
                schedule.setdefault(head, []).append([])
                current = head
                order.append(current)
                if parts[-1] == '/':
                    current = None
                continue
            if _terminates_record(line):
                schedule.setdefault(current, []).append(parts)
            else:
                pending = [line]
            continue

        # Outside any block a leading uppercase token opens a keyword.
        # Each block opens with an empty record -- the separator the
        # downstream consumer (``_consume_schedule_keyword_group``) uses
        # to tell one block of a keyword from the next, mirroring how the
        # old reader emitted ``[]`` for a bare keyword line.
        if re.fullmatch(r'[A-Z][A-Z0-9_]*', head):
            schedule.setdefault(head, []).append([])
            current = head
            order.append(current)
            rest = parts[1:]
            if '/' in parts:
                data = parts[:parts.index('/')]
                if data:
                    schedule.setdefault(current, []).append(data)
                current = None
            elif rest:
                # Keyword and record data on the same line, record open.
                schedule.setdefault(current, []).append(rest)
            continue

        schedule.setdefault('UNKNOWN', []).append(parts)

    if pending and current is not None:
        schedule.setdefault(current, []).append(' '.join(pending).split())

    # Convert token lists into numeric arrays where possible
    out = {}
    for k, recs in schedule.items():
        parsed = []
        for rec in recs:
            vals = _flatten_tokens(rec)
            if all(isinstance(v, (int, float)) for v in vals) and vals:
                parsed.append(np.asarray(vals, dtype=float))
            else:
                parsed.append(vals)
        out[k] = parsed

    out['_order'] = order

    # MRST's control/step structure, alongside -- never in place of --
    # the flat lists above.  A deck this parser cannot split into
    # controls still reads exactly as it did before.
    try:
        control, step, missing = read_schedule_control(block, start=start)
    except Exception as exc:                      # noqa: BLE001
        import warnings
        warnings.warn('SCHEDULE control splitting failed (%s); the deck '
                      'keeps its flat keyword lists only.' % exc,
                      RuntimeWarning)
    else:
        # SKIPRESTART belongs to the section, not to a step; see
        # read_schedule_control's docstring for why it rides along.
        if step.pop('SKIPRESTART', False):
            out['SKIPRESTART'] = True
        if control:
            out['control'] = control
            out['step'] = step
        if missing:
            out['_missing_control_keywords'] = missing

    return out
