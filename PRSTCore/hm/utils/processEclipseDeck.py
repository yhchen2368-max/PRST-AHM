"""Port of MRST ``processEclipseDeck.m`` (mrst-2026a/hm/utils).

Conditions a parsed ECLIPSE deck for a history-matching run:

* force unified, unformatted output;
* make sure endpoint scaling is advertised (ENDSCALE / SCALECRS);
* drop leading schedule steps and controls that precede the first real
  control and the first WELSPECS, folding their durations into the first
  surviving step;
* drop WELSPECS/COMPDAT rows for wells that have no control record;
* fold each control's WEFAC efficiency factor into the well target rates.
"""

import copy as _copy

import numpy as _np

# Columns (0-based) each keyword scales by WEFAC.
_WEFAC_COLUMNS = {
    'WCONINJE': (4,),
    'WCONINJH': (3,),
    'WCONPROD': (3, 4, 5),
    'WCONHIST': (3, 4, 5),
}


def processEclipseDeck(deck):
    """Return ``deck`` conditioned in place."""
    runspec = deck.setdefault('RUNSPEC', {})
    runspec['UNIFOUT'] = True
    runspec['FMTOUT'] = False
    runspec.setdefault('ENDSCALE', ['NODIR', 'REVERS', 1, 20, 0])

    props = deck.setdefault('PROPS', {})
    props.setdefault('SCALECRS', ['NO'])

    schedule = deck.get('SCHEDULE')
    if not schedule:
        return deck
    schedule = _copy.deepcopy(schedule)

    step = schedule.get('step', {})
    control_ix = _np.atleast_1d(_np.asarray(step.get('control', []))).ravel()
    values = _np.atleast_1d(_np.asarray(step.get('val', []), dtype=float)).ravel()

    # Drop leading steps whose control index is zero.
    # ``find(SCHEDULE.step.control > 0, 1, 'first')``: MATLAB numbers
    # controls from 1, so nought there means *no control*, and this drops
    # the leading steps that have none. PRSTCore numbers them from 0, so
    # nought is a perfectly good control and "no control" is negative.
    # Testing ``> 0`` here dropped the first report step of every deck and
    # folded its duration into the second -- QIEDIE went from 63 steps to
    # 62, with the first two weeks merged into one fortnight and control 0
    # never used.
    positive = _np.flatnonzero(control_ix >= 0)
    if positive.size and positive[0] > 0:
        ix = int(positive[0])
        step['control'] = control_ix[ix:]
        step['val'] = _np.concatenate([[values[:ix + 1].sum()], values[ix + 1:]])
        control_ix, values = step['control'], step['val']

    # Drop leading controls that carry no WELSPECS.
    controls = schedule.get('control', [])
    with_wells = [i for i, c in enumerate(controls) if c.get('WELSPECS')]
    if with_wells and with_wells[0] > 0:
        ix = int(with_wells[0])
        schedule['control'] = controls[ix:]
        # ``step.control(ix:end) - step.control(ix-1)`` renumbers so the
        # first surviving control is the lowest valid index -- 1 in
        # MATLAB. Here it is 0, and ``ix`` controls were dropped, so the
        # new index of old control k is k - ix.
        step['control'] = control_ix[ix:] - ix
        step['val'] = _np.concatenate([[values[:ix + 1].sum()], values[ix + 1:]])

    for control in schedule.get('control', []):
        _removeExtraWellCompdat(control)
        _apply_wefac(control)

    deck['SCHEDULE'] = schedule
    return deck


def _apply_wefac(control):
    """Scale each well's target rates by its WEFAC efficiency factor."""
    wefac = control.get('WEFAC')
    if not wefac:
        return
    factors = {str(row[0]): float(row[1]) for row in wefac if len(row) >= 2}
    for keyword, columns in _WEFAC_COLUMNS.items():
        records = control.get(keyword)
        if not records:
            continue
        for row in records:
            if not row:
                continue
            factor = factors.get(str(row[0]))
            if factor is None:
                continue
            for col in columns:
                if col < len(row) and isinstance(row[col], (int, float)):
                    row[col] = row[col] * factor


def _removeExtraWellCompdat(control):
    """Port of ``removeExtraWellCompdat``: keep only wells with a control."""
    named = []
    for keyword in ('WCONINJE', 'WCONINJH', 'WCONPROD', 'WCONHIST'):
        for row in control.get(keyword) or []:
            if row:
                named.append(str(row[0]))
    known = set(named)
    for keyword in ('WELSPECS', 'COMPDAT'):
        records = control.get(keyword)
        if not records:
            continue
        kept = [row for row in records if row and str(row[0]) in known]
        if len(kept) != len(records):
            control[keyword] = kept
    return control
