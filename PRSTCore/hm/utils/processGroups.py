"""Port of MRST ``processGroups.m`` (mrst-2026a/hm/utils).

Turns a schedule control's GCONINJE/GCONPROD records into group control
structures shaped like MRST's well structs -- ``name``, ``type``, ``val``,
``compi``, ``sign``, ``status`` and a ``lims`` sub-struct.

Injection groups come first (GCONINJE), then production (GCONPROD), which
is the order the two stages run in the MATLAB.

Sign convention: injection targets are positive as given, production
targets are negated, so a group's ``val`` always reads as a signed rate.
"""

import numpy as _np

_INJ_COMPI = {'w': [1.0, 0.0, 0.0], 'o': [0.0, 1.0, 0.0], 'g': [0.0, 0.0, 1.0]}

# Production control mode -> (0-based GCONPROD column, compi).
_PROD_MODES = {
    'orat': (2, [0.0, 1.0, 0.0]),
    'wrat': (3, [1.0, 0.0, 0.0]),
    'grat': (4, [0.0, 0.0, 1.0]),
    # LIQUID rate is water + oil at surface conditions.
    'lrat': (5, [1.0, 1.0, 0.0]),
    'resv': (13, [1.0, 1.0, 1.0]),
}


def processGroups(control, InnerProduct='ip_tpf', Verbose=False):
    """Return the list of group controls for one schedule control."""
    G = []
    if 'GCONINJE' in control:
        G = _process_gconinje(G, control, Verbose)
    if 'GCONPROD' in control:
        G = _process_gconprod(G, control, Verbose)
    return G


def _process_gconinje(G, control, verbose):
    """Port of ``process_gconinje``."""
    for row in control.get('GCONINJE') or []:
        if not row:
            continue
        name = row[0]
        mode = str(row[2]).lower()
        if mode == 'rate':
            val = row[3]
        elif mode == 'resv':
            val = row[4]
        else:
            if verbose:
                print("Control mode '%s' unsupported for group '%s'.  Ignored."
                      % (mode.upper(), name))
            val = 0

        phase = str(row[1])[:1].lower()
        if phase not in _INJ_COMPI:
            if verbose:
                print("Injection phase '%s' is unknown.  Well ignored." % row[1])
            continue

        G.append({
            'name': name, 'type': mode, 'val': val,
            'compi': _np.asarray(_INJ_COMPI[phase], dtype=float),
            'sign': 1, 'status': True,
            'lims': {'rate': row[3]},
        })
    return G


def _process_gconprod(G, control, verbose):
    """Port of ``process_gconprod``."""
    for row in control.get('GCONPROD') or []:
        if not row:
            continue
        name = row[0]
        mode = str(row[1]).lower()
        if mode in _PROD_MODES:
            column, compi = _PROD_MODES[mode]
            val = -_num(row, column)
        else:
            if verbose:
                print("Control mode '%s' unsupported for group '%s'.  Ignored."
                      % (mode.upper(), name))
            val, compi = 0, [0.0, 1.0, 0.0]

        G.append({
            'name': name, 'type': mode, 'val': val,
            'compi': _np.asarray(compi, dtype=float),
            'sign': -1, 'status': True,
            'lims': {'orat': -_num(row, 2), 'wrat': -_num(row, 3),
                     'grat': -_num(row, 4), 'lrat': -_num(row, 5)},
        })
    return G


def _num(row, index):
    """Numeric item ``index`` of a record, or 0 when absent/non-numeric."""
    if index >= len(row):
        return 0.0
    value = row[index]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
