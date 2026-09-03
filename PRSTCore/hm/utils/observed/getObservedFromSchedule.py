"""Port of MRST ``getObservedFromSchedule.m``
(mrst-2026a/hm/utils/observed).

Builds the observed-data container from a schedule whose wells already
carry rates -- the "truth case" a synthetic history match is scored
against.

Two shapes, matching the simulator the data will be compared with:

``'mrst'``   one dict per report step, each with a ``wellSol`` list and
             the step's ``dt`` -- the shape the objective functions read.
``'Jutul'``  one dict of ``(nstep, nwell)`` arrays plus ``names``.
"""

import numpy as _np

_FIELDS = ('qWs', 'qOs', 'qGs', 'bhp')


def _control_index(step, i, ncontrols):
    """Which control governs report step ``i``.

    MATLAB's ``schedule.control(ctrl)`` is 1-based, so a literal
    transcription subtracts one. PRSTCore's ``step['control']`` is
    already 0-based, and subtracting again shifts every step back by
    one -- step 0 then reads ``controls[-1]``, the *last* control, and
    the observed rates a match is scored against belong to the wrong
    date throughout. Off by one and still perfectly plausible: the
    numbers are real rates, just not this step's.
    """
    which = int(_np.asarray(step['control']).ravel()[i])
    return min(max(which, 0), ncontrols - 1)


def getObservedFromSchedule(schedule, simulator='mrst'):
    """Return the observed container for ``schedule``."""
    step = schedule['step']
    dt = _np.atleast_1d(_np.asarray(step['val'], dtype=float)).ravel()
    nstep = dt.size
    controls = schedule['control']
    nwell = len(controls[0]['W'])

    if str(simulator).lower() == 'mrst':
        observed = []
        for i in range(nstep):
            # MATLAB's control index is one-based.
            W = controls[_control_index(step, i, len(controls))]['W']
            entry = {'dt': float(dt[i])}
            if W:
                entry['wellSol'] = [
                    {'name': w['name'], 'sign': w.get('sign'),
                     'status': w.get('status'),
                     'qWs': w.get('qWs'), 'qOs': w.get('qOs'),
                     'qGs': w.get('qGs'), 'bhp': w.get('bhp')}
                    for w in W]
            observed.append(entry)
        return observed

    if str(simulator).lower() == 'jutul':
        out = {name: _np.zeros((nstep, nwell)) for name in _FIELDS}
        for i in range(nstep):
            W = controls[_control_index(step, i, len(controls))]['W']
            for name in _FIELDS:
                out[name][i, :] = [float(_np.ravel(w.get(name, 0.0))[0])
                                   for w in W]
        out['names'] = [w['name'] for w in controls[0]['W']]
        return out

    raise ValueError('Unsupported keywords')
