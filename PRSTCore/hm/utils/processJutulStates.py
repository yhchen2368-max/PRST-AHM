"""Port of MRST ``processJutulStates.m`` (mrst-2026a/hm/utils).

Converts the well solutions and states a Jutul (Julia) run returns into the
shapes MRST's own post-processing expects: each timestep's schedule wells
are refreshed with the rates and bhp Jutul reported, matched by name.

Only the first loop is ported. The MATLAB's second loop -- rebuilding an AD
state through ``model.getStateAD``/``reduceState``/
``FacilityModel.updateAfterConvergence`` -- depends on a ``setupNew``
variable the function never receives and never defines, so it cannot run as
written; see :func:`processJutulStates` for the details.
"""

import copy as _copy

from PRSTCore.hm.utils.controlIndex import control_index


def processJutulStates(setup, wellSols, states):
    """Return ``(wellSols, states)`` with schedule wells carrying Jutul's
    reported rates.

    The MATLAB's second loop reads ``setupNew.model``/``setupNew.state0``,
    but ``setupNew`` is neither an argument nor assigned anywhere in the
    file -- in MATLAB that raises ``Undefined variable``. It is therefore
    not ported; doing so would mean inventing an interface the source does
    not define. The first loop, which is what the name promises, is ported
    in full.
    """
    schedule = setup['schedule'] if isinstance(setup, dict) else setup.schedule
    step = schedule['step']

    for i in range(len(states)):
        control_no = control_index(step, i, len(schedule['control']))
        W = _copy.deepcopy(schedule['control'][control_no]['W'])

        reported = {str(w['name']): w for w in wellSols[i]}
        for well in W:
            source = reported.get(str(well['name']))
            if source is None:
                continue
            for field in ('qWs', 'qOs', 'qGs', 'bhp', 'status'):
                if field in source:
                    well[field] = source[field]

        states[i]['wellSols'] = W
        wellSols[i] = W

    return wellSols, states
