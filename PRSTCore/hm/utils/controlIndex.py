"""Which control governs a report step.

MATLAB's ``schedule.control(ctrl)`` is 1-based, so a literal
transcription of ``schedule.control(schedule.step.control(i))`` writes
``control[step['control'][i] - 1]``. PRSTCore's ``step['control']`` is
already 0-based -- ``schedule_control`` builds it that way and
``write_schedule`` addresses it that way -- so the subtraction shifts
every step back by one. Step 0 then reads ``controls[-1]``, the *last*
control, and every later step reads its predecessor's.

This is the quietest defect in the port. Nothing raises, nothing is out
of range, and the values that come back are real well controls -- just
the wrong date's. It has been found four separate times in four
different files, so the arithmetic lives here once rather than being
retyped at each call site.
"""

import numpy as _np

__all__ = ['control_index']


def control_index(step, i, ncontrols):
    """Return the 0-based control index governing report step ``i``.

    Clamped to the available controls: a schedule may name a control for
    a step it does not carry, and MATLAB would error where the intent is
    plainly the nearest one.
    """
    which = int(_np.atleast_1d(_np.asarray(step['control'])).ravel()[i])
    if ncontrols <= 0:
        return 0
    return min(max(which, 0), int(ncontrols) - 1)
