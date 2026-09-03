"""Port of MRST ``optimizeBoundConstrainedForFAHM.m``
(mrst-2026a/hm/utils/optimizer).

The optimiser FAHM drives. Algorithmically it is
:mod:`optimizeLinIneqConstrained` -- diffing the two MATLAB files gives
124 added lines against 953 shared ones -- so this imports that module's
internals rather than repeating them, the same way MRST's own
``unitBoxLMMulti2`` is a thin variant of ``unitBoxLMMulti``.

What FAHM adds:

* **A case directory per objective evaluation.** ``dir.work/case<it>`` is
  wiped and recreated before each evaluation, and the objective is handed
  the path so the simulator has somewhere to write. This is why the
  objective here takes ``f(u, dir)`` rather than ``f(u)``.
* **``params`` carried in the history**, so a run can be resumed or
  inspected against the parameter set it was tuning.
* **``maximize`` defaults to true**, unlike every sibling in this module.
  A caller minimising a mismatch must pass ``maximize=False`` or the
  optimiser will faithfully make the match worse.
* Progress plotting, which has no headless equivalent and is left out;
  the same information is in the returned history.
"""

import os as _os
import shutil as _shutil

import numpy as _np

from .optimizeLinIneqConstrained import DEFAULTS, _run

# FAHM's own defaults. Note `maximize` -- see the module docstring.
FAHM_DEFAULTS = dict(DEFAULTS)
FAHM_DEFAULTS.update(maximize=True, params=None)


def optimizeBoundConstrainedForFAHM(u0, f, dir, **kwargs):
    """Return ``(v, u, history)``.

    ``f(u, case_dir)`` must return ``(value, gradient)`` and may write into
    ``case_dir``, which is freshly created for it. ``dir`` is a mapping
    with a ``'work'`` key naming the run directory; the history is written
    to ``<work>/history.npz`` after every iteration, as the MATLAB writes
    ``history.mat``.
    """
    opt = dict(FAHM_DEFAULTS)
    opt.update(kwargs)
    work = dir['work'] if isinstance(dir, dict) else getattr(dir, 'work')

    def per_iteration(it):
        """The objective for iteration ``it``, with its own case directory.

        The directory is created when the objective is *called*, not when
        it is built, so a line search that evaluates several times reuses
        one directory per iteration -- which is what the MATLAB does, its
        wipe-and-recreate happening inside fNegative.
        """
        case_dir = _os.path.join(work, 'case%d' % it)

        def objective(u):
            _makeCaseDir(case_dir)
            return f(u, case_dir)
        return objective

    return _run(u0, per_iteration(0), opt, per_iteration=per_iteration,
                extra=opt['params'], on_iteration=_saver(work))


def _makeCaseDir(path):
    """Port of ``fNegative``'s directory handling: wipe and recreate.

    The MATLAB falls back to a PowerShell ``Remove-Item`` when ``rmdir``
    fails, which is a Windows workaround for long paths; ``shutil`` needs
    no such fallback.
    """
    if _os.path.isdir(path):
        _shutil.rmtree(path, ignore_errors=True)
    _os.makedirs(path, exist_ok=True)
    return path


def _saver(work):
    """Write the history after each iteration, as MRST saves history.mat."""
    def save(history):
        try:
            _os.makedirs(work, exist_ok=True)
            _np.savez(_os.path.join(work, 'history.npz'),
                      val=_np.asarray(history['val'], dtype=float),
                      pg=_np.asarray(history['pg'], dtype=float),
                      u=_np.asarray(history['u'], dtype=float))
        except Exception:
            # Losing the checkpoint must not abort an expensive run.
            pass
    return save
