"""Port of MRST ``getPerforationToWellMapping.m``
(mrst-2026a/autodiff/ad-core/utils).

Maps a global perforation number to the well it belongs to.
"""

import numpy as _np
import scipy.sparse as _sp


def getPerforationToWellMapping(w, with_Rw=False):
    """Return ``perf2well``, or ``(perf2well, Rw)`` when ``with_Rw``.

    ``perf2well[ix]`` is the (0-based) well number of global perforation
    ``ix``; ``Rw`` is the perforation-to-well scatter matrix, or the
    scalar 1 when every well has exactly one perforation -- the same
    shortcut the MATLAB takes.
    """
    if w is None or len(w) == 0:
        return (_np.zeros(0, dtype=int), None) if with_Rw \
            else _np.zeros(0, dtype=int)

    nw = len(w)
    nConn = _np.array([_np.size(_cells(well)) for well in w], dtype=int)
    perf2well = _np.repeat(_np.arange(nw), nConn)

    if not with_Rw:
        return perf2well

    nperf = perf2well.size
    if nperf == nw:
        return perf2well, 1
    Rw = _sp.csr_matrix((_np.ones(nperf), (_np.arange(nperf), perf2well)),
                        shape=(nperf, nw))
    return perf2well, Rw


def _cells(well):
    cells = well['cells'] if isinstance(well, dict) else getattr(well, 'cells')
    return _np.atleast_1d(_np.asarray(cells)).ravel()
