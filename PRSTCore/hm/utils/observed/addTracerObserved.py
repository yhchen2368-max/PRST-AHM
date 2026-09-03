"""Port of MRST ``addTracerObserved.m`` (mrst-2026a/hm/utils/observed).

Sets up an interwell tracer test: builds the injection source for each
tracer slug and records the measured breakthrough at the producers.

The injected mass is spread over the injector's perforations according to
the logged injection interval, and the source concentration is the dosage
divided by the total injected volume over one day.

The MATLAB cannot run as written -- two undefined variables:

* ``wellnames = {W.name}'`` on the first line, but ``W`` is never
  assigned. Every sibling in this directory opens with
  ``W = schedule.control(1).W``, which is what the port does.
* ``tracerNames`` is read in the loop, but the variable defined above is
  ``tracNames``.

Both are filled in with the evident intent rather than reproduced.
"""

import numpy as _np

from .addProfileObserved import getDepthDependentAdditive
from .getCellFacesDepth import getCellFacesDepth

_DAY = 86400.0
_SOURCE_RATE = 0.1 / _DAY   # 0.1 m^3/day per perforation before weighting


def addTracerObserved(observed, time_sim, data, G, schedule, phNames):
    """Return ``(observed, schedule)``.

    ``data`` is a list of slug records, each with ``injector``, ``name``,
    ``depth`` (an ``(n, 2)`` top/bottom array), ``dosage``, ``date``,
    ``producer`` (the monitored well names) and ``output`` (rows of
    ``[date, c_1, ..., c_nproducer]``).
    """
    W = schedule['control'][0]['W']
    wellnames = [w['name'] for w in W]
    tracerNames = _unique_stable([str(rec['name']) for rec in data])
    nwells = len(W)
    nphase = len(phNames)
    ntracer = len(tracerNames)

    for step in range(len(observed)):
        sols = observed[step].setdefault('wellsol', [{} for _ in range(nwells)])
        for w in range(nwells):
            sols[w]['tracer'] = _np.zeros(ntracer)

    for rec in data:
        w = _index(wellnames, rec['injector'])
        c = _index(tracerNames, rec['name'])
        if w is None or c is None:
            continue

        depth = _np.atleast_2d(_np.asarray(rec['depth'], dtype=float))
        # A virtual unit-height log over the injection interval, so the
        # integral gives each cell's share of the interval.
        virtual = {'top': depth[:, 0], 'bottom': depth[:, 1],
                   'cell': _np.ones(depth.shape[0])}
        top, bottom = getCellFacesDepth(G, W[w]['cells'])
        ratio = getDepthDependentAdditive(virtual, top, bottom, 'cell')

        srcCells = _np.atleast_1d(_np.asarray(W[w]['cells'], dtype=int)).ravel()
        srcVals = _SOURCE_RATE * _np.ones(srcCells.size) * ratio
        srcSat = _np.zeros(nphase)
        srcSat[0] = 1.0
        srcTracer = _np.zeros((srcCells.size, ntracer))
        total = float(srcVals.sum())
        if total != 0:
            srcTracer[:, c] = float(rec['dosage']) / (total * _DAY)

        src = {'cell': srcCells, 'rate': srcVals,
               'sat': _np.tile(srcSat, (srcCells.size, 1)),
               'tracer': srcTracer}
        step = _step_for(time_sim, rec['date'])
        if step is not None:
            schedule['control'][step]['src'] = src

        output = _np.atleast_2d(_np.asarray(rec['output'], dtype=object))
        producers = list(rec['producer'])
        for row in output:
            step = _step_for(time_sim, row[0])
            if step is None:
                continue
            for k, producer in enumerate(producers):
                pw = _index(wellnames, producer)
                if pw is None:
                    continue
                observed[step]['wellsol'][pw]['tracer'][c] = float(row[k + 1])

    return observed, schedule


def _index(names, target):
    lowered = str(target).lower()
    for i, n in enumerate(names):
        if str(n).lower() == lowered:
            return i
    return None


def _step_for(time_sim, value):
    matches = _np.flatnonzero(_np.asarray(time_sim) == value)
    return int(matches[0]) if matches.size else None


def _unique_stable(values):
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
