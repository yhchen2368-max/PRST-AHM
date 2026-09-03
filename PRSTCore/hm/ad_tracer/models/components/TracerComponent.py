"""Port of MRST ``TracerComponent.m``
(mrst-2026a/hm/ad-tracer/models/components).

A ``ConcentrationComponent`` carried entirely by the water phase.  MRST
declares it against the StateFunction dependency graph; PRSTCore's models
assemble procedurally, so the dependency declarations have no counterpart
and only the four evaluation methods are ported.  Each returns a per-phase
list whose water entry is populated and whose other entries are ``None``,
matching MATLAB's ``cell(1, nph)`` with only ``c{wIx}`` assigned.
"""

import numpy as _np

_W_IX = 0  # wIx = 1 in the MATLAB source (water is the first phase)


class TracerComponent:

    def __init__(self, tracerIndex=0, tracerName='tracer'):
        self.tracerIndex = int(tracerIndex)
        self.tracerName = str(tracerName)
        self.name = self.tracerName

    def _select(self, ct):
        """MATLAB's ``if iscell(ct), ct = ct{tIx}; else ct = ct(:,tIx); end``."""
        if isinstance(ct, (list, tuple)):
            return ct[self.tracerIndex]
        arr = _np.asarray(ct)
        if arr.ndim == 2:
            return arr[:, self.tracerIndex]
        return arr

    def getComponentDensity(self, ct, b, nph):
        """``c{wIx} = ct .* b{wIx}``."""
        c = [None] * nph
        c[_W_IX] = self._select(ct) * b[_W_IX]
        return c

    def getComponentMass(self, ct, b, pv, sw, nph):
        """``c{wIx} = pv .* (sw .* ct .* bW)``."""
        c = [None] * nph
        c[_W_IX] = pv * (sw * self._select(ct) * b[_W_IX])
        return c

    def getComponentMobility(self, ct, b, mob, nph):
        """``cmob{wIx} = ct .* bW .* mobW``."""
        cmob = [None] * nph
        cmob[_W_IX] = self._select(ct) * b[_W_IX] * mob[_W_IX]
        return cmob

    def getInjectionMassFraction(self, force, rhoWS):
        """``c = vertcat(force.tracer); c = c(:,tIx)./model.fluid.rhoWS``."""
        tracer = force['tracer'] if isinstance(force, dict) else getattr(force, 'tracer')
        return self._select(tracer) / float(rhoWS)
