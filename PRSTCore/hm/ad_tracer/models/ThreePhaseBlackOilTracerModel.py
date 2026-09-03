"""Port of MRST ``ThreePhaseBlackOilTracerModel.m``
(mrst-2026a/hm/ad-tracer/models).

Three-phase black oil with any number of passive water-borne tracers.
``stepFunctionIsLinear = true``: the tracers ride the water phase and do
not feed back into it, so one Newton step solves the coupled system
exactly.
"""

import numpy as _np

from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel

from ..utils.equationsThreePhaseBlackOilTracer import equationsThreePhaseBlackOilTracer


class ThreePhaseBlackOilTracerModel(GenericBlackOilModel):

    def __init__(self, G=None, rock=None, fluid=None, tracerNames=None,
                 *args, **kwargs):
        kwargs.setdefault('mrst_generic_assembly', True)
        super().__init__(G=G, rock=rock, fluid=fluid, *args, **kwargs)
        self.tracerNames = list(tracerNames or [])
        self.stepFunctionIsLinear = True
        self.toleranceTracer = float(kwargs.get('toleranceTracer', 1.0e-3))

    def getNumberOfTracers(self):
        """Port of ``getNumberOfTracers``."""
        return len(self.tracerNames)

    def getComponentNames(self):
        """Port of ``getComponentNames``: base component names + tracers."""
        names = list(getattr(super(), 'getComponentNames', lambda: [])() or [])
        return names + list(self.tracerNames)

    def getVariableField(self, name):
        """Port of ``getVariableField``.

        A tracer name resolves to ``('tracer', index)``; ``'tracer'`` itself
        to the whole container; ``qw<name>`` to that tracer's well rate.
        """
        lowered = str(name).lower()
        for i, tname in enumerate(self.tracerNames):
            if lowered == str(tname).lower():
                return 'tracer', i
        for tname in self.tracerNames:
            if lowered == ('qw' + str(tname)).lower():
                return 'qW' + str(tname), slice(None)
        if lowered == 'tracer':
            return 'tracer', slice(None)
        parent = getattr(super(), 'getVariableField', None)
        if parent is not None:
            return parent(name)
        return name, slice(None)

    def validateState(self, state):
        """Port of ``validateState``: default every tracer to zero."""
        state = super().validateState(state)
        nc = self._num_cells()
        if 'tracer' not in state:
            state['tracer'] = [_np.zeros(nc, dtype=float)
                               for _ in range(self.getNumberOfTracers())]
        return state

    def getExtraWellEquationNames(self):
        """Port of ``getExtraWellEquationNames``: one 'perf' equation per tracer."""
        names, types = [], []
        parent = getattr(super(), 'getExtraWellEquationNames', None)
        if parent is not None:
            names, types = parent()
            names, types = list(names), list(types)
        for tname in self.tracerNames:
            names.append(str(tname) + 'Wells')
            types.append('perf')
        return names, types

    def getExtraWellPrimaryVariableNames(self):
        """Port of ``getExtraWellPrimaryVariableNames``."""
        names = []
        parent = getattr(super(), 'getExtraWellPrimaryVariableNames', None)
        if parent is not None:
            names = list(parent())
        return names + ['qW' + str(t) for t in self.tracerNames]

    def get_equations(self, state0, state, dt, drivingForces=None, **kwargs):
        state = self.validateState(state)
        state0 = self.validateState(state0)
        if drivingForces is None:
            drivingForces = {}
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        self._mrst_write_wellsol(state, wells)
        assembled, meta = equationsThreePhaseBlackOilTracer(
            self, state0, state, dt, drivingForces, wells)
        return self._tracer_problem(assembled, meta, state, state0, dt,
                                    drivingForces, wells)

    def _tracer_problem(self, assembled, meta, state, state0, dt, drivingForces, wells):
        """Build the ``problem`` dict the Newton driver consumes."""
        nc, nw = self._num_cells(), len(wells)
        nt = self.getNumberOfTracers()
        nph = 3 if self.gas else 2
        names = (['water', 'oil', 'gas'][:nph] + list(self.tracerNames)
                 + ['waterWells', 'oilWells', 'gasWells'][:nph] + ['closureWells'])
        problem = {
            'Residuals': assembled.val,
            'Jacobian': assembled.jac,
            'State': state, 'State0': state0, 'dt': float(dt),
            'drivingForces': drivingForces,
            'primaryVars': (['pressure', 'sW', 'x'][:nph] + list(self.tracerNames)),
            'equationNames': names,
            'nc': nc, 'nw': nw, 'nt': nt,
        }
        problem.update(meta or {})
        return problem, state

    def updateState(self, state, problem, dx, drivingForces=None):
        state, report = super().updateState(state, problem, dx, drivingForces)
        return state, report
