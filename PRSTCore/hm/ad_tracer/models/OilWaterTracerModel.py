"""Port of MRST ``OilWaterTracerModel.m`` (mrst-2026a/hm/ad-tracer/models).

Two-phase oil/water with passive water-borne tracers.  The MATLAB class
forces the phase flags after construction:

    model.water = true; model.oil = true; model.gas = false;
    model.disgas = false; model.vapoil = false;
    model.stepFunctionIsLinear = true;
"""

from .ThreePhaseBlackOilTracerModel import ThreePhaseBlackOilTracerModel
from ..utils.equationsOilWaterTracer import equationsOilWaterTracer


class OilWaterTracerModel(ThreePhaseBlackOilTracerModel):

    def __init__(self, G=None, rock=None, fluid=None, tracerNames=None,
                 *args, **kwargs):
        kwargs.setdefault('gas', False)
        super().__init__(G=G, rock=rock, fluid=fluid, tracerNames=tracerNames,
                         *args, **kwargs)
        self.water = True
        self.oil = True
        self.gas = False
        self.disgas = False
        self.vapoil = False
        self.stepFunctionIsLinear = True

    def get_equations(self, state0, state, dt, drivingForces=None, **kwargs):
        state = self.validateState(state)
        state0 = self.validateState(state0)
        if drivingForces is None:
            drivingForces = {}
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        self._mrst_write_wellsol(state, wells)
        assembled, meta = equationsOilWaterTracer(
            self, state0, state, dt, drivingForces, wells)
        return self._tracer_problem(assembled, meta, state, state0, dt,
                                    drivingForces, wells)
