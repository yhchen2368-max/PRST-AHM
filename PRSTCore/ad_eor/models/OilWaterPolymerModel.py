"""Port of MRST ``OilWaterPolymerModel.m``: two-phase oil/water + polymer.

Subclasses :class:`GenericBlackOilModel` and wires
``ad_eor.utils.equationsOilWaterPolymer`` into the base class's Newton
driver by overriding ``get_equations``/``updateState``/``checkConvergence``,
following the exact ``problem``-dict contract
``_get_equations_mrst_generic_ow`` et al. already establish for the plain
black-oil model (see ``ad_eor.models`` package docstring).
"""

import numpy as _np

from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel

from ..utils.equationsOilWaterPolymer import equationsOilWaterPolymer


class OilWaterPolymerModel(GenericBlackOilModel):
    """Two-phase oil/water system with polymer (``model.polymer = True``)."""

    def __init__(self, G=None, rock=None, fluid=None, *args, **kwargs):
        kwargs.setdefault('gas', False)
        kwargs.setdefault('mrst_generic_assembly', True)
        super().__init__(G=G, rock=rock, fluid=fluid, *args, **kwargs)
        self.polymer = True
        # GenericSurfactantPolymerModel.toleranceEOR: single dimensionless
        # tolerance for every EOR-component conservation equation.
        self.toleranceEOR = float(kwargs.get('toleranceEOR', 1.0e-3))

    def validateState(self, state):
        state = super().validateState(state)
        nc = self._num_cells()
        if 'polymer' not in state:
            state['polymer'] = _np.zeros(nc, dtype=float)
        if 'polymermax' not in state:
            state['polymermax'] = _np.asarray(state['polymer'], dtype=float).copy()
        return state

    def get_equations(self, state0, state, dt, drivingForces=None, **kwargs):
        state = self.validateState(state)
        state0 = self.validateState(state0)
        if drivingForces is None:
            drivingForces = {}
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        self._mrst_write_wellsol(state, wells)
        assembled, meta = equationsOilWaterPolymer(self, state0, state, dt, drivingForces, wells)
        nc, nw = self._num_cells(), len(wells)
        problem = {
            'Residuals': assembled.val, 'Jacobian': assembled.jac,
            'State': state, 'State0': state0, 'dt': float(dt), 'drivingForces': drivingForces,
            'equationNames': (['water'] * nc + ['oil'] * nc + ['polymer'] * nc +
                               ['waterWells'] * nw + ['oilWells'] * nw + ['closureWells'] * nw),
            'types': ['cell'] * (3 * nc) + ['perf'] * (2 * nw) + ['well'] * nw,
            'primaryVariables': ['pressure', 'sW', 'polymer', 'qWs', 'qOs', 'bhp'],
            'facilityPrimaryVariables': ['qWs', 'qOs', 'bhp'],
            'wellSol': state.get('wellSol', []),
        }
        return problem, state

    def updateState(self, state, problem, dx, drivingForces=None):
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        nc, nw = self._num_cells(), len(wells)
        dx = _np.asarray(dx, dtype=float).ravel()
        if dx.size != 3 * nc + 3 * nw:
            raise ValueError('Expected OilWaterPolymerModel update of length %d, got %d'
                              % (3 * nc + 3 * nw, dx.size))
        p0 = _np.asarray(state['pressure'], dtype=float).ravel()
        state['pressure'] = self.limit_pressure_increment(p0, dx[:nc])
        sw0 = _np.asarray(state['sW'], dtype=float).ravel()
        state['sW'] = _np.clip(self.limit_saturation_increment(sw0, dx[nc:2 * nc]), 0.0, 1.0)

        cp0 = _np.asarray(state['polymer'], dtype=float).ravel()
        cp = cp0 + dx[2 * nc:3 * nc]
        cpmax = float(self.fluid['cpmax'])
        state['polymer'] = _np.clip(cp, 0.0, cpmax)

        start = 3 * nc
        state['facility_qWs'] = _np.asarray(state['facility_qWs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_qOs'] = _np.asarray(state['facility_qOs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_bhp'] = _np.asarray(state['facility_bhp'], dtype=float) + dx[start:start + nw]
        self._mrst_apply_well_limits(state, wells)
        self._mrst_write_wellsol(state, wells)
        return state

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        state['polymermax'] = _np.maximum(
            _np.asarray(state.get('polymermax', state['polymer']), dtype=float),
            _np.asarray(state['polymer'], dtype=float))
        return state

    def checkConvergence(self, problem):
        residual = _np.asarray(problem['Residuals'], dtype=float).ravel()
        state, nc = problem['State'], self._num_cells()
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sw = _np.asarray(state['sW'], dtype=float).ravel()
        pW, pO, _ = self._phase_pressures(p, sw, _np.zeros(nc))
        pvt = self._phase_pvt_from_phase_pressures(
            pW, pO, pO, rs_override=_np.zeros(nc), rv_override=_np.zeros(nc), sG_override=_np.zeros(nc),
        )
        rhoW, rhoO, _ = self._mrst_surface_densities()
        pv, dt = self._mrst_pore_volume(p), float(problem['dt'])
        cnv, mb = [], []
        for iph, (b, rho) in enumerate(((pvt['bw'], rhoW), (pvt['bo'], rhoO))):
            c, m = self.cnv_mb_from_residual(residual[iph * nc:(iph + 1) * nc], b, rho, pv, dt)
            cnv.append(c)
            mb.append(m)
        # Port of GenericSurfactantPolymerModel.getConvergenceValues's EOR
        # scaling: the polymer residual is in "surface volume x
        # concentration" units (equationsOilWaterPolymer has no rhoWS
        # factor), so ``dt/(pv*cpmax)`` makes it dimensionless, comparable
        # against a single ``toleranceEOR`` regardless of grid/timestep size.
        resP = residual[2 * nc:3 * nc]
        cpmax = float(self.fluid['cpmax'])
        mbP = float(_np.max(_np.abs(resP * (dt / (pv * cpmax))))) if resP.size else 0.0
        nw = max(0, (residual.size - 3 * nc) // 3)
        well_values = []
        for i in range(3):
            block = residual[3 * nc + i * nw:3 * nc + (i + 1) * nw]
            if nw:
                well_values.append(float(_np.max(_np.abs(block))))
        values = _np.asarray(cnv + mb + [mbP] + well_values)
        tolerances = _np.asarray([self.toleranceCNV] * 2 + [self.toleranceMB] * 2 + [self.toleranceEOR] +
                                  [self.toleranceWellRate] * len(well_values))
        names = (['CNV_W', 'CNV_O', 'MB_W', 'MB_O', 'MB_Polymer'] +
                 ['waterWells (perf)', 'oilWells (perf)', 'closureWells (well)'][:len(well_values)])
        return values < tolerances, values, names[:len(values)]
