"""Port of MRST ``ThreePhaseBlackOilPolymerModel.m``: three-phase black-oil
+ polymer, including PLYSHEAR/PLYSHLOG reservoir-side shear thinning (see
``ad_eor.utils.equationsThreePhaseBlackOilPolymer``'s module docstring for
the scope of what is/isn't ported for shear).
"""

import numpy as _np

from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel

from ..utils.equationsThreePhaseBlackOilPolymer import equationsThreePhaseBlackOilPolymer


class ThreePhaseBlackOilPolymerModel(GenericBlackOilModel):

    def __init__(self, G=None, rock=None, fluid=None, *args, **kwargs):
        kwargs.setdefault('gas', True)
        kwargs.setdefault('mrst_generic_assembly', True)
        super().__init__(G=G, rock=rock, fluid=fluid, *args, **kwargs)
        self.polymer = True
        self.toleranceEOR = float(kwargs.get('toleranceEOR', 1.0e-3))
        fluid = fluid or {}
        has_shrate = 'shrate' in fluid
        has_plyshlog = 'plyshlog' in fluid
        has_plyshmult = 'plyshearMult' in fluid
        if has_shrate and not has_plyshlog:
            raise ValueError('SHRATE is specified while PLYSHLOG is not specified')
        if has_plyshmult and has_plyshlog:
            raise ValueError('PLYSHLOG and PLYSHEAR are existing together')
        self.usingShear = has_plyshmult
        self.usingShearLog = has_plyshlog and not has_shrate
        self.usingShearLogshrate = has_plyshlog and has_shrate

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
        assembled, meta = equationsThreePhaseBlackOilPolymer(self, state0, state, dt, drivingForces, wells)
        nc, nw = self._num_cells(), len(wells)
        problem = {
            'Residuals': assembled.val, 'Jacobian': assembled.jac,
            'State': state, 'State0': state0, 'dt': float(dt), 'drivingForces': drivingForces,
            'equationNames': (['water'] * nc + ['oil'] * nc + ['gas'] * nc + ['polymer'] * nc +
                               ['waterWells'] * nw + ['oilWells'] * nw + ['gasWells'] * nw +
                               ['closureWells'] * nw),
            'types': ['cell'] * (4 * nc) + ['perf'] * (3 * nw) + ['well'] * nw,
            'blackOilStatus': meta.get('status'),
            'primaryVariables': ['pressure', 'sW', 'x', 'polymer', 'qWs', 'qOs', 'qGs', 'bhp'],
            'facilityPrimaryVariables': ['qWs', 'qOs', 'qGs', 'bhp'],
            'wellSol': state.get('wellSol', []),
            'rs': _np.asarray(state['rs'], dtype=float).ravel(),
            'rv': _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel(),
        }
        return problem, state

    def updateState(self, state, problem, dx, drivingForces=None):
        # Reuse the plain 3-phase black-oil update (pressure/sW/x status
        # switching + qWs/qOs/qGs/bhp) for the first 3*nc+4*nw entries, then
        # apply the polymer update to the appended block, matching MRST's
        # ``updateState@ThreePhaseBlackOilModel`` followed by the polymer
        # clamp in ``OilWaterPolymerModel.updateState``/
        # ``ThreePhaseBlackOilPolymerModel.updateState``.
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        nc, nw = self._num_cells(), len(wells)
        dx = _np.asarray(dx, dtype=float).ravel()
        expected = 4 * nc + 4 * nw
        if dx.size != expected:
            raise ValueError('Expected ThreePhaseBlackOilPolymerModel update of length %d, got %d'
                              % (expected, dx.size))
        dx_bo = _np.concatenate([dx[:3 * nc], dx[3 * nc + nc:]])
        state = self._update_state_mrst_generic(state, problem, dx_bo, drivingForces)

        cp0 = _np.asarray(state['polymer'], dtype=float).ravel()
        cp = cp0 + dx[3 * nc:4 * nc]
        cpmax = float(self.fluid['cpmax'])
        state['polymer'] = _np.clip(cp, 0.0, cpmax)
        return state, {'Converged': False}

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        state['polymermax'] = _np.maximum(
            _np.asarray(state.get('polymermax', state['polymer']), dtype=float),
            _np.asarray(state['polymer'], dtype=float))
        return state

    def checkConvergence(self, problem):
        residual = _np.asarray(problem['Residuals'], dtype=float).ravel()
        state = problem['State']
        nc = self._num_cells()
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sw = _np.asarray(state['sW'], dtype=float).ravel()
        sg = _np.asarray(state['sG'], dtype=float).ravel()
        so = 1.0 - sw - sg
        pW, pO, pG = self._phase_pressures(p, sw, sg, state.get('pcowScale'))
        pvt = self._phase_pvt_from_phase_pressures(
            pW, pO, pG,
            rs_override=state.get('rs'), rv_override=state.get('rv'), sG_override=sg,
            oil_saturated_override=(sg > 0.0), gas_saturated_override=(so > 0.0),
        )
        b = (pvt['bw'], pvt['bo'], pvt['bg'])
        rho_s = self._mrst_surface_densities()
        pv, dt = self._mrst_pore_volume(p), float(problem['dt'])
        cnv, mb = [], []
        for iph in range(3):
            c, m = self.cnv_mb_from_residual(residual[iph * nc:(iph + 1) * nc], b[iph], rho_s[iph], pv, dt)
            cnv.append(c)
            mb.append(m)
        # Dimensionless EOR scaling, see OilWaterPolymerModel.checkConvergence.
        resP = residual[3 * nc:4 * nc]
        cpmax = float(self.fluid['cpmax'])
        mbP = float(_np.max(_np.abs(resP * (dt / (pv * cpmax))))) if resP.size else 0.0
        nw = max(0, (residual.size - 4 * nc) // 4)
        well_values = []
        for i in range(4):
            block = residual[4 * nc + i * nw:4 * nc + (i + 1) * nw]
            if nw:
                well_values.append(float(_np.max(_np.abs(block))))
        values = _np.asarray(cnv + mb + [mbP] + well_values)
        tolerances = _np.asarray([self.toleranceCNV] * 3 + [self.toleranceMB] * 3 + [self.toleranceEOR] +
                                  [self.toleranceWellRate] * len(well_values))
        names = (['CNV_W', 'CNV_O', 'CNV_G', 'MB_W', 'MB_O', 'MB_G', 'MB_Polymer'] +
                 ['waterWells (perf)', 'oilWells (perf)', 'gasWells (perf)', 'closureWells (well)'][:len(well_values)])
        return values < tolerances, values, names[:len(values)]
