import numpy as _np
from copy import deepcopy
try:
    import scipy.sparse as _sp
except Exception:
    _sp = None

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import is_ad as _is_ad
from PRSTCore.ad_core.conservation import CellVariableLayout as _CellVariableLayout
from PRSTCore.ad_core.conservation import DivergenceAssembler as _DivergenceAssembler
from PRSTCore.ad_core.conservation import FaceValue as _FaceValue
from PRSTCore.ad_core.conservation import upwind_flag as _upwind_flag
from PRSTCore.ad_core.backends import get_backend as _get_backend
from PRSTCore.ad_core.adi import ad_interp_linear as _ad_interp_linear
from PRSTCore.ad_core.adi import ad_maximum as _ad_maximum
from PRSTCore.ad_core.adi import ad_minimum as _ad_minimum
from PRSTCore.ad_core.adi import ad_select as _ad_select
from PRSTCore.ad_core.models.facility_model import FacilityModel as _FacilityModel
from PRSTCore.ad_core.models.reservoir_model import ReservoirModel as _ReservoirModel


def _value(x):
    """The numbers behind a quantity that may or may not carry
    derivatives.

    Endpoint scalers are plain arrays in every forward run and AD only
    while the adjoint is reading ``dR/dendpoint`` out of them. A branch
    test wants the values either way: which side of a kink a saturation
    falls on is a switch, not a differentiable function of it.
    """
    return x.val if _is_ad(x) else x


def _ad_where(mask, when_true, when_false):
    """``numpy.where`` that also accepts an AD ``when_false``."""
    if _is_ad(when_false) or _is_ad(when_true):
        return _ad_select(_np.asarray(mask, dtype=bool), when_true,
                          when_false)
    return _np.where(mask, when_true, when_false)


class GenericBlackOilModel(_ReservoirModel):
    """Python port of MRST GenericBlackOilModel (three-phase black-oil).

    Implements water/oil/gas conservation equations with:
    - Dissolved gas (Rs) and vaporized oil (Rv) mass transfer
    - Three-phase relative permeability and mobility
    - Upstream-weighted formation volume factor flux scaling
    - Sparse Jacobian assembly for (p, sW, sG) primary variables

    Inherits from :class:`ReservoirModel` (dpMax/dsMax limiting, CNV/MB
    convergence primitives, phase flags) but keeps its own defaults below
    where they differ from ``ReservoirModel``'s (e.g. ``nonlinearTolerance``,
    ``useCNVConvergence`` default to values tuned for this model).
    """

    def __init__(self, G=None, rock=None, fluid=None, *args, **kwargs):
        super().__init__(
            G=G, rock=rock, fluid=fluid,
            water=kwargs.get('water', True), oil=kwargs.get('oil', True), gas=kwargs.get('gas', False),
            dpMaxRel=kwargs.get('dpMaxRel', _np.inf), dpMaxAbs=kwargs.get('dpMaxAbs', _np.inf),
            dsMaxAbs=kwargs.get('dsMaxAbs', 0.2),
            minimumPressure=kwargs.get('minimumPressure', -_np.inf),
            maximumPressure=kwargs.get('maximumPressure', _np.inf),
            useCNVConvergence=kwargs.get('useCNVConvergence', True),
            toleranceCNV=kwargs.get('toleranceCNV', 1.0e-3),
            toleranceMB=kwargs.get('toleranceMB', 1.0e-7),
        )
        self.inputdata = None
        self.FacilityModel = _FacilityModel(well_cells_fn=self._well_cells)
        self.AquiferModel = None
        self.operators = None
        self.porevolume = None
        self.stepFunctionIsLinear = False
        self.nonlinearTolerance = 1e-8
        # ReservoirModel defaults used by the MRST generic black-oil path.
        self.toleranceWellRate = float(kwargs.get('toleranceWellRate', 1.0 / 86400.0))
        self.toleranceWellBHP = float(kwargs.get('toleranceWellBHP', 1.0e5))
        self.totalCompressibility = float(kwargs.get('totalCompressibility', 1e-3))
        self.defaultViscosityW = float(kwargs.get('muW', 1.0))
        self.defaultViscosityO = float(kwargs.get('muO', 3.0))
        self.defaultViscosityG = float(kwargs.get('muG', 0.05))
        self.defaultWI = float(kwargs.get('defaultWI', 1e-3))
        # Phase configuration flags not covered by ReservoirModel.
        self.disgas = bool(kwargs.get('disgas', True))
        self.vapoil = bool(kwargs.get('vapoil', False))
        self.enable_facility_unknowns = bool(kwargs.get('enable_facility_unknowns', False))
        # Deck-created models opt into the GenericBlackOilModel/FACILITY
        # assembly below.  Keep the older compatibility assembly available
        # for callers that construct this small Python model directly.
        self._use_mrst_generic_assembly = bool(kwargs.get('mrst_generic_assembly', False))
        self.gravity = _np.asarray(kwargs.get('gravity', [0.0, 0.0, 9.80665]), dtype=float)

    def setupOperators(self, G, rock):
        """Prepare discrete operators and attach them to the model."""
        self.G = G
        self.rock = rock
        self.operators = getattr(self, 'operators', None)
        if self.operators is None:
            try:
                from PRSTCore.ad_core.operators import setup_operators
                self.operators = setup_operators(G, rock)
            except Exception:
                self.operators = {}
        return self

    def getComponentNames(self):
        return ['W', 'O', 'G']

    def validateState(self, state):
        if state is None:
            state = {}
        nc = self._num_cells()
        if 'pressure' not in state:
            state['pressure'] = _np.ones(nc, dtype=float)
        if 'sW' not in state:
            # Backward compat: if old 'saturation' field exists, treat as sW
            if 'saturation' in state:
                state['sW'] = _np.asarray(state.pop('saturation'), dtype=float).ravel()
            else:
                state['sW'] = _np.zeros(nc, dtype=float)
        if 'sG' not in state:
            state['sG'] = _np.zeros(nc, dtype=float)
        if 'rs' not in state:
            state['rs'] = _np.zeros(nc, dtype=float)
        if 'rv' not in state:
            state['rv'] = _np.zeros(nc, dtype=float)
        if 'time' not in state:
            state['time'] = 0.0
        if 'wellSol' not in state:
            state['wellSol'] = []
        return state

    def validateSchedule(self, schedule):
        if schedule is None:
            raise ValueError('Schedule must be provided')
        if 'step' not in schedule or 'val' not in schedule['step']:
            raise ValueError('Schedule missing step.val')
        if 'control' not in schedule['step']:
            schedule['step']['control'] = _np.arange(len(schedule['step']['val']), dtype=int)
        if 'control' not in schedule:
            schedule['control'] = [{'W': []} for _ in schedule['step']['val']]
        return schedule

    def getValidDrivingForces(self):
        return {'W': []}

    def getDrivingForces(self, control):
        if control is None:
            return {'W': []}
        return {'W': control.get('W', [])}

    def updateForChangedControls(self, state, fstruct):
        # ``_mrst_active_wells``/``_ensure_mrst_facility_state`` cache
        # ``state['facility_wells']`` so within-ministep well-limit
        # switching (``_mrst_apply_well_limits``) doesn't get clobbered by
        # re-reading the (control-immutable) driving forces on every Newton
        # iteration -- see the comment on ``_mrst_active_wells``. That cache
        # must be invalidated here, at a genuine report-step control
        # change, or a later WCONPROD/WCONINJE/WPOLYMER/WSURFACT record
        # changing a well's rate/type/concentration would silently never
        # take effect for the rest of the simulation.
        # ``_ensure_mrst_facility_state`` already contains the correct
        # from-scratch reinitialization of qWs/qOs/qGs/bhp for a changed
        # well signature (the branch normally exercised on the very first
        # call); dropping the cached signature alongside the well list
        # reuses that exact same path instead of duplicating it here.
        if isinstance(state, dict):
            state.pop('facility_wells', None)
            state.pop('facility_well_signature', None)
        return self, state

    def prepareReportstep(self, state, state0, dt, drivingForces):
        state = deepcopy(state)
        state['time'] = float(state0.get('time', 0.0)) + float(dt)
        self._mrst_update_resv_controls(state, state0, drivingForces)
        return self, state

    def _mrst_update_resv_controls(self, state, state0, drivingForces):
        """Port of ``GenericFacilityModel.updateRESVControls``.

        A RESV well is rate-controlled at *reservoir* conditions, so its
        control equation needs surface-to-reservoir conversion factors --
        MRST's ``ControlDensity`` -- one per phase per well.  They come from
        shrinkage factors evaluated at the previous step's pore-volume
        weighted average pressure and dissolved-gas state, which is why this
        runs once per report step from ``prepareReportstep`` rather than per
        Newton iteration: the factors are frozen for the step, exactly as
        MRST freezes them.

        ``resv_history`` additionally converts to plain ``resv``, its target
        recomputed from the historical *surface* rates through those same
        factors.  The deck reader already produces both control types; only
        this half was missing, so every deck using RESV -- Norne and SPE10
        model 2 among them -- stopped at "Unsupported MRST well control
        type" the moment the first residual was assembled.
        """
        wells = self._mrst_active_wells(drivingForces, state)
        resv = [w for w in wells
                if str(w.get('type', '')).lower() in ('resv', 'resv_history')]
        if not resv:
            return

        nph = 3 if (self.water and self.oil and self.gas) else 2
        water_index, oil_index, gas_index = 0, 1, 2
        nc = self._num_cells()

        # Pore volume of the hydrocarbon part of the previous state, used to
        # weight the averages: MRST removes the water saturation from it.
        pore_volume = _np.asarray(self._state0_value(getattr(self, 'porevolume', None),
                                                     _np.ones(nc)), dtype=float).ravel()
        if pore_volume.size != nc:
            pore_volume = _np.ones(nc)
        if self.water:
            sw0 = _np.asarray(self._state0_value(state0.get('sW'), _np.zeros(nc)),
                              dtype=float).ravel()
            pore_volume = pore_volume * (1.0 - sw0)
        total_pv = float(_np.sum(pore_volume))
        if not _np.isfinite(total_pv) or total_pv <= 0.0:
            pore_volume = _np.ones(nc)
            total_pv = float(nc)

        p0 = _np.asarray(state0.get('pressure', _np.zeros(nc)), dtype=float).ravel()
        rs0_field = _np.asarray(state0.get('rs', _np.zeros(nc)), dtype=float).ravel()
        rv0_field = _np.asarray(state0.get('rv', _np.zeros(nc)), dtype=float).ravel()

        # One PVT region here, matching the single-region path everywhere
        # else in this port; MRST averages per region.
        p_mean = float(_np.sum(p0 * pore_volume) / total_pv)
        rs_mean = float(_np.sum(rs0_field * pore_volume) / total_pv) if self.disgas else 0.0
        rv_mean = float(_np.sum(rv0_field * pore_volume) / total_pv) if self.vapoil else 0.0

        for w in resv:
            is_history = str(w.get('type', '')).lower() == 'resv_history'
            compi = _np.asarray(w.get('compi', _np.zeros(nph)), dtype=float).ravel()
            if compi.size < nph:
                compi = _np.pad(compi, (0, nph - compi.size))
            qs = float(w.get('val', 0.0)) * compi[:nph]

            rs = rs_mean
            rv = rv_mean
            if is_history:
                # A historical RESV target's own gas/oil split bounds the
                # dissolved amounts: you cannot dissolve more gas than the
                # well actually produced.
                if self.disgas and abs(qs[oil_index]) > 0.0:
                    rs = min(qs[gas_index] / qs[oil_index], rs_mean)
                if self.vapoil and abs(qs[gas_index]) > 0.0:
                    rv = min(qs[oil_index] / qs[gas_index], rv_mean)

            pvt = self._phase_pvt(_np.full(1, p_mean),
                                  rs_override=_np.full(1, rs),
                                  rv_override=_np.full(1, rv))
            bW = float(pvt['bw'][0])
            bO = float(pvt['bo'][0])
            bG = float(pvt['bg'][0])
            shrink = 1.0 - rs * rv
            shrink0 = 1.0 - rs_mean * rv_mean

            factors = _np.zeros(nph)
            new_rate = 0.0
            if self.water:
                factors[water_index] = 1.0 / bW
                new_rate += qs[water_index] / bW
            if self.oil:
                oil_rate = qs[oil_index]
                factors[oil_index] += 1.0 / (bO * shrink0)
                if self.vapoil:
                    oil_rate = oil_rate - rv * qs[gas_index]
                    factors[gas_index] -= rv / (bO * shrink0)
                new_rate += oil_rate / (bO * shrink)
            if self.gas:
                gas_rate = qs[gas_index]
                factors[gas_index] += 1.0 / (bG * shrink0)
                if self.disgas:
                    gas_rate = gas_rate - rs * qs[oil_index]
                    factors[oil_index] -= rs / (bG * shrink0)
                new_rate += gas_rate / (bG * shrink)

            if is_history:
                w['val'] = float(new_rate)
                w['type'] = 'resv'
            w['ControlDensity'] = factors

        # ``updateRESVControls`` carries the new target and type across to
        # state.wellSol and stops there. Writing the whole wellSol would be
        # wrong here: this runs at report-step preparation, before
        # ``_ensure_mrst_facility_state`` has created the facility primary
        # variables, so the rate and pressure arrays it reads are still
        # empty and indexing them raises.
        well_sol = state.get('wellSol')
        if isinstance(well_sol, list):
            by_name = {str(w.get('name')): w for w in resv}
            for entry in well_sol:
                source = by_name.get(str(entry.get('name'))) if isinstance(entry, dict) else None
                if source is not None:
                    entry['val'] = source['val']
                    entry['type'] = source['type']

    def prepareTimestep(self, state, state0, dt, drivingForces):
        """Port ``GenericFacilityModel.prepareTimestep`` for deck AD runs.

        MRST freezes the well-bore hydrostatic connection pressure drops at
        the beginning of each mini-step and applies control limits before
        the first Newton system is assembled.  This must occur outside the
        Newton loop (``NonLinearSolver.solveMinistep``), not in a well
        residual evaluation.
        """
        if not getattr(self, '_use_mrst_generic_assembly', False):
            return self, state
        state = self.validateState(state)
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        self._mrst_update_connection_pressure_drop(state, wells)
        self._mrst_write_wellsol(state, wells)
        self._mrst_apply_well_limits(state, wells)
        self._mrst_write_wellsol(state, wells)
        return self, state

    def getMaximumTimestep(self, state, state0, dt, drivingForces):
        """PhysicalModel's default maximum timestep (MRST: ``inf``)."""
        return _np.inf

    def validateModel(self, fstruct=None, checkOperators=False):
        return self

    def reduceState(self, state, keepProperties):
        return state

    def makeStepReport(self, **kwargs):
        report = {
            'LinearSolver': None,
            'UpdateState': None,
            'Failure': False,
            'FailureMsg': '',
            'Converged': False,
            'Solved': False,
            'FinalUpdate': None,
            'Residuals': None,
            'ResidualsConverged': None,
            'Iterations': 0,
            'EarlyStop': False,
            'Time': None,
            'StepSize': None,
        }
        report.update(kwargs)
        return report

    def checkConvergence(self, problem):
        values, tolerances, names = self.getConvergenceValues(problem)
        return values < tolerances, values, names

    def getConvergenceValues(self, problem):
        """Port ``PhysicalModel.getConvergenceValues``: raw residual values,
        their per-entry tolerances and the equation names.

        ``NonLinearSolver.applyLinesearch`` normalizes with the tolerances,
        exactly as MRST's line search does.  The CNV/MB branch mirrors
        ``GenericBlackOilModel.getConvergenceValues`` (through
        ``getConvergenceValuesCNV``/``getConvergenceValuesWells``), the
        default branch the plain ``PhysicalModel`` inf-norm check.
        """
        if getattr(self, '_use_mrst_generic_assembly', False) and \
                getattr(self, 'useCNVConvergence', False) and isinstance(problem, dict):
            if not self.gas:
                return self._convergence_values_mrst_generic_ow(problem)
            if not self.water:
                return self._convergence_values_mrst_generic_og(problem)
            return self._convergence_values_mrst_generic(problem)
        return self._convergence_values_default(problem)

    def _convergence_values_mrst_generic(self, problem):
        """Port ``getConvergenceValuesCNV`` plus well convergence checks."""
        if not self.gas:
            return self._convergence_values_mrst_generic_ow(problem)
        if not self.water:
            return self._convergence_values_mrst_generic_og(problem)
        residual = _np.asarray(problem['Residuals'], dtype=float).ravel()
        state = problem['State']
        nc = self._num_cells()
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sw = _np.asarray(state['sW'], dtype=float).ravel()
        sg = _np.asarray(state['sG'], dtype=float).ravel()
        so = 1.0 - sw - sg
        # MRST's ShrinkageFactors state function evaluates PVTG with the
        # phase-status saturation flag from getCellStatusVO.  For VAPOIL,
        # oil-present cells use the saturated gas branch (rv = rvMax),
        # while gas-only cells use the undersaturated branch.  Calling the
        # raw PVT evaluator without this flag makes Norne's bG about 25 %
        # too large in oil-only cells and corrupts CNV/MB scaling.
        pW, pO, pG = self._phase_pressures(p, sw, sg, state.get('pcowScale'))
        pvt = self._phase_pvt_from_phase_pressures(
            pW, pO, pG,
            rs_override=state.get('rs'),
            rv_override=state.get('rv'),
            sG_override=sg,
            oil_saturated_override=(sg > 0.0),
            gas_saturated_override=(so > 0.0),
        )
        b = (pvt['bw'], pvt['bo'], pvt['bg'])
        rho_w, rho_o, rho_g = self._mrst_surface_densities()
        rho_s = (rho_w, rho_o, rho_g)
        pv = self._mrst_pore_volume(p)
        dt = float(problem['dt'])
        cnv, mb = [], []
        for iph in range(3):
            c, m = self.cnv_mb_from_residual(residual[iph * nc:(iph + 1) * nc], b[iph], rho_s[iph], pv, dt)
            cnv.append(c)
            mb.append(m)
        # getConvergenceValuesWells takes inf norms of the four separate
        # FacilityModel equation groups, all with tolerance 1/day.
        nw = max(0, (residual.size - 3 * nc) // 4)
        well_values = []
        if nw:
            start = 3 * nc
            for i in range(4):
                block = residual[start + i * nw:start + (i + 1) * nw]
                well_values.append(float(_np.max(_np.abs(block))) if block.size else 0.0)
        values = _np.asarray(cnv + mb + well_values, dtype=float)
        tolerances = _np.asarray(
            [self.toleranceCNV] * 3 + [self.toleranceMB] * 3 +
            [self.toleranceWellRate] * len(well_values), dtype=float
        )
        names = (['CNV_W', 'CNV_O', 'CNV_G', 'MB_W', 'MB_O', 'MB_G'] +
                 ['waterWells (perf)', 'oilWells (perf)',
                  'gasWells (perf)', 'closureWells (well)'][:len(well_values)])
        return values, tolerances, names

    def _convergence_values_mrst_generic_og(self, problem):
        """Active oil/gas subset of MRST ``getConvergenceValuesCNV`` (no
        water; mirrors :meth:`_convergence_values_mrst_generic_ow`)."""
        residual = _np.asarray(problem['Residuals'], dtype=float).ravel()
        state, nc = problem['State'], self._num_cells()
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sg = _np.asarray(state.get('sG', _np.zeros(nc)), dtype=float).ravel()
        so = 1.0 - sg
        zeros_nc = _np.zeros(nc)
        pW, pO, pG = self._phase_pressures(p, zeros_nc, sg)
        pvt = self._phase_pvt_from_phase_pressures(
            pW, pO, pG,
            rs_override=state.get('rs'), rv_override=state.get('rv'), sG_override=sg,
            oil_saturated_override=(sg > 0.0), gas_saturated_override=(so > 0.0),
        )
        _, rhoO, rhoG = self._mrst_surface_densities()
        pv, dt = self._mrst_pore_volume(p), float(problem['dt'])
        cnv, mb = [], []
        for iph, (b, rho) in enumerate(((pvt['bo'], rhoO), (pvt['bg'], rhoG))):
            c, m = self.cnv_mb_from_residual(residual[iph * nc:(iph + 1) * nc], b, rho, pv, dt)
            cnv.append(c)
            mb.append(m)
        nw = max(0, (residual.size - 2 * nc) // 3)
        well_values = []
        for i in range(3):
            block = residual[2 * nc + i * nw:2 * nc + (i + 1) * nw]
            if nw:
                well_values.append(float(_np.max(_np.abs(block))))
        values = _np.asarray(cnv + mb + well_values)
        tolerances = _np.asarray([self.toleranceCNV] * 2 + [self.toleranceMB] * 2 +
                                 [self.toleranceWellRate] * len(well_values))
        names = ['CNV_O', 'CNV_G', 'MB_O', 'MB_G', 'oilWells (perf)',
                 'gasWells (perf)', 'closureWells (well)']
        return values, tolerances, names[:len(values)]

    def _convergence_values_mrst_generic_ow(self, problem):
        """Active water/oil subset of MRST ``getConvergenceValuesCNV``."""
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
        nw = max(0, (residual.size - 2 * nc) // 3)
        well_values = []
        for i in range(3):
            block = residual[2 * nc + i * nw:2 * nc + (i + 1) * nw]
            if nw:
                well_values.append(float(_np.max(_np.abs(block))))
        values = _np.asarray(cnv + mb + well_values)
        tolerances = _np.asarray([self.toleranceCNV] * 2 + [self.toleranceMB] * 2 +
                                 [self.toleranceWellRate] * len(well_values))
        names = ['CNV_W', 'CNV_O', 'MB_W', 'MB_O', 'waterWells (perf)',
                 'oilWells (perf)', 'closureWells (well)']
        return values, tolerances, names[:len(values)]

    def updateState(self, state, problem, dx, drivingForces=None):
        nc = self._num_cells()
        x = _np.asarray(dx, dtype=float).ravel()
        if getattr(self, '_use_mrst_generic_assembly', False) and isinstance(problem, dict):
            if not self.gas:
                return self._update_state_mrst_generic_ow(state, problem, x, drivingForces)
            if not self.water:
                return self._update_state_mrst_generic_og(state, problem, x, drivingForces)
            return self._update_state_mrst_generic(state, problem, x, drivingForces)
        # MRST FacilityModel appends q_s/bhp primary variables after reservoir
        # variables. Split those increments before the reservoir update.
        if self.enable_facility_unknowns and isinstance(problem, dict):
            facility_names = list(problem.get('facilityPrimaryVariables', []))
            nfac = len(facility_names)
            if nfac and x.size > nfac:
                nres = x.size - nfac
                # Recurse on the reservoir-only portion with facility vars cleared
                # so that the inner call does not enter this branch again.
                problem_inner = dict(problem)
                problem_inner['facilityPrimaryVariables'] = []
                state = self.updateState(state, problem_inner, x[:nres], drivingForces)
                qnames = [n for n in facility_names if n.startswith('q')]
                bnames = [n for n in facility_names if n.startswith('bhp:')]
                qold = _np.asarray(state.get('facility_qs', _np.zeros(len(qnames))), dtype=float).ravel()
                bold = _np.asarray(state.get('facility_bhp', _np.zeros(len(bnames))), dtype=float).ravel()
                if qold.size != len(qnames):
                    qold = _np.zeros(len(qnames), dtype=float)
                if bold.size != len(bnames):
                    bold = _np.zeros(len(bnames), dtype=float)
                state['facility_qs'] = qold + x[nres:nres + len(qnames)]
                state['facility_bhp'] = bold + x[nres + len(qnames):]
                return state
        if x.size == nc:
            state['pressure'] = _np.asarray(state.get('pressure', 0.0), dtype=float).ravel() + x
        elif x.size == 2 * nc:
            p = _np.asarray(state.get('pressure', 0.0), dtype=float).ravel()
            sW = _np.asarray(state.get('sW', state.get('saturation', 0.0)), dtype=float).ravel()
            state['pressure'] = p + x[:nc]
            state['sW'] = _np.clip(sW + x[nc:], 0.0, 1.0)
        elif x.size == 3 * nc:
            p = _np.asarray(state.get('pressure', 0.0), dtype=float).ravel()
            sW = _np.asarray(state.get('sW', state.get('saturation', 0.0)), dtype=float).ravel()
            sG = _np.asarray(state.get('sG', 0.0), dtype=float).ravel()
            rs = _np.asarray(state.get('rs', 0.0), dtype=float).ravel()
            rv = _np.asarray(state.get('rv', 0.0), dtype=float).ravel()
            state['pressure'] = p + x[:nc]
            dsW = x[nc:2*nc]
            dx = x[2*nc:]
            status = problem.get('blackOilStatus', None)
            if status is None:
                gas_present = sG > _np.sqrt(_np.finfo(float).eps)
                oil_present = (1.0 - sW - sG) > _np.sqrt(_np.finfo(float).eps)
                st1 = oil_present & ~gas_present
                st2 = ~oil_present & gas_present
                st3 = oil_present & gas_present
            else:
                st1 = _np.asarray(status[0], dtype=bool)
                st2 = _np.asarray(status[1], dtype=bool)
                st3 = _np.asarray(status[2], dtype=bool)
            # MRST ThreePhaseBlackOilModel.updateState mapping:
            # x=Rs in st1, x=Rv in st2, x=Sg in st3.
            dSg = st3 * dx - st2 * dsW
            dRs = st1 * dx
            dRv = st2 * dx
            state['sW'] = _np.clip(sW + dsW, 0.0, 1.0)
            state['sG'] = _np.clip(sG + dSg, 0.0, 1.0)
            state['rs'] = _np.maximum(rs + dRs, 0.0)
            state['rv'] = _np.maximum(rv + dRv, 0.0)
            # Reconstruct oil saturation implicitly and apply MRST-style
            # normalization when saturation sum crosses one.
            s_sum = state['sW'] + state['sG']
            oversat = s_sum > 1.0
            if _np.any(oversat):
                state['sW'][oversat] *= 0.999 / _np.maximum(s_sum[oversat], 1e-12)
                state['sG'][oversat] *= 0.999 / _np.maximum(s_sum[oversat], 1e-12)
        else:
            raise ValueError('Expected dx of length %d, %d, or %d, got %d' % (nc, 2*nc, 3*nc, x.size))
        return state

    def _well_rates(self, drivingForces):
        well_sol = []
        qs_w = 0.0
        qs_o = 0.0
        for w in drivingForces.get('W', []):
            val = float(w.get('val', 0.0))
            sign = float(w.get('sign', -1))
            if w.get('type') == 'rate':
                # Keep positive phase magnitudes in wellSol, but signed values are
                # used in equation assembly.
                qWs = abs(val) * (1.0 if sign > 0 else 0.2)
                qOs = abs(val) * (0.0 if sign > 0 else 0.8)
                bhp = 100.0 + 10.0 * val * (1.0 if sign > 0 else -1.0)
            elif w.get('type') == 'bhp':
                qWs = max(0.0, 0.1 * (100.0 - val))
                qOs = max(0.0, 0.2 * (100.0 - val))
                bhp = float(val)
            else:
                qWs = 0.0
                qOs = 0.0
                bhp = 100.0
            qs_w += qWs
            qs_o += qOs
            well_sol.append({
                'name': w.get('name', ''),
                'status': bool(w.get('status', True)),
                'qWs': qWs,
                'qOs': qOs,
                'bhp': bhp,
                'sign': sign,
            })
        return qs_w, qs_o, well_sol

    def _well_cells(self, w):
        # Schedule conversion has already applied MRST's
        # make_cart_to_active/sub2ind mapping.  Prefer those perforations
        # verbatim instead of reconstructing them from WELSPECS.
        specified = w.get('cells', None)
        if specified is not None:
            cells = _np.asarray(specified, dtype=int).ravel()
            if cells.size:
                nc = self._num_cells()
                # The deck converter has already applied processWells'
                # unique(..., 'last') rule.  Preserve its ordering so each
                # perforation remains aligned with WI/dZ/cstatus.
                return [int(c) for c in cells if 0 <= int(c) < nc]
        nx, ny, nz = self._cart_dims()
        if nx <= 0 or ny <= 0 or nz <= 0:
            return []
        i = max(1, int(w.get('i', 1)))
        j = max(1, int(w.get('j', 1)))
        ks = w.get('k', None) or [1]
        cells = []
        for kz in ks:
            k = max(1, int(kz))
            if i > nx or j > ny or k > nz:
                continue
            # MATLAB's sub2ind(G.cartDims, i, j, k): i is contiguous.
            cart = (i - 1) + nx * (j - 1) + nx * ny * (k - 1)
            c2a = self.G.get('cart_to_active') if isinstance(self.G, dict) else None
            c = int(_np.asarray(c2a, dtype=int).ravel()[cart]) if c2a is not None else cart
            if c >= 0:
                cells.append(c)
        return sorted(set(cells))

    def _phase_pvt(self, p, rs_override=None, rv_override=None, sG_override=None,
                   oil_saturated_override=None, gas_saturated_override=None):
        """Evaluate PVT properties at pressure and optional Rs/Rv state values."""
        p = _np.asarray(p, dtype=float).ravel()
        n = p.size
        one = _np.ones(n, dtype=float)
        out = {
            'bw': one.copy(),
            'bo': one.copy(),
            'bg': one.copy(),
            'muw': _np.full(n, self.defaultViscosityW, dtype=float),
            'muo': _np.full(n, self.defaultViscosityO, dtype=float),
            'mug': _np.full(n, self.defaultViscosityG, dtype=float),
            'rs': _np.zeros(n, dtype=float),
            'rv': _np.zeros(n, dtype=float),
        }
        pvt = getattr(self, '_blackoil_pvt', None)
        if pvt is not None:
            try:
                vals = pvt.eval(
                    p,
                    rs_override=rs_override,
                    rv_override=rv_override,
                    oil_saturated_override=(
                        oil_saturated_override if oil_saturated_override is not None else
                        (None if sG_override is None else _np.asarray(sG_override, dtype=float).ravel() > 0.0)
                    ),
                    gas_saturated_override=gas_saturated_override,
                )
                for k in out.keys():
                    if k in vals and vals[k] is not None:
                        arr = _np.asarray(vals[k], dtype=float).ravel()
                        if arr.size == n:
                            out[k] = arr
                return out
            except Exception:
                pass
        # Fallback via model callables
        for key, fn_name in [('bw', 'bw'), ('bo', 'bo'), ('bg', 'bg'),
                              ('muw', 'mu_w'), ('muo', 'mu_o'), ('mug', 'mu_g'),
                              ('rs', 'rs'), ('rv', 'rv')]:
            fn = getattr(self, fn_name, None)
            if callable(fn):
                out[key] = _np.asarray(fn(p), dtype=float).ravel()
        return out

    def _get_relperm_scaling(self, nc, tables):
        """Build MRST ``initRelpermScaling`` drainage endpoint data.

        ``BaseRelativePermeability.m`` receives full Cartesian endpoint
        arrays through ``rock.krscale`` and indexes them with
        ``G.cells.indexMap``.  Retain exactly that active-cell mapping here;
        missing endpoint keywords are represented by the corresponding
        saturation-table point, as ``SaturationProperty.getPair`` does.
        """
        deck = getattr(self, 'inputdata', None)
        if not isinstance(deck, dict) or 'ENDSCALE' not in deck.get('RUNSPEC', {}):
            return None
        # The cache is keyed on the endpoint source as well as the cell
        # count: tuning an endpoint replaces ``rock.krscale``, and a cache
        # keyed on ``nc`` alone would keep handing back the values from
        # before the tuning for the rest of the run.
        krscale_key = _krscale_fingerprint(self.rock)
        cache = getattr(self, '_mrst_relperm_scaling', None)
        if (cache is not None and cache.get('nc') == int(nc)
                and cache.get('krscale_key') == krscale_key):
            return cache

        sw, sg = tables['swof'], tables['sgof']
        if sg is None:
            return None

        # ``assignSWOF.m``/``assignSGOF.m`` construct these table points.
        # ``_get_relperm_tables`` has already sliced the requested SATNUM
        # region out of any multi-region keyword, so these are the same
        # single-region curves the relperm evaluation interpolates -- the
        # scaling endpoints and the curve they scale must come from one
        # table, not two different slices of it.
        #
        # ``PRSTCore.ad_props.kr_points.get_kr_points`` computes the same
        # sets from the raw PROPS section, for every region, and is what
        # ``fluid.krPts`` and history matching's endpoint parameters use.
        # The two are kept separate -- this one has the guard below, and
        # reads the already-sliced table rather than re-slicing -- and
        # held level by tests/test_kr_points_parity.py.
        iw = _np.flatnonzero(sw[:, 1] == 0.0)
        ig = _np.flatnonzero(sg[:, 1] == 0.0)
        iow = _np.flatnonzero(sw[:, 2] <= _np.finfo(float).eps)
        iog = _np.flatnonzero(sg[:, 2] <= _np.finfo(float).eps)
        if not (iw.size and ig.size and iow.size and iog.size):
            return None
        pts = {
            'w': _np.array([sw[0, 0], sw[iw[-1], 0], sw[-1, 0], sw[-1, 1]], dtype=float),
            'ow': _np.array([0.0, 1.0 - sw[iow[0], 0], 1.0, sw[0, 2]], dtype=float),
            'og': _np.array([0.0, 1.0 - sg[iog[0], 0] - sw[0, 0], 1.0, sg[0, 2]], dtype=float),
            'g': _np.array([sg[0, 0], sg[ig[-1], 0], sg[-1, 0], sg[-1, 1]], dtype=float),
        }
        props = deck.get('PROPS', {})
        index_map = _np.arange(int(nc), dtype=int)
        try:
            index_map = _np.asarray(self.G.get('cells', {}).get('indexMap', index_map), dtype=int).ravel()
        except Exception:
            pass
        if index_map.size != int(nc):
            index_map = _np.arange(int(nc), dtype=int)

        def endpoint(name, default):
            raw = props.get(name)
            if raw is None:
                return _np.full(int(nc), float(default), dtype=float)
            try:
                value = _np.asarray(raw, dtype=float).ravel()
            except (TypeError, ValueError):
                return _np.full(int(nc), float(default), dtype=float)
            if value.size == 1:
                out = _np.full(int(nc), value[0], dtype=float)
            elif value.size > int(index_map.max(initial=-1)):
                out = value[index_map]
            elif value.size == int(nc):
                out = value.copy()
            else:
                out = _np.full(int(nc), float(default), dtype=float)
            return _np.where(_np.isfinite(out), out, float(default))

        names = {
            'w': ('SWL', 'SWCR', 'SWU', 'KRW'),
            'ow': ('SOWL', 'SOWCR', 'SOWU', 'KRO'),
            'og': ('SOGL', 'SOGCR', 'SOGU', 'KRO'),
            'g': ('SGL', 'SGCR', 'SGU', 'KRG'),
        }
        target = {
            phase: _np.column_stack([endpoint(name, pts[phase][i])
                                     for i, name in enumerate(phase_names)])
            for phase, phase_names in names.items()
        }

        # ``rock.krscale`` wins where it exists. MRST keeps the endpoints
        # in exactly one place -- ``initRelpermScaling`` fills
        # ``rock.krscale`` from the deck and everything downstream reads
        # it there -- while this builds them from PROPS instead. Both
        # stores then exist and disagree: ``imposeRelpermScaling`` and
        # ``ModelParameter``'s endpoint parameters write to krscale (that
        # is where their ``location`` points), and without this overlay
        # the residual goes on reading the deck's original values. A
        # tuned endpoint would change nothing and report success.
        krscale = (self.rock or {}).get('krscale') \
            if isinstance(self.rock, dict) else None
        drainage = (krscale or {}).get('drainage') or {}
        for phase, columns in drainage.items():
            if phase not in target or columns is None:
                continue
            columns = _np.atleast_2d(_np.asarray(columns, dtype=float))
            if columns.shape[0] != int(nc):
                continue
            width = min(columns.shape[1], target[phase].shape[1])
            block = columns[:, :width]
            # ``initRelpermScaling`` fills krscale with NaN and writes only
            # the columns whose keyword the deck actually has;
            # ``getConnateWater`` then does
            # ``swcon(nix) = fluid.krPts.w(regions(nix), 1)``. So an entry
            # is taken from krscale only where it *has* one -- copying the
            # NaNs through would replace every defaulted endpoint with a
            # NaN that propagates into every saturation it scales.
            target[phase] = target[phase].copy()
            take = _np.isfinite(block)
            target[phase][:, :width] = _np.where(take, block,
                                                 target[phase][:, :width])
        scalecrs = props.get('SCALECRS', [])
        if isinstance(scalecrs, _np.ndarray):
            scalecrs = scalecrs.ravel().tolist()
        if not isinstance(scalecrs, (list, tuple)):
            scalecrs = [scalecrs]
        three_point = bool(scalecrs) and str(scalecrs[0]).lower().startswith('y')
        cache = {'nc': int(nc), 'table': pts, 'target': target,
                 'points': 3 if three_point else 2,
                 'krscale_key': krscale_key}
        self._mrst_relperm_scaling = cache
        return cache

    def _endpoint(self, target, phase, column):
        """One column of the scaled endpoint table.

        Normally this is exactly ``target[phase][:, column]``. The adjoint
        seeds an endpoint as an AD variable to read ``dR/dendpoint`` out of
        the assembled Jacobian, and a column of AD cannot live inside a
        float array; ``_relperm_endpoint_seed`` carries it alongside
        instead. The attribute is absent on every model that is not being
        differentiated, so the ordinary path is unchanged.
        """
        seed = getattr(self, '_relperm_endpoint_seed', None)
        if seed:
            seeded = seed.get((phase, int(column)))
            if seeded is not None:
                return seeded
        return target[phase][:, column]

    def _relperm_scaling_parameters(self, phase, nc, tables):
        """Port the scaler construction in ``SaturationProperty.m``."""
        scale = self._get_relperm_scaling(nc, tables)
        if scale is None:
            return None
        table, target = scale['table'], scale['target']
        col = self._endpoint
        p = table[phase]
        if scale['points'] == 2:
            if phase in ('w', 'g'):
                su, SU = p[2], col(target, phase, 2)
            else:
                su = 1.0 - table['w'][0] - table['g'][0]
                SU = 1.0 - col(target, 'w', 0) - col(target, 'g', 0)
            scr, SCR = p[1], col(target, phase, 1)
            with _np.errstate(divide='ignore', invalid='ignore'):
                m = (su - scr) / (SU - SCR)
            c = scr - SCR * m
            return {'points': 2, 'm': m, 'c': c, 'p1': SCR, 'p2': SU,
                    'k': col(target, phase, 3) / p[3]}

        if phase == 'w':
            sr = 1.0 - table['ow'][1] - table['g'][0]
            SR = 1.0 - col(target, 'ow', 1) - col(target, 'g', 0)
            su, SU = p[2], col(target, phase, 2)
        elif phase == 'g':
            sr = 1.0 - table['og'][1] - table['w'][0]
            SR = 1.0 - col(target, 'og', 1) - col(target, 'w', 0)
            su, SU = p[2], col(target, phase, 2)
        elif phase == 'ow':
            sr = 1.0 - table['w'][1] - table['g'][0]
            SR = 1.0 - col(target, 'w', 1) - col(target, 'g', 0)
            su = 1.0 - table['w'][0] - table['g'][0]
            SU = 1.0 - col(target, 'w', 0) - col(target, 'g', 0)
        elif phase == 'og':
            sr = 1.0 - table['g'][1] - table['w'][0]
            SR = 1.0 - col(target, 'g', 1) - col(target, 'w', 0)
            su = 1.0 - table['g'][0] - table['w'][0]
            SU = 1.0 - col(target, 'g', 0) - col(target, 'w', 0)
        else:
            raise ValueError('Unknown relative-permeability phase %s' % phase)
        scr, SCR = p[1], col(target, phase, 1)
        with _np.errstate(divide='ignore', invalid='ignore'):
            m1 = (sr - scr) / (SR - SCR)
            m2 = (su - sr) / (SU - SR)
        c1 = scr - SCR * m1
        c2 = sr - SR * m2
        degenerate = _value(SU) <= _value(SR)
        m2 = _ad_where(degenerate, 0.0, m2)
        c2 = _ad_where(degenerate, 0.0, c2)
        return {'points': 3, 'm1': m1, 'c1': c1, 'm2': m2, 'c2': c2,
                'p1': SCR, 'p2': SR, 'p3': SU,
                'k': col(target, phase, 3) / p[3]}

    def _scale_relperm_saturation(self, saturation, phase, tables):
        saturation = _np.asarray(saturation, dtype=float).ravel()
        par = self._relperm_scaling_parameters(phase, saturation.size, tables)
        if par is None:
            return saturation, _np.ones(saturation.size, dtype=float)
        if par['points'] == 2:
            scaled = _np.where(saturation < _value(par['p1']), 0.0,
                               _np.where(saturation >= _value(par['p2']), 1.0,
                                         par['m'] * saturation + par['c']))
        else:
            ix1 = ((saturation >= _value(par['p1']))
                   & (saturation < _value(par['p2'])))
            ix2 = ((saturation >= _value(par['p2']))
                   & (saturation < _value(par['p3'])))
            ix3 = saturation >= _value(par['p3'])
            scaled = ix1 * (par['m1'] * saturation + par['c1'])
            scaled += ix2 * (par['m2'] * saturation + par['c2']) + ix3
        return scaled, par['k']

    def _scale_relperm_saturation_adi(self, saturation, phase, tables):
        par = self._relperm_scaling_parameters(phase, saturation.val.size, tables)
        if par is None:
            return saturation, _np.ones(saturation.val.size, dtype=float)
        if par['points'] == 2:
            inside = saturation * par['m'] + par['c']
            one = type(saturation).constant(_np.ones(saturation.val.size), saturation.nvar)
            scaled = _ad_select(saturation.val >= _value(par['p2']), one,
                                inside)
            scaled = _ad_select(saturation.val < _value(par['p1']),
                                _np.zeros(saturation.val.size), scaled)
        else:
            ix1 = ((saturation.val >= _value(par['p1']))
                   & (saturation.val < _value(par['p2'])))
            ix2 = ((saturation.val >= _value(par['p2']))
                   & (saturation.val < _value(par['p3'])))
            ix3 = saturation.val >= _value(par['p3'])
            scaled = _ad_select(ix1, saturation * par['m1'] + par['c1'],
                                _np.zeros(saturation.val.size))
            scaled = _ad_select(ix2, saturation * par['m2'] + par['c2'], scaled)
            one = type(saturation).constant(_np.ones(saturation.val.size), saturation.nvar)
            scaled = _ad_select(ix3, one, scaled)
        return scaled, par['k']

    def _relperm_connate_water(self, nc, tables):
        sw_table = tables['swof']
        scale = self._get_relperm_scaling(nc, tables)
        # For a two-phase curve without a gas table, use the tabulated
        # connate value.  Three-phase endpoint scaling returns SWL here.
        if scale is None:
            return _np.full(int(nc), float(sw_table[0, 0]), dtype=float)
        return self._endpoint(scale['target'], 'w', 0)

    def _scale_pcow_saturation(self, saturation, tables):
        saturation = _np.asarray(saturation, dtype=float).ravel()
        scale = self._get_relperm_scaling(saturation.size, tables)
        if scale is None:
            return saturation
        sw, SW = scale['table']['w'], scale['target']['w']
        pcscale = self._pcscale_drainage('w', saturation.size)
        if pcscale is not None:
            specified = _np.isfinite(pcscale[:, 0])
            SW = SW.copy()
            SW[specified, 0] = pcscale[specified, 0]
        with _np.errstate(divide='ignore', invalid='ignore'):
            return sw[0] + (saturation - SW[:, 0]) * (sw[2] - sw[0]) / (SW[:, 2] - SW[:, 0])

    def _scale_pcow_saturation_adi(self, saturation, tables):
        scale = self._get_relperm_scaling(saturation.val.size, tables)
        if scale is None:
            return saturation
        sw, target = scale['table']['w'], scale['target']
        swl = self._endpoint(target, 'w', 0)
        swu = self._endpoint(target, 'w', 2)
        pcscale = self._pcscale_drainage('w', saturation.val.size)
        if pcscale is not None:
            specified = _np.isfinite(pcscale[:, 0])
            swl = _ad_where(specified, pcscale[:, 0], swl)
        return sw[0] + (saturation - swl) * (sw[2] - sw[0]) / (swu - swl)

    def _scale_pcog_saturation(self, saturation, tables):
        """Apply MRST's SGLPC/endpoint remapping before evaluating PcOG."""
        saturation = _np.asarray(saturation, dtype=float).ravel()
        scale = self._get_relperm_scaling(saturation.size, tables)
        if scale is None:
            return saturation
        sg, SG = scale['table']['g'], scale['target']['g']
        pcscale = self._pcscale_drainage('g', saturation.size)
        if pcscale is not None:
            specified = _np.isfinite(pcscale[:, 0])
            SG = SG.copy()
            SG[specified, 0] = pcscale[specified, 0]
        with _np.errstate(divide='ignore', invalid='ignore'):
            return sg[0] + (saturation - SG[:, 0]) * (sg[2] - sg[0]) / (SG[:, 2] - SG[:, 0])

    def _scale_pcog_saturation_adi(self, saturation, tables):
        scale = self._get_relperm_scaling(saturation.val.size, tables)
        if scale is None:
            return saturation
        sg, target = scale['table']['g'], scale['target']
        sgl = self._endpoint(target, 'g', 0)
        sgu = self._endpoint(target, 'g', 2)
        pcscale = self._pcscale_drainage('g', saturation.val.size)
        if pcscale is not None:
            specified = _np.isfinite(pcscale[:, 0])
            sgl = _ad_where(specified, pcscale[:, 0], sgl)
        return sg[0] + (saturation - sgl) * (sg[2] - sg[0]) / (sgu - sgl)

    def _pcscale_drainage(self, phase, nc):
        """Return active-cell Pc endpoint scalers, if MRST created them."""
        if not isinstance(self.rock, dict):
            return None
        pcscale = self.rock.get('pcscale')
        if not isinstance(pcscale, dict):
            return None
        drainage = pcscale.get('drainage')
        if not isinstance(drainage, dict) or drainage.get(phase) is None:
            return None
        values = _np.asarray(drainage[phase], dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[0] != int(nc) or values.shape[1] < 2:
            return None
        return values[:, :2]

    def _pcscale_multiplier(self, phase, table, nc):
        """Port the PCW/PCG multiplier in BlackOilCapillaryPressure.m."""
        pcscale = self._pcscale_drainage(phase, nc)
        if pcscale is None:
            return None
        base = self._interp_relperm_table(table[:, 0], table[:, 3],
                                          _np.full(int(nc), table[0, 0]))
        with _np.errstate(divide='ignore', invalid='ignore'):
            multiplier = pcscale[:, 1] / base
        # The MATLAB source replaces NaN only (not Inf).
        multiplier[_np.isnan(multiplier)] = 1.0
        return multiplier

    def _relative_perm(self, sW, sG):
        """Evaluate SPE/ECLIPSE SWOF/SGOF tables as MRST does.

        This is a direct numerical port of ``assignSWOF.m``,
        ``assignSGOF.m`` and the three-phase branch of ``assignRelPerm.m``.
        The old quadratic fallback is retained only for programmatic models
        that have no deck saturation-function tables.
        """
        sW = _np.asarray(sW, dtype=float).ravel()
        sG = _np.asarray(sG, dtype=float).ravel()
        tables = self._get_relperm_tables()
        if tables is None:
            sW_c = _np.clip(sW, 0.0, 1.0)
            sG_c = _np.clip(sG, 0.0, 1.0)
            sO = _np.clip(1.0 - sW_c - sG_c, 0.0, 1.0)
            return sW_c * sW_c, sO * sO, sG_c * sG_c

        sw_table, sg_table = tables['swof'], tables['sgof']
        if sw_table is None:
            # relPermOG: two-phase oil/gas, no water phase at all.  krOG
            # was built (in assignSGOF.m) at coordinate ``1 - Sg - swcon``
            # with ``swcon = 0`` whenever no water table exists to source
            # it from, so the oil curve is evaluated at plain ``1 - Sg``.
            sg_eval, krg_max = self._scale_relperm_saturation(sG, 'g', tables)
            krg = krg_max * self._interp_relperm_table(sg_table[:, 0], sg_table[:, 1], sg_eval)
            so_g = 1.0 - sg_table[::-1, 0]
            so_eval, kro_max = self._scale_relperm_saturation(1.0 - sG, 'og', tables)
            kro = kro_max * self._interp_relperm_table(so_g, sg_table[::-1, 2], so_eval)
            return _np.zeros_like(sG), kro, krg
        swcon_table = float(sw_table[0, 0])
        swcon = self._relperm_connate_water(sW.size, tables)
        sw_eval, krw_max = self._scale_relperm_saturation(sW, 'w', tables)
        krw = krw_max * self._interp_relperm_table(sw_table[:, 0], sw_table[:, 1], sw_eval)
        # ``assignRelPerm.m`` selects relPermWO when no SGOF/krOG curve is
        # present: the SWOF oil curve is evaluated directly in So.
        if sg_table is None:
            so_w = 1.0 - sw_table[::-1, 0]
            so_eval, kro_max = self._scale_relperm_saturation(1.0 - sW, 'ow', tables)
            kro = kro_max * self._interp_relperm_table(so_w, sw_table[::-1, 2], so_eval)
            return krw, kro, _np.zeros_like(sW)
        sg_eval, krg_max = self._scale_relperm_saturation(sG, 'g', tables)
        krg = krg_max * self._interp_relperm_table(sg_table[:, 0], sg_table[:, 1], sg_eval)

        sO = 1.0 - sW - sG
        # assignSWOF reverses the oil table so its coordinate is So.
        so_w = 1.0 - sw_table[::-1, 0]
        sow_eval, krow_max = self._scale_relperm_saturation(sO, 'ow', tables)
        krow = krow_max * self._interp_relperm_table(so_w, sw_table[::-1, 2], sow_eval)
        # assignSGOF uses So = 1 - Sg - Swcon for the oil-gas curve.
        so_g = 1.0 - sg_table[::-1, 0] - swcon_table
        sog_eval, krog_max = self._scale_relperm_saturation(sO, 'og', tables)
        krog = krog_max * self._interp_relperm_table(so_g, sg_table[::-1, 2], sog_eval)

        # Exactly the WOG blending in assignRelPerm.m.  The 1e-5 adjustment
        # is MRST's guard against a 0/0 value at connate water saturation.
        swcon_eff = _np.minimum(swcon, sW - 1.0e-5)
        d = sG + sW - swcon_eff
        ww = (sW - swcon_eff) / d
        wg = 1.0 - ww
        kro = wg * krog + ww * krow
        return krw, kro, krg

    def _relative_perm_adi(self, sW, sG):
        """ADI port of the table operations in ``_relative_perm``."""
        tables = self._get_relperm_tables()
        if tables is None:
            raise NotImplementedError('Deck AD assembly requires SWOF/SGOF tables')
        sw_table, sg_table = tables['swof'], tables['sgof']

        def interp_extended(x, y, xi):
            # assignSWOF/assignSGOF call extendTab before interpTable.  The
            # duplicated endpoint one saturation unit beyond each table end
            # makes relperm and Pc constant outside the supplied range;
            # direct extrapolation would give non-MRST mobilities in SPE9's
            # Sw=1 water leg.
            x = _np.asarray(x, dtype=float).ravel()
            y = _np.asarray(y, dtype=float).ravel()
            return _ad_interp_linear(_np.r_[x[0] - 1.0, x, x[-1] + 1.0],
                                     _np.r_[y[0], y, y[-1]], xi)

        if sw_table is None:
            # relPermOG: two-phase oil/gas, no water phase at all (see the
            # non-ADI ``_relative_perm`` for the ``swcon = 0`` rationale).
            sg_eval, krg_max = self._scale_relperm_saturation_adi(sG, 'g', tables)
            krg = interp_extended(sg_table[:, 0], sg_table[:, 1], sg_eval) * krg_max
            so_g = 1.0 - sg_table[::-1, 0]
            so_eval, kro_max = self._scale_relperm_saturation_adi(1.0 - sG, 'og', tables)
            kro = interp_extended(so_g, sg_table[::-1, 2], so_eval) * kro_max
            return type(sG).constant(_np.zeros(sG.val.size), sG.nvar), kro, krg

        swcon_table = float(sw_table[0, 0])
        swcon = self._relperm_connate_water(sW.val.size, tables)
        sw_eval, krw_max = self._scale_relperm_saturation_adi(sW, 'w', tables)
        krw = interp_extended(sw_table[:, 0], sw_table[:, 1], sw_eval) * krw_max
        if sg_table is None:
            so_w = 1.0 - sw_table[::-1, 0]
            so_eval, kro_max = self._scale_relperm_saturation_adi(1.0 - sW, 'ow', tables)
            kro = interp_extended(so_w, sw_table[::-1, 2], so_eval) * kro_max
            return krw, kro, type(sW).constant(_np.zeros(sW.val.size), sW.nvar)
        sg_eval, krg_max = self._scale_relperm_saturation_adi(sG, 'g', tables)
        krg = interp_extended(sg_table[:, 0], sg_table[:, 1], sg_eval) * krg_max
        so = 1.0 - sW - sG
        so_w = 1.0 - sw_table[::-1, 0]
        sow_eval, krow_max = self._scale_relperm_saturation_adi(so, 'ow', tables)
        krow = interp_extended(so_w, sw_table[::-1, 2], sow_eval) * krow_max
        so_g = 1.0 - sg_table[::-1, 0] - swcon_table
        sog_eval, krog_max = self._scale_relperm_saturation_adi(so, 'og', tables)
        krog = interp_extended(so_g, sg_table[::-1, 2], sog_eval) * krog_max
        swcon_eff = _ad_minimum(swcon, sW - 1.0e-5)
        d = sG + sW - swcon_eff
        ww = (sW - swcon_eff) / d
        wg = 1.0 - ww
        kro = wg * krog + ww * krow
        return krw, kro, krg

    def _phase_pressures(self, pressure, sW, sG, pcow_scale=None):
        """Port ``BlackOilCapillaryPressure -> PhasePressures``.

        The primary pressure in a black-oil state is the oil pressure.
        MRST evaluates water PVT/viscosity at ``pO - pcOW(Sw)`` and gas
        PVT/viscosity at ``pO + pcOG(Sg)``; those phase pressures are also
        the pressures used in the corresponding phase-potential gradient.
        """
        p = _np.asarray(pressure, dtype=float).ravel()
        sW = _np.asarray(sW, dtype=float).ravel()
        sG = _np.asarray(sG, dtype=float).ravel()
        tables = self._get_relperm_tables()
        if tables is None:
            return p, p, p
        swof, sgof = tables['swof'], tables['sgof']
        if swof is None:
            pcow = _np.zeros_like(p)
        else:
            sW_pc = self._scale_pcow_saturation(sW, tables)
            pcow = self._interp_relperm_table(swof[:, 0], swof[:, 3], sW_pc)
        if pcow_scale is None and isinstance(self.rock, dict):
            pcow_scale = self.rock.get('pcowScale')
        if pcow_scale is not None and swof is not None:
            pcow = pcow * _np.asarray(pcow_scale, dtype=float).ravel()
        elif swof is not None:
            multiplier = self._pcscale_multiplier('w', swof, p.size)
            if multiplier is not None:
                pcow = pcow * multiplier
        if sgof is not None:
            sG_pc = self._scale_pcog_saturation(sG, tables)
            pcog = self._interp_relperm_table(sgof[:, 0], sgof[:, 3], sG_pc)
            multiplier = self._pcscale_multiplier('g', sgof, p.size)
            if multiplier is not None:
                pcog = pcog * multiplier
        else:
            pcog = _np.zeros_like(p)
        # BlackOilCapillaryPressure.m: water has ``-pcOW`` and gas has
        # ``+pcOG`` relative to the oil reference pressure.
        return p - pcow, p, p + pcog

    def _phase_pressures_adi(self, pressure, sW, sG, pcow_scale=None):
        """Sparse-ADI equivalent of :meth:`_phase_pressures`."""
        tables = self._get_relperm_tables()
        if tables is None:
            return pressure, pressure, pressure

        def interp_extended(x, y, xi):
            x = _np.asarray(x, dtype=float).ravel()
            y = _np.asarray(y, dtype=float).ravel()
            return _ad_interp_linear(_np.r_[x[0] - 1.0, x, x[-1] + 1.0],
                                     _np.r_[y[0], y, y[-1]], xi)

        swof, sgof = tables['swof'], tables['sgof']
        if swof is None:
            pcow = type(sG).constant(_np.zeros(sG.val.size), sG.nvar)
        else:
            sW_pc = self._scale_pcow_saturation_adi(sW, tables)
            pcow = interp_extended(swof[:, 0], swof[:, 3], sW_pc)
            if pcow_scale is None and isinstance(self.rock, dict):
                pcow_scale = self.rock.get('pcowScale')
            if pcow_scale is not None:
                pcow = pcow * _np.asarray(pcow_scale, dtype=float).ravel()
            else:
                multiplier = self._pcscale_multiplier('w', swof, sW.val.size)
                if multiplier is not None:
                    pcow = pcow * multiplier
        if sgof is not None:
            sG_pc = self._scale_pcog_saturation_adi(sG, tables)
            pcog = interp_extended(sgof[:, 0], sgof[:, 3], sG_pc)
            multiplier = self._pcscale_multiplier('g', sgof, sG.val.size)
            if multiplier is not None:
                pcog = pcog * multiplier
        else:
            pcog = type(sW).constant(_np.zeros(sW.val.size), sW.nvar)
        return pressure - pcow, pressure, pressure + pcog

    def _phase_pvt_from_phase_pressures(self, pW, pO, pG, rs_override=None,
                                        rv_override=None, sG_override=None,
                                        oil_saturated_override=None,
                                        gas_saturated_override=None):
        """Evaluate each PVT phase at MRST's corresponding phase pressure."""
        water = self._phase_pvt(pW, rs_override=rs_override,
                                rv_override=rv_override, sG_override=sG_override,
                                oil_saturated_override=oil_saturated_override,
                                gas_saturated_override=gas_saturated_override)
        oil = self._phase_pvt(pO, rs_override=rs_override,
                              rv_override=rv_override, sG_override=sG_override,
                              oil_saturated_override=oil_saturated_override,
                              gas_saturated_override=gas_saturated_override)
        gas = self._phase_pvt(pG, rs_override=rs_override,
                              rv_override=rv_override, sG_override=sG_override,
                              oil_saturated_override=oil_saturated_override,
                              gas_saturated_override=gas_saturated_override)
        return {
            'bw': water['bw'], 'muw': water['muw'],
            'bo': oil['bo'], 'muo': oil['muo'],
            'bg': gas['bg'], 'mug': gas['mug'],
            'rs': oil['rs'], 'rv': gas['rv'],
        }

    def _phase_pvt_from_phase_pressures_adi(self, pW, pO, pG,
                                            rs_override=None, rv_override=None,
                                            sG_override=None,
                                            oil_saturated_override=None,
                                            gas_saturated_override=None):
        """Sparse-ADI phase-pressure PVT evaluation used by MRST PVTProps."""
        water = self._phase_pvt_adi(pW, rs_override=rs_override,
                                    rv_override=rv_override, sG_override=sG_override,
                                    oil_saturated_override=oil_saturated_override,
                                    gas_saturated_override=gas_saturated_override)
        oil = self._phase_pvt_adi(pO, rs_override=rs_override,
                                  rv_override=rv_override, sG_override=sG_override,
                                  oil_saturated_override=oil_saturated_override,
                                  gas_saturated_override=gas_saturated_override)
        gas = self._phase_pvt_adi(pG, rs_override=rs_override,
                                  rv_override=rv_override, sG_override=sG_override,
                                  oil_saturated_override=oil_saturated_override,
                                  gas_saturated_override=gas_saturated_override)
        return {
            'bw': water['bw'], 'muw': water['muw'],
            'bo': oil['bo'], 'muo': oil['muo'],
            'bg': gas['bg'], 'mug': gas['mug'],
            'rs': oil['rs'], 'rv': gas['rv'],
        }

    @staticmethod
    def _interp_table(x, y, xi):
        """MRST interpTable: piecewise-linear interpolation/extrapolation."""
        x = _np.asarray(x, dtype=float).ravel()
        y = _np.asarray(y, dtype=float).ravel()
        xi = _np.asarray(xi, dtype=float)
        if x.size == 1:
            return _np.full_like(xi, y[0], dtype=float)
        order = _np.argsort(x)
        x, y = x[order], y[order]
        out = _np.interp(xi, x, y)
        below = xi < x[0]
        above = xi > x[-1]
        if _np.any(below):
            out[below] = y[0] + (xi[below] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        if _np.any(above):
            out[above] = y[-1] + (xi[above] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
        return out

    @staticmethod
    def _interp_relperm_table(x, y, xi):
        """``assignSWOF/assignSGOF`` interpolation after ``extendTab``.

        ``extendTab`` duplicates the first and last table row at coordinate
        minus/plus one.  Over physical saturations [0, 1] this is precisely
        endpoint-clamped linear interpolation.
        """
        x = _np.asarray(x, dtype=float).ravel()
        y = _np.asarray(y, dtype=float).ravel()
        order = _np.argsort(x)
        return _np.interp(_np.asarray(xi, dtype=float), x[order], y[order],
                          left=y[order][0], right=y[order][-1])

    def _saturation_region(self):
        """Return the 0-based SATNUM region whose saturation tables apply.

        MRST evaluates every saturation function through a per-cell region
        index (``f.krPts.(phase)(reg, index)``, ``interpReg``). PRSTCore has
        no per-cell region dispatch anywhere in the relperm/PVT path, so it
        can only honour a grid-uniform SATNUM. Rather than quietly using
        region 1 for a deck whose SATNUM genuinely varies -- which would
        give every cell outside that region the wrong curve -- report it.
        """
        deck = getattr(self, 'inputdata', None)
        regions = deck.get('REGIONS', {}) if isinstance(deck, dict) else {}
        satnum = regions.get('SATNUM')
        if satnum is None:
            return 0
        satnum = _np.asarray(satnum, dtype=float).ravel()
        satnum = satnum[_np.isfinite(satnum)].astype(int)
        if satnum.size == 0:
            return 0
        unique = _np.unique(satnum)
        if unique.size > 1:
            raise NotImplementedError(
                'SATNUM assigns %d different saturation regions; PRSTCore '
                'evaluates saturation functions with a single grid-uniform '
                'region and has no per-cell dispatch yet.' % unique.size)
        return max(int(unique[0]) - 1, 0)

    def _get_relperm_tables(self):
        cached = getattr(self, '_mrst_relperm_tables', None)
        if cached is not None:
            return cached
        deck = getattr(self, 'inputdata', None)
        props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
        from PRSTCore.ad_props.relperm_tables import build_swof_sgof_tables
        swof, sgof = build_swof_sgof_tables(props, region=self._saturation_region())
        # assignRelPerm.m's three-way dispatch (relPermWO / relPermOG /
        # relPermWOG) is keyed on which of krOG/krOW exist, i.e. exactly on
        # which of SWOF/SGOF families are present -- not on ``self.water``
        # directly, but a model can only need the water-oil curve if water
        # is active, and the gas-oil curve if gas is active.
        if self.water and swof is None:
            return None
        if self.gas and sgof is None:
            return None
        if not self.water:
            swof = None
        if not self.gas:
            sgof = None
        if swof is None and sgof is None:
            return None
        self._mrst_relperm_tables = {'swof': swof, 'sgof': sgof}
        return self._mrst_relperm_tables

    def _three_phase_mobility(self, p, sW, sG, rs_override=None, rv_override=None):
        """Compute reservoir phase mobility λ = kr / μ.

        MRST computes phase flux with mobility and applies the upstream
        shrinkage factor B separately in equationsBlackOil.m.
        """
        pvt = self._phase_pvt(p, rs_override=rs_override, rv_override=rv_override)
        krw, kro, krg = self._relative_perm(sW, sG)
        lamW = krw / _np.maximum(pvt['muw'], 1e-12)
        lamO = kro / _np.maximum(pvt['muo'], 1e-12)
        lamG = krg / _np.maximum(pvt['mug'], 1e-12)
        return lamW, lamO, lamG, pvt

    def _assemble_flux_divergence(self, p, lamW, lamO, lamG, pvt=None):
        """Compute surface-condition flux divergence and sparse pressure blocks.

        One pass over all faces at once.  The per-face Python loop this
        replaces did a handful of ``float()`` conversions and three
        four-element ``list.extend`` calls per face, so its cost was tens of
        microseconds times the face count -- a quarter of a second on SPE9's
        25665 faces, and several seconds on a field model, every time a
        residual is assembled.  Nothing here is inherently scalar: the
        upstream choice is a ``where``, and accumulating both signed
        contributions into the cells is a ``bincount``.
        """
        nc = self._num_cells()
        empty_triplet = (_np.zeros(0, dtype=int), _np.zeros(0, dtype=int), _np.zeros(0))

        ops = self.operators or {}
        N = _np.asarray(ops.get('N', _np.zeros((0, 2), dtype=int)), dtype=int)
        T = _np.asarray(ops.get('T', _np.zeros((0,), dtype=float)), dtype=float).ravel()
        if N.size == 0 or T.size == 0:
            zeros = _np.zeros((nc,), dtype=float)
            return (zeros, zeros.copy(), zeros.copy(),
                    empty_triplet, empty_triplet, empty_triplet)

        # MRST's N is one-based, and a face naming a cell outside the active
        # set is skipped rather than clamped.
        nface = min(N.shape[0], T.size)
        c1 = N[:nface, 0] - 1
        c2 = N[:nface, 1] - 1
        keep = (c1 >= 0) & (c2 >= 0) & (c1 < nc) & (c2 < nc)
        if not _np.all(keep):
            c1, c2 = c1[keep], c2[keep]
            tf = T[:nface][keep]
        else:
            tf = T[:nface]

        p = _np.asarray(p, dtype=float).ravel()
        dp = p[c1] - p[c2]
        # ``dp >= 0`` selects c1, matching the loop's tie-breaking at dp == 0.
        upstream = _np.where(dp >= 0.0, c1, c2)

        def mobility(lam, key):
            # MRST: bOvO/bGvG/bWvW = faceUpstr(B_phase) .* v_phase.
            # pvt exposes MRST shrinkage factors b = 1/B.
            b_up = (_np.asarray(pvt[key], dtype=float).ravel()[upstream]
                    if pvt is not None else 1.0)
            return tf * _np.asarray(lam, dtype=float).ravel()[upstream] * b_up

        gw = mobility(lamW, 'bw')
        go = mobility(lamO, 'bo')
        gg = mobility(lamG, 'bg')

        def divergence(g):
            flux = g * dp
            return (_np.bincount(c1, weights=flux, minlength=nc) -
                    _np.bincount(c2, weights=flux, minlength=nc))

        def triplet(g):
            # d(flux)/dp contributes (+g, -g) on row c1 and (-g, +g) on row
            # c2.  Zero conductivities are dropped so the triplet matches the
            # loop's ``if g != 0`` exactly, rather than carrying explicit
            # zeros into the assembled matrix.
            nonzero = g != 0.0
            gn = g[nonzero]
            a, b = c1[nonzero], c2[nonzero]
            rows = _np.concatenate([a, a, b, b])
            cols = _np.concatenate([a, b, a, b])
            vals = _np.concatenate([gn, -gn, -gn, gn])
            return rows, cols, vals

        return (divergence(gw), divergence(go), divergence(gg),
                triplet(gw), triplet(go), triplet(gg))

    def _well_sources(self, p, sW, sG, drivingForces, lamW, lamO, lamG, pvt):
        """Compute three-phase well source terms and their pressure derivatives."""
        nc = self._num_cells()
        src_w = _np.zeros((nc,), dtype=float)
        src_o = _np.zeros((nc,), dtype=float)
        src_g = _np.zeros((nc,), dtype=float)
        dsrcw_dp = _np.zeros((nc,), dtype=float)
        dsrco_dp = _np.zeros((nc,), dtype=float)
        dsrcg_dp = _np.zeros((nc,), dtype=float)
        well_sol = []

        for w in drivingForces.get('W', []):
            if not bool(w.get('status', True)):
                continue
            cells = self._well_cells(w)
            if not cells:
                continue
            nperf = float(len(cells))
            sign = float(w.get('sign', -1.0))
            wtype = str(w.get('type', 'rate')).lower()
            val = float(w.get('val', 0.0))
            
            # Handle WI: can be scalar, list, or array (one per completion).
            # A numpy array reaches here for e.g. a horizontal NWM well
            # (NearWellboreModel.getSimSchedule sets W['WI'] from
            # getWellCellPara, one entry per wellbore-grid perforation);
            # isinstance(..., (list, tuple)) alone misses that case.
            wi_data = w.get('WI', self.defaultWI)
            wi_arr = _np.atleast_1d(wi_data)
            if wi_arr.size == len(cells):
                wi_list = [float(x) for x in wi_arr]
            else:
                wi_list = [float(wi_arr[0])] * len(cells)
            
            phase = str(w.get('phase', 'OIL')).upper()
            qw_tot, qo_tot, qg_tot = 0.0, 0.0, 0.0
            bhp = float(w.get('bhp', val if wtype == 'bhp' else 0.0))

            if wtype == 'rate' and sign > 0:
                # Injector: distribute equally to all perforations
                qin = val / nperf
                for c in cells:
                    if phase == 'WATER':
                        src_w[c] += qin; qw_tot += qin
                    elif phase == 'GAS':
                        src_g[c] += qin; qg_tot += qin
                    else:
                        src_o[c] += qin; qo_tot += qin
            elif wtype in ('rate', 'lrat', 'orat', 'wrat', 'grat'):
                # Producer surface-rate controls (WCONPROD RATE/LRAT/ORAT/WRAT/GRAT):
                # fractional-flow split with mobility-pressure coupling.
                # 'rate'/'lrat' target the combined liquid (water+oil) rate;
                # 'orat'/'wrat'/'grat' target one phase's rate directly, with
                # the other phases following at their current mobility ratio
                # (i.e. total withdrawal = target / fraction-of-target-phase).
                # Distribute equally to all perforations (can be weighted by
                # WI for more accuracy).
                target = abs(val) / nperf
                for i, c in enumerate(cells):
                    lt = float(lamW[c] + lamO[c] + lamG[c])
                    fw = float(lamW[c] / lt) if lt > 0.0 else 0.0
                    fo = float(lamO[c] / lt) if lt > 0.0 else 1.0
                    fg = float(lamG[c] / lt) if lt > 0.0 else 0.0
                    if wtype in ('rate', 'lrat'):
                        fliq = fw + fo
                        qout = (target / fliq) if fliq > 0.0 else 0.0
                    elif wtype == 'orat':
                        qout = (target / fo) if fo > 0.0 else 0.0
                    elif wtype == 'wrat':
                        qout = (target / fw) if fw > 0.0 else 0.0
                    else:  # 'grat'
                        qout = (target / fg) if fg > 0.0 else 0.0
                    qw = -fw * qout; qo = -fo * qout; qg = -fg * qout
                    src_w[c] += qw
                    src_o[c] += qo
                    src_g[c] += qg
                    qw_tot += abs(qw); qo_tot += abs(qo); qg_tot += abs(qg)
                    # Pressure coupling for the rate-well's *own* Jacobian
                    # (d(q_phase)/dp via fractional-flow sensitivity):
                    # uses this perforation's own well index as the
                    # coupling scale, matching the BHP-well branch below.
                    # Previously this used a fixed
                    # ``self.defaultWI * 0.01`` (1e-5) placeholder
                    # regardless of the well's actual WI -- for NWM.data's
                    # horizontal producer (WI ~1e-12, seven orders of
                    # magnitude smaller) that placeholder swamped the real
                    # pressure sensitivity and stalled Newton's convergence
                    # rate to linear well below the point a normalized
                    # residual could reach a tight tolerance.
                    wi_perf = wi_list[i] if i < len(wi_list) else wi_list[0]
                    dsrcw_dp[c] += -fw * wi_perf
                    dsrco_dp[c] += -fo * wi_perf
                    dsrcg_dp[c] += -fg * wi_perf
            elif wtype == 'bhp':
                for i, c in enumerate(cells):
                    wi_perf = wi_list[i] if i < len(wi_list) else wi_list[0]
                    qt = wi_perf * (bhp - float(p[c]))
                    if sign > 0:
                        if phase == 'WATER':
                            src_w[c] += qt * float(pvt['bw'][c]); qw_tot += abs(qt)
                            dsrcw_dp[c] += -wi_perf
                        elif phase == 'GAS':
                            src_g[c] += qt * float(pvt['bg'][c]); qg_tot += abs(qt)
                            dsrcg_dp[c] += -wi_perf
                        else:
                            src_o[c] += qt * float(pvt['bo'][c]); qo_tot += abs(qt)
                            dsrco_dp[c] += -wi_perf
                    else:
                        lt = float(lamW[c] + lamO[c] + lamG[c])
                        fw = float(lamW[c] / lt) if lt > 0.0 else 0.0
                        fo = float(lamO[c] / lt) if lt > 0.0 else 1.0
                        fg = float(lamG[c] / lt) if lt > 0.0 else 0.0
                        src_w[c] += fw * qt * float(pvt['bw'][c]); qw_tot += abs(fw * qt)
                        src_o[c] += fo * qt * float(pvt['bo'][c]); qo_tot += abs(fo * qt)
                        src_g[c] += fg * qt * float(pvt['bg'][c]); qg_tot += abs(fg * qt)
                        dsrcw_dp[c] += fw * (-wi_perf)
                        dsrco_dp[c] += fo * (-wi_perf)
                        dsrcg_dp[c] += fg * (-wi_perf)
            else:
                continue

            well_sol.append({
                'name': w.get('name', ''),
                'status': True,
                'qWs': float(qw_tot), 'qOs': float(qo_tot), 'qGs': float(qg_tot),
                'bhp': float(bhp), 'sign': sign,
            })
        return src_w, src_o, src_g, dsrcw_dp, dsrco_dp, dsrcg_dp, well_sol

    def _facility_layout(self, drivingForces):
        """Return MRST-style per-well q_s/bhp primary-variable layout."""
        wells = [w for w in drivingForces.get('W', []) if bool(w.get('status', True))]
        phases = ['W', 'O'] + (['G'] if self.gas else [])
        names = []
        for w in wells:
            name = str(w.get('name', 'WELL'))
            for ph in phases:
                names.append(f'q{ph}s:{name}')
            names.append(f'bhp:{name}')
        return wells, phases, names

    def _augment_facility_system(self, residuals, jacobian, names, types,
                                 state, well_sol, drivingForces, nc):
        """Append q_s/bhp closure and control equations like GenericFacilityModel."""
        if not self.enable_facility_unknowns:
            return residuals, jacobian, names, types, state
        import scipy.sparse as _sp
        wells, phases, primary_names = self._facility_layout(drivingForces)
        if not wells:
            return residuals, jacobian, names, types, state
        nres = int(residuals.size)
        nwell = len(primary_names)
        qvals = _np.asarray(state.get('facility_qs', _np.zeros(len(wells)*len(phases))), dtype=float).ravel()
        bvals = _np.asarray(state.get('facility_bhp', _np.zeros(len(wells))), dtype=float).ravel()
        if qvals.size != len(wells)*len(phases):
            qvals = _np.zeros(len(wells)*len(phases), dtype=float)
        if bvals.size != len(wells):
            bvals = _np.zeros(len(wells), dtype=float)
        # Build well target vector indexed by (well, phase).
        # phase_order maps 'W'->0, 'O'->1, 'G'->2 for the active phases.
        nph = len(phases)
        well_targets_q = _np.zeros((len(wells), nph), dtype=float)
        well_targets_bhp = _np.zeros(len(wells), dtype=float)
        for iw, w in enumerate(wells):
            wn = str(w.get('name', 'WELL'))
            sign = float(w.get('sign', -1.0))
            ctrl = str(w.get('control', w.get('type', 'RATE'))).upper()
            val = float(w.get('val', 0.0))
            if ctrl == 'BHP' or str(w.get('type', '')).lower() == 'bhp':
                well_targets_bhp[iw] = val
            else:
                # ORAT→ oil, WRAT→ water, GRAT→ gas, RATE→ total
                for ip, ph in enumerate(phases):
                    if ctrl == 'ORAT' and ph == 'O':
                        well_targets_q[iw, ip] = abs(val)
                    elif ctrl == 'WRAT' and ph == 'W':
                        well_targets_q[iw, ip] = abs(val)
                    elif ctrl == 'GRAT' and ph == 'G':
                        well_targets_q[iw, ip] = abs(val)
                    elif ctrl in ('RATE', 'LRAT', 'RESV') and ph == 'O':
                        # Approximate: apply entire rate to oil for now.
                        well_targets_q[iw, ip] = abs(val)

        fres = []
        f_names = []
        f_types = []
        col = 0
        for iw, w in enumerate(wells):
            wn = str(w.get('name', 'WELL'))
            ctrl = str(w.get('control', w.get('type', 'RATE'))).upper()
            bhp_ctrl = ctrl == 'BHP' or str(w.get('type', '')).lower() == 'bhp'
            for ip, ph in enumerate(phases):
                target = well_targets_q[iw, ip]
                # Closure: q_s{ph} - target = 0 for rate controls.
                # For BHP wells, target is zero (rate floats freely) and
                # the BHP closure equation determines the actual rate.
                fres.append(float(qvals[col]) - target)
                f_names.append(f'q{ph} closure:{wn}')
                f_types.append('well')
                col += 1
            if bhp_ctrl:
                # BHP closure: bhp - target_bhp = 0
                fres.append(float(bvals[iw]) - well_targets_bhp[iw])
                f_names.append(f'BHP control:{wn}')
            else:
                # Rate control: bhp - bhp_state (trivial; bhp floats)
                fres.append(float(bvals[iw]))
                f_names.append(f'{ctrl} control:{wn}')
            f_types.append('well')
        fres = _np.asarray(fres, dtype=float)

        # Build sparse coupling blocks between reservoir equations and q_s.
        # Reservoir residual = acc + div - src.  Here src is provided by the
        # _well_sources using current q_s from state.  The cross-block
        # d(res_reservoir)/d(q_s) = -identity mapped through cell-perforation.
        # Approximate: one q_s per well is inserted into one representative cell.
        ph_idx = {ph: ip for ip, ph in enumerate(phases)}
        nph = len(phases)

        # dres_reservoir / d(q_s): for each well's qW/qO/qG, the source is
        # applied to its perforation cells.  Approximate by the mean cell index.
        Jrw_rows = []
        Jrw_cols = []
        Jrw_vals = []
        col = 0
        for iw, w in enumerate(wells):
            cells = self._well_cells(w)
            if not cells:
                col += nph + 1
                continue
            sign = float(w.get('sign', -1.0))
            for ip, ph in enumerate(phases):
                eq_offset = {'W': 0, 'O': nc, 'G': 2*nc if self.gas else nc}.get(ph, 0)
                for c in cells:
                    nperf = max(1, len(cells))
                    Jrw_rows.append(eq_offset + c)
                    Jrw_cols.append(col)
                    # d(res)/d(q_s) = -sign/nperf  (source = sign * q_s / nperf)
                    Jrw_vals.append(-sign / nperf)
                col += 1
            col += 1  # skip bhp column

        if _sp.issparse(jacobian):
            n_qbhp = nwell
            Jrw = _sp.csr_matrix((Jrw_vals, (Jrw_rows, Jrw_cols)),
                                  shape=(nres, n_qbhp))
            Jfw = _sp.eye(n_qbhp, format='csr')
            jacobian = _sp.bmat([[jacobian, Jrw],
                                 [_sp.csr_matrix((n_qbhp, nres)), Jfw]], format='csr')
        else:
            n_qbhp = nwell
            Jrw_dense = _np.zeros((nres, n_qbhp), dtype=float)
            for r, c, v in zip(Jrw_rows, Jrw_cols, Jrw_vals):
                Jrw_dense[r, c] += v
            jacobian = _np.block([[jacobian, Jrw_dense],
                                  [_np.zeros((n_qbhp, nres)), _np.eye(n_qbhp)]])
        state['facility_qs'] = qvals
        state['facility_bhp'] = bvals
        state['facility_primary_variables'] = primary_names
        return (_np.concatenate([residuals, fres]), jacobian,
                list(names) + f_names, list(types) + f_types, state)

    def stepFunction(self, state, state0, dt, drivingForces=None,
                     linsolver=None, nonlinsolver=None, iteration=1, **kwargs):
        state = deepcopy(state)
        state0 = self.validateState(state0)
        if drivingForces is None:
            drivingForces = {}
        state['time'] = float(state0.get('time', 0.0)) + float(dt)

        if linsolver is None or nonlinsolver is None:
            qs_w, qs_o, well_sol = self._well_rates(drivingForces)
            state['wellSol'] = well_sol
            if 'pressure' in state and state['pressure'] is not None:
                pv = self._average_porevolume() or 1.0
                dpressure = (qs_w - qs_o) * float(dt) / max(pv, 1.0)
                state['pressure'] = _np.asarray(state['pressure'], dtype=float) + dpressure
            if 'sW' in state:
                sw = _np.asarray(state['sW'], dtype=float)
                adjustment = min(1.0, max(-1.0, qs_w * float(dt) / max(self._average_porevolume(), 1.0)))
                state['sW'] = _np.clip(sw + adjustment, 0.0, 1.0)
            report = self.makeStepReport(
                Converged=True,
                Iterations=1,
                EarlyStop=False,
                Time=state['time'],
                StepSize=float(dt),
                Residuals=_np.zeros(self._num_cells() * 3, dtype=float),
                ResidualsConverged=_np.ones(self._num_cells() * 3, dtype=bool),
            )
            return state, report

        problem, state = self.get_equations(state0, state, dt, drivingForces,
                                           iteration=iteration, **kwargs)
        state['wellSol'] = problem.get('wellSol', [])
        if self.stepFunctionIsLinear:
            problem, state = self.get_equations(state0, state, dt, drivingForces,
                                               ResOnly=True,
                                               iteration=iteration + 1,
                                               **kwargs)
            state = self.reduceState(state, True)
            convergence, values, resnames = self.checkConvergence(problem)
            modelConverged = all(convergence)
            isConverged = modelConverged
            report = self.makeStepReport(
                LinearSolver={},
                UpdateState={},
                Failure=not modelConverged,
                FailureMsg='' if modelConverged else 'Linear step failed to converge',
                Converged=isConverged,
                Solved=not isConverged,
                Residuals=values,
                ResidualsConverged=convergence,
                Iterations=1,
                Time=state['time'],
                StepSize=float(dt),
            )
            return state, report

        iterationcount = iteration
        updates = 0
        convergence, values, resnames = self.checkConvergence(problem)
        if hasattr(nonlinsolver, 'printNewtonTrace'):
            nonlinsolver.printNewtonTrace(0, values, convergence, {}, resnames)
        doneMinIts = updates >= nonlinsolver.minIterations
        failure = False
        failureMsg = ''
        linearReport = {}
        updateReport = {}
        stabilizeReport = {}
        residualHistory = [_np.asarray(values, dtype=float).copy()]
        convergenceHistory = [_np.asarray(convergence, dtype=bool).copy()]
        linearReports = []

        while (not all(convergence) or not doneMinIts) and updates < nonlinsolver.maxIterations:
            # MRST's PhysicalModel.stepFunction computes convergence values
            # before applying the Newton update and returns those values in
            # stepReport.Residuals.  NonLinearSolver.solveMinistep updates
            # the relaxation history from that report after the state has
            # been advanced.  Keep a copy of the pre-update values so that
            # damping/oscillation detection follows the same sequence even
            # though this compact Python stepFunction performs the loop
            # internally.
            # PhysicalModel.stepFunction tags the problem with the current
            # nonlinear iteration so the line search reassembles the same
            # system (problem.iterationNo).
            problem['iterationNo'] = iterationcount
            residual_before_update = values
            convergence_before_update = convergence
            try:
                dx, _, linearReport = linsolver.solveLinearProblem(problem, self)
                if isinstance(linearReport, dict):
                    linearReports.append(dict(linearReport))
                else:
                    linearReports.append({})
                if not _np.all(_np.isfinite(dx)):
                    failure = True
                    failureMsg = 'Linear solver produced non-finite values.'
                    break
            except Exception as ex:
                failure = True
                failureMsg = str(ex)
                break
            # Direct port of PhysicalModel.stepFunction: stabilize after
            # the linear solve and immediately before updateState, then --
            # only when the solver is struggling and line search is on --
            # enter the residual bisection instead of the plain update.
            if hasattr(nonlinsolver, 'stabilizeNewtonIncrements'):
                dx, stabilizeReport = nonlinsolver.stabilizeNewtonIncrements(dx)
            self._print_update_increment_trace(dx, problem, nonlinsolver)
            is_struggling = (
                bool(getattr(nonlinsolver, 'alwaysUseStabilization', False))
                or bool(getattr(nonlinsolver, 'convergenceIssues', False)))
            if is_struggling and bool(getattr(nonlinsolver, 'useLinesearch', False)) \
                    and hasattr(nonlinsolver, 'applyLinesearch'):
                state, updateReport, ls_report = nonlinsolver.applyLinesearch(
                    self, state0, state, problem, dx, drivingForces, **kwargs)
                stabilizeReport['linesearch'] = ls_report
            else:
                state = self.updateState(state, problem, dx, drivingForces)
            iterationcount += 1
            updates += 1
            problem, state = self.get_equations(state0, state, dt, drivingForces,
                                               iteration=iterationcount, **kwargs)
            state['wellSol'] = problem.get('wellSol', [])
            convergence, values, resnames = self.checkConvergence(problem)
            # NonLinearSolver.solveMinistep (enforceResidualDecrease): a
            # mini-step that stops making progress is abandoned so the outer
            # loop can cut the timestep rather than burn Newton iterations.
            if updates >= 1 and getattr(nonlinsolver, 'enforceResidualDecrease', False):
                if bool(_np.all(_np.asarray(values, dtype=float) >=
                                _np.asarray(residual_before_update, dtype=float))):
                    break
            residualHistory.append(_np.asarray(values, dtype=float).copy())
            convergenceHistory.append(_np.asarray(convergence, dtype=bool).copy())
            doneMinIts = updates >= nonlinsolver.minIterations
            if hasattr(nonlinsolver, 'printNewtonTrace'):
                nonlinsolver.printNewtonTrace(updates, values, convergence, linearReport, resnames)
            # ``NonLinearSolver.solveMinistep`` stores the residual reported
            # by the just-completed ``model.stepFunction`` call.  In MRST
            # that residual is evaluated before the Newton update, while
            # the returned state has already been updated.  This slightly
            # counter-intuitive ordering controls damping for the following
            # increment and matters for oscillatory NORNE-style steps.
            if hasattr(nonlinsolver, 'updateRelaxationFromResidual'):
                nonlinsolver.updateRelaxationFromResidual(
                    residual_before_update, convergence_before_update
                )

        # PhysicalModel.stepFunction (lines 771--775): when the Newton
        # iteration budget is exhausted and the solver's acceptanceFactor is
        # relaxed (default 1 == disabled), re-evaluate convergence against
        # ``acceptanceFactor*tol`` so an "almost converged" mini-step is
        # accepted instead of cut and retried from half the timestep.
        outOfIterations = updates >= nonlinsolver.maxIterations
        modelConverged = all(convergence)
        acceptance = float(getattr(nonlinsolver, 'acceptanceFactor', 1.0))
        if outOfIterations and acceptance != 1.0:
            values, tolerances, resnames = self.getConvergenceValues(problem)
            modelConverged = bool(_np.all(
                _np.asarray(values, dtype=float) <
                acceptance * _np.asarray(tolerances, dtype=float)))
        isConverged = modelConverged and doneMinIts
        report = self.makeStepReport(
            LinearSolver=linearReport,
            UpdateState=updateReport,
            StabilizeReport=stabilizeReport,
            # In MRST an exhausted Newton iteration budget is a normal
            # non-convergence that triggers timestep cutting; ``Failure``
            # is reserved for a failed linear solve or non-finite update.
            Failure=failure,
            FailureMsg=failureMsg if failure else ('' if isConverged else 'Nonlinear iteration did not converge'),
            Converged=isConverged,
            Solved=not isConverged,
            Residuals=values,
            ResidualsConverged=convergence,
            ResidualHistory=residualHistory,
            ResidualConvergenceHistory=convergenceHistory,
            LinearSolverReports=linearReports,
            Iterations=updates,
            Time=state['time'],
            StepSize=float(dt),
        )
        return state, report

    def _print_update_increment_trace(self, dx, problem, nonlinsolver):
        """Verbose MRST-parity diagnostics for one Newton increment."""
        if nonlinsolver is None or not getattr(nonlinsolver, 'verbose', False):
            return
        try:
            x = _np.asarray(dx, dtype=float).ravel()
            nc = self._num_cells()
            if nc <= 0 or x.size < nc:
                return

            def maxabs(v):
                v = _np.asarray(v, dtype=float).ravel()
                return float(_np.max(_np.abs(v))) if v.size else 0.0

            parts = [f"relax={float(getattr(nonlinsolver, 'relaxationParameter', 1.0)):.3g}"]
            if getattr(self, '_use_mrst_generic_assembly', False) and isinstance(problem, dict):
                if self.gas and x.size >= 3 * nc:
                    parts.extend([
                        f"|dp|={maxabs(x[:nc]):.3e}",
                        f"|dsW|={maxabs(x[nc:2*nc]):.3e}",
                        f"|dx|={maxabs(x[2*nc:3*nc]):.3e}",
                    ])
                    nw = max(0, (x.size - 3 * nc) // 4)
                    if nw:
                        start = 3 * nc
                        parts.extend([
                            f"|dqW|={maxabs(x[start:start+nw]):.3e}",
                            f"|dqO|={maxabs(x[start+nw:start+2*nw]):.3e}",
                            f"|dqG|={maxabs(x[start+2*nw:start+3*nw]):.3e}",
                            f"|dBHP|={maxabs(x[start+3*nw:start+4*nw]):.3e}",
                        ])
                elif x.size >= 2 * nc:
                    parts.extend([
                        f"|dp|={maxabs(x[:nc]):.3e}",
                        f"|dsW|={maxabs(x[nc:2*nc]):.3e}",
                    ])
                    nw = max(0, (x.size - 2 * nc) // 3)
                    if nw:
                        start = 2 * nc
                        parts.extend([
                            f"|dqW|={maxabs(x[start:start+nw]):.3e}",
                            f"|dqO|={maxabs(x[start+nw:start+2*nw]):.3e}",
                            f"|dBHP|={maxabs(x[start+2*nw:start+3*nw]):.3e}",
                        ])
            else:
                parts.append(f"|dx|={maxabs(x):.3e}")
            print("      update: " + ", ".join(parts), flush=True)
        except Exception:
            # Diagnostics must never affect nonlinear solves.
            return

    def getAdjointEquations(self, state0, state, dt, forces, **kwargs):
        """Port of ``PhysicalModel.getAdjointEquations``.

        MRST allows a model to assemble different equations for the
        adjoint than for the forward solve -- hysteresis is the case it
        exists for -- and forwards to ``getEquations`` when it does not.
        Black oil does not, so this is the forward assembly with
        ``reverseMode`` passed through.
        """
        return self.get_equations(state0, state, dt, forces, **kwargs)

    def getReverseStateAD(self, state, *args, **kwargs):
        """Port of ``PhysicalModel.getReverseStateAD``, which is
        ``getStateAD`` under another name."""
        return self.getStateAD(state, *args, **kwargs)

    def get_equations(self, state0, state, dt, drivingForces=None, **kwargs):
        """Black-oil conservation equations with adaptive phase count.

        When self.gas is False, uses the 2-equation oil-water model.
        When self.water is False (and gas is active), uses the 2-equation
        oil-gas model (with Rs/Rv). Otherwise uses the full 3-equation
        black-oil model.
        """
        state = self.validateState(state)
        state0 = self.validateState(state0)
        if getattr(self, '_use_mrst_generic_assembly', False):
            if not self.gas:
                return self._get_equations_mrst_generic_ow(state0, state, dt, drivingForces, **kwargs)
            if not self.water:
                return self._get_equations_mrst_generic_og(state0, state, dt, drivingForces, **kwargs)
            return self._get_equations_mrst_generic(state0, state, dt, drivingForces, **kwargs)
        nc = self._num_cells()

        if not self.gas:
            return self._get_equations_ow(state0, state, dt, drivingForces, **kwargs)
        return self._get_equations_3ph(state0, state, dt, drivingForces, **kwargs)

    # ------------------------------------------------------------------
    # GenericBlackOilModel / GenericFacilityModel assembly
    # ------------------------------------------------------------------
    # The routines in this section are a direct numerical translation of
    # MRST's GenericBlackOilModel, FlowDiscretization and GenericFacilityModel
    # path.  They deliberately use *component masses* (rather than the old
    # Python compatibility model's surface-volume approximation).

    def _mrst_active_wells(self, drivingForces, state=None):
        # ``GenericFacilityModel`` mutates state.wellSol control fields on a
        # limit switch.  Retain a state-owned copy of W so the force control
        # remains immutable while subsequent Newton systems use the switched
        # mode.
        if isinstance(state, dict) and isinstance(state.get('facility_wells'), list):
            wells = state['facility_wells']
        else:
            wells = []
            forces = drivingForces or {}
            wells = forces.get('W', []) if isinstance(forces, dict) else []
        return [w for w in wells if isinstance(w, dict) and bool(w.get('status', True))]

    def _mrst_surface_densities(self):
        """Return [water, oil, gas] surface density from ECLIPSE DENSITY."""
        deck = getattr(self, 'inputdata', None)
        props = deck.get('PROPS', {}) if isinstance(deck, dict) else {}
        vals = _np.asarray(props.get('DENSITY', []), dtype=float).ravel()
        # ECLIPSE/MRST DENSITY ordering is oil, water, gas.  The generic
        # model's component order is water, oil, gas.
        if vals.size >= 3:
            return float(vals[1]), float(vals[0]), float(vals[2])
        return 1000.0, 800.0, 1.0

    def _mrst_pore_volume(self, pressure):
        """BlackOilPoreVolume: base PV times assignROCK pvMultR."""
        p = _np.asarray(pressure, dtype=float).ravel()
        pv = self._param_value(self._porevolume_vector())
        rock = self.rock if isinstance(self.rock, dict) else {}
        cr = _np.asarray(rock.get('cr', 0.0), dtype=float).ravel()
        pref = _np.asarray(rock.get('pref', 0.0), dtype=float).ravel()
        if cr.size == 0:
            cr = _np.zeros_like(p)
        if pref.size == 0:
            pref = _np.zeros_like(p)
        if cr.size == 1:
            cr = _np.full(p.size, float(cr[0]))
        if pref.size == 1:
            pref = _np.full(p.size, float(pref[0]))
        if cr.size != p.size:
            cr = _np.resize(cr, p.size)
        if pref.size != p.size:
            pref = _np.resize(pref, p.size)
        x = cr * (p - pref)
        return pv * (1.0 + x + 0.5 * x * x)

    def _mrst_pore_volume_adi(self, pressure):
        """ADI form of BlackOilPoreVolume used by MRST's CNV residual."""
        p = pressure
        nc = p.val.size
        pv = self._param_value(self._porevolume_vector())
        rock = self.rock if isinstance(self.rock, dict) else {}
        cr = _np.asarray(rock.get('cr', 0.0), dtype=float).ravel()
        pref = _np.asarray(rock.get('pref', 0.0), dtype=float).ravel()
        if cr.size == 0:
            cr = _np.zeros(nc)
        if pref.size == 0:
            pref = _np.zeros(nc)
        if cr.size == 1:
            cr = _np.full(nc, float(cr[0]))
        if pref.size == 1:
            pref = _np.full(nc, float(pref[0]))
        if cr.size != nc:
            cr = _np.resize(cr, nc)
        if pref.size != nc:
            pref = _np.resize(pref, nc)
        x = (p - pref) * cr
        return (1.0 + x + 0.5 * x * x) * pv

    def _phase_pvt_adi(self, pressure, rs_override=None, rv_override=None,
                       sG_override=None, oil_saturated_override=None,
                       gas_saturated_override=None):
        """Evaluate deck PVT through the direct ADI table path."""
        pvt = getattr(self, '_blackoil_pvt', None)
        if pvt is None or not hasattr(pvt, 'eval_adi'):
            raise NotImplementedError('Deck AD assembly requires DeckBlackOilPVT.eval_adi')
        oil_saturated = (
            oil_saturated_override if oil_saturated_override is not None else
            (None if sG_override is None else (_np.asarray(sG_override.val) > 0.0))
        )
        return pvt.eval_adi(pressure, rs_override=rs_override,
                            rv_override=rv_override,
                            oil_saturated_override=oil_saturated,
                            gas_saturated_override=gas_saturated_override)

    def _mrst_blackoil_status(self, state):
        """Port of getCellStatusVO for the active disgas/vapoil choices."""
        nc = self._num_cells()
        prior = state.get('status', None) if isinstance(state, dict) else None
        if prior is not None:
            code = _np.asarray(prior, dtype=int).ravel()
            if code.size == nc:
                return code == 1, code == 2, code == 3
        sw = _np.asarray(state.get('sW', _np.zeros(nc)), dtype=float).ravel()
        sg = _np.asarray(state.get('sG', _np.zeros(nc)), dtype=float).ravel()
        so = 1.0 - sw - sg
        etol = _np.sqrt(_np.finfo(float).eps)
        wat_only = sw > 1.0 - etol
        oil_present = _np.ones(nc, dtype=bool) if not self.vapoil else ((so > 0.0) | wat_only)
        gas_present = _np.ones(nc, dtype=bool) if not self.disgas else ((sg > 0.0) | wat_only)
        return oil_present & ~gas_present, ~oil_present & gas_present, oil_present & gas_present

    @staticmethod
    def _mrst_well_signature(wells):
        signature = []
        for w in wells:
            cells = tuple(_np.asarray(w.get('cells', []), dtype=int).ravel().tolist())
            # Controls intentionally change in prepareTimestep.  The
            # primary-variable layout only depends on the active well and
            # its perforations, not on the currently selected constraint.
            signature.append((str(w.get('name', '')), float(w.get('sign', 0.0)), cells))
        return tuple(signature)

    def _mrst_write_wellsol(self, state, wells):
        qws = _np.asarray(state.get('facility_qWs', []), dtype=float).ravel()
        qos = _np.asarray(state.get('facility_qOs', []), dtype=float).ravel()
        qgs = _np.asarray(state.get('facility_qGs', []), dtype=float).ravel()
        bhp = _np.asarray(state.get('facility_bhp', []), dtype=float).ravel()
        cdps = state.get('facility_cdp', []) if isinstance(state, dict) else []
        entries = []
        for i, w in enumerate(wells):
            nperf = len(self._well_cells(w))
            cdp = _np.asarray(cdps[i], dtype=float).ravel().copy() if i < len(cdps) else _np.zeros(nperf, dtype=float)
            if cdp.size != nperf:
                cdp = _np.resize(cdp, nperf)
            entries.append({
                'name': w.get('name', ''), 'status': bool(w.get('status', True)),
                'type': str(w.get('type', '')).lower(), 'val': float(w.get('val', 0.0)),
                'sign': float(w.get('sign', 0.0)),
                'bhp': float(bhp[i]), 'qWs': float(qws[i]),
                'qOs': float(qos[i]), 'qGs': float(qgs[i]),
                'cstatus': list(w.get('cstatus', [True] * nperf)),
                'cdp': cdp,
            })
        state['wellSol'] = entries

    def _ensure_mrst_facility_state(self, state, drivingForces):
        """Port the relevant ``initWellSolAD`` initialization fields."""
        if not isinstance(state.get('facility_wells'), list):
            raw_wells = self._mrst_active_wells(drivingForces)
            # Never cache an *empty* well list. The states the adjoint
            # sweeps are the caller's own, so caching "no wells" here --
            # which happens whenever this is reached without usable
            # driving forces -- would persist into every later use of
            # that state and silently empty its wellSol.
            if raw_wells:
                state['facility_wells'] = deepcopy(raw_wells)
        wells = self._mrst_active_wells(drivingForces, state)
        if not wells and state.get('wellSol'):
            # Nothing to build from, but this state already carries a
            # well solution (read back from a restart file, say).
            # Overwriting it with empty facility primaries loses data the
            # caller still needs; leave the state as it stands.
            return []
        signature = self._mrst_well_signature(wells)
        nw = len(wells)
        valid = (
            state.get('facility_well_signature', None) == signature and
            all(_np.asarray(state.get(k, []), dtype=float).size == nw
                for k in ('facility_qWs', 'facility_qOs', 'facility_qGs', 'facility_bhp'))
        )
        if not valid:
            p = _np.asarray(state.get('pressure', _np.zeros(self._num_cells())), dtype=float).ravel()
            eps = _np.finfo(float).eps
            qws = _np.empty(nw, dtype=float)
            qos = _np.empty(nw, dtype=float)
            qgs = _np.empty(nw, dtype=float)
            bhp = _np.empty(nw, dtype=float)
            for i, w in enumerate(wells):
                sign = float(w.get('sign', 0.0))
                cells = self._well_cells(w)
                if not cells:
                    raise ValueError('Active well %r has no valid perforation cells' % w.get('name', ''))
                qws[i] = sign * eps
                qos[i] = sign * eps
                qgs[i] = sign * eps
                bhp[i] = p[cells[0]] + 5.0e5 * sign
                typ = str(w.get('type', '')).lower()
                val = float(w.get('val', 0.0))
                compi = _np.asarray(w.get('compi', [0.0, 0.0, 0.0]), dtype=float).ravel()
                if compi.size < 3:
                    compi = _np.pad(compi, (0, 3 - compi.size))
                if typ == 'bhp':
                    bhp[i] = val
                elif typ == 'rate':
                    qws[i], qos[i], qgs[i] = val * compi[:3]
                elif typ == 'orat':
                    qos[i] = val
                elif typ == 'wrat':
                    qws[i] = val
                elif typ == 'grat':
                    qgs[i] = val
            state['facility_qWs'] = qws
            state['facility_qOs'] = qos
            state['facility_qGs'] = qgs
            state['facility_bhp'] = bhp
            state['facility_well_signature'] = signature
            state['facility_well_names'] = [w.get('name', '') for w in wells]
            state['facility_cdp'] = [_np.zeros(len(self._well_cells(w)), dtype=float) for w in wells]
        self._mrst_write_wellsol(state, wells)
        return wells

    def _mrst_update_connection_pressure_drop(self, state, wells):
        """Numerical port of ``SimpleWell.updateConnectionPressureDropState``.

        SPE9 wells use the default serial topology and
        ``simplePressureDrop = false``.  The same ``wb2in`` triangular solve
        therefore becomes a suffix accumulation of the perforation fluxes.
        """
        nc = self._num_cells()
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sw = _np.asarray(state['sW'], dtype=float).ravel()
        sg = _np.asarray(state['sG'], dtype=float).ravel()
        rs = _np.asarray(state['rs'], dtype=float).ravel()
        rv = _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel()
        pW, pO, pG = self._phase_pressures(p, sw, sg, state.get('pcowScale'))
        so = 1.0 - sw - sg
        pvt = self._phase_pvt_from_phase_pressures(
            pW, pO, pG, rs_override=rs, rv_override=rv, sG_override=sg,
            oil_saturated_override=(sg > 0.0),
            gas_saturated_override=(so > 0.0),
        )
        rhoWS, rhoOS, rhoGS = self._mrst_surface_densities()
        rho = _np.column_stack((rhoWS * pvt['bw'],
                                pvt['bo'] * (rhoOS + rs * rhoGS),
                                rhoGS * pvt['bg']))
        krW, krO, krG = self._relative_perm(sw, sg)
        mobility = _np.column_stack((krW / _np.maximum(pvt['muw'], 1.0e-30),
                                     krO / _np.maximum(pvt['muo'], 1.0e-30),
                                     krG / _np.maximum(pvt['mug'], 1.0e-30)))
        gravity = float(_np.linalg.norm(_np.asarray(
            getattr(self, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()))
        cdps = []
        previous_phase_flux = state.get('facility_perforation_phase_flux', [])
        previous_component_flux = state.get('facility_perforation_component_flux', [])
        for w in wells:
            iw = len(cdps)
            cells = self._well_cells(w)
            nperf = len(cells)
            if nperf == 0:
                cdps.append(_np.zeros(0, dtype=float))
                continue
            wi = _np.asarray(w.get('WI', []), dtype=float).ravel()
            if wi.size == 1 and nperf > 1:
                wi = _np.full(nperf, wi[0])
            if wi.size < nperf:
                wi = _np.pad(wi, (0, nperf - wi.size), constant_values=0.0)
            compi = _np.asarray(w.get('compi', [0.0, 0.0, 0.0]), dtype=float).ravel()
            if compi.size < 3:
                compi = _np.pad(compi, (0, 3 - compi.size))
            valid = _np.asarray([(0 <= c < nc) for c in cells], dtype=bool)
            use_previous_flux = (
                iw < len(previous_phase_flux) and iw < len(previous_component_flux) and
                _np.asarray(previous_component_flux[iw], dtype=float).size and
                _np.sum(_np.abs(_np.asarray(previous_component_flux[iw], dtype=float))) >= 1.0e-20
            )
            if use_previous_flux:
                # SimpleWell.m lines 368--371: use the converged
                # connection fluxes stored by updateAfterConvergence.
                qphase = _np.asarray(previous_phase_flux[iw], dtype=float)
                qcomponent = _np.asarray(previous_component_flux[iw], dtype=float)
                if qphase.shape != (nperf, 3) or qcomponent.shape != (nperf, 3):
                    use_previous_flux = False
                else:
                    qvol = _np.sum(qphase, axis=1)
                    qmass = _np.sum(qcomponent, axis=1)
                    # Respect SimpleWell.allowCrossflow.  Its default is
                    # true; the branch below is nevertheless the exact
                    # source condition for these standard wells.
                    sign = float(w.get('sign', 0.0))
                    if sign != 0.0:
                        xflow = _np.sign(qmass) != sign
                        xflow = xflow & ~_np.all(xflow)
                        qmass[xflow] = 0.0
                        qvol[xflow] = 0.0
            if not use_previous_flux:
                qphase = _np.zeros((nperf, 3), dtype=float)
                if float(w.get('sign', 0.0)) < 0.0:
                    # First-step producer branch in SimpleWell.m: WI .* mob_res.
                    qphase[valid, :] = wi[valid, None] * mobility[_np.asarray(cells, dtype=int)[valid], :]
                else:
                    # Injector branch: WI * compi.
                    qphase[valid, :] = wi[valid, None] * compi[:3]
                rho_perf = _np.zeros_like(qphase)
                rho_perf[valid, :] = rho[_np.asarray(cells, dtype=int)[valid], :]
                qvol = _np.sum(qphase, axis=1)
                qmass = _np.sum(qphase * rho_perf, axis=1)
            # ``wb2in`` for the default topo [0,1; 1,2; ...] has C with
            # a diagonal of one and a -1 superdiagonal.  abs(C\\q) is the
            # suffix sum used below.
            wbvol = _np.abs(_np.cumsum(qvol[::-1])[::-1])
            wbmass = _np.abs(_np.cumsum(qmass[::-1])[::-1])
            rhomix = _np.divide(wbmass, wbvol, out=_np.zeros(nperf), where=_np.abs(wbvol) > 0.0)
            dz_abs = _np.asarray(w.get('dZ', _np.zeros(nperf)), dtype=float).ravel()
            if dz_abs.size < nperf:
                dz_abs = _np.pad(dz_abs, (0, nperf - dz_abs.size), constant_values=0.0)
            dz = _np.diff(_np.r_[0.0, dz_abs[:nperf]])
            cdps.append(_np.cumsum(gravity * rhomix * dz))
        state['facility_cdp'] = cdps

    def _mrst_apply_well_limits(self, state, wells):
        """Port ``GenericFacilityModel.applyWellLimitsWellSol``."""
        qws = _np.asarray(state['facility_qWs'], dtype=float).ravel()
        qos = _np.asarray(state['facility_qOs'], dtype=float).ravel()
        qgs = _np.asarray(state['facility_qGs'], dtype=float).ravel()
        bhp = _np.asarray(state['facility_bhp'], dtype=float).ravel()
        for iw, w in enumerate(wells):
            if not bool(w.get('status', True)):
                continue
            typ = str(w.get('type', '')).lower()
            lims = dict(w.get('lims', {}))
            total = qws[iw] + qos[iw] + qgs[iw]
            if float(w.get('sign', 0.0)) > 0.0:
                modes = ('bhp', 'rate', 'vrat')
                limits = {'bhp': _np.inf, 'rate': _np.inf, 'vrat': -_np.inf}
                limits.update(lims)
                flags = (bhp[iw] > limits['bhp'], total > limits['rate'], total < limits['vrat'])
            else:
                modes = ('bhp', 'orat', 'lrat', 'grat', 'wrat', 'vrat')
                limits = {'bhp': -_np.inf, 'orat': -_np.inf, 'lrat': -_np.inf,
                          'grat': -_np.inf, 'wrat': -_np.inf, 'vrat': _np.inf}
                limits.update(lims)
                flags = (bhp[iw] < limits['bhp'], qos[iw] < limits['orat'],
                         qws[iw] + qos[iw] < limits['lrat'], qgs[iw] < limits['grat'],
                         qws[iw] < limits['wrat'], total > limits['vrat'])
            switch = next((mode for mode, flag in zip(modes, flags) if mode != typ and flag), None)
            if switch is None:
                continue
            w['type'] = switch
            w['control'] = switch.upper()
            w['val'] = float(limits[switch])
            # The initialization below is verbatim from
            # applyWellLimitsWellSol.m lines 521--536.
            if switch == 'rate':
                compi = _np.asarray(w.get('compi', [0.0, 0.0, 0.0]), dtype=float).ravel()
                if compi.size < 3:
                    compi = _np.pad(compi, (0, 3 - compi.size))
                qws[iw], qos[iw], qgs[iw] = w['val'] * compi[:3]
            elif switch == 'orat':
                qos[iw] = w['val']
            elif switch == 'wrat':
                qws[iw] = w['val']
            elif switch == 'grat':
                qgs[iw] = w['val']
        state['facility_qWs'] = qws
        state['facility_qOs'] = qos
        state['facility_qGs'] = qgs
        state['facility_bhp'] = bhp

    def _mrst_pack_primary(self, state, drivingForces):
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        nc = self._num_cells()
        st1, st2, st3 = self._mrst_blackoil_status(state)
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sw = _np.asarray(state['sW'], dtype=float).ravel()
        sg = _np.asarray(state['sG'], dtype=float).ravel()
        rs = _np.asarray(state['rs'], dtype=float).ravel()
        rv = _np.asarray(state['rv'], dtype=float).ravel()
        if self.disgas or self.vapoil:
            x = st1 * rs + st2 * rv + st3 * sg
        else:
            # MRST ThreePhaseBlackOilModel.getPrimaryVariables: without
            # dissolution/vaporization the third primary variable is gas
            # saturation itself (``else: x = sG; gvar = 'sG'``), not the
            # Rs/Rv/Sg status switch.  For a dead-oil/dry-gas deck whose
            # cells start with no free gas (sG = 0), the status-gated x
            # would be rs = 0 -- a constant the gas equation does not
            # depend on, leaving one structurally empty column per cell.
            x = sg
        values = _np.concatenate((p, sw, x,
                                  _np.asarray(state['facility_qWs'], dtype=float).ravel(),
                                  _np.asarray(state['facility_qOs'], dtype=float).ravel(),
                                  _np.asarray(state['facility_qGs'], dtype=float).ravel(),
                                  _np.asarray(state['facility_bhp'], dtype=float).ravel()))
        return values, (st1, st2, st3), wells

    def _mrst_unpack_primary(self, state, primary, status, wells):
        """Port ThreePhaseBlackOilModel.initStateAD variable switching."""
        nc = self._num_cells()
        nw = len(wells)
        primary = _np.asarray(primary, dtype=float).ravel()
        expected = 3 * nc + 4 * nw
        if primary.size != expected:
            raise ValueError('MRST generic primary vector is %d entries, got %d' % (expected, primary.size))
        st1, st2, st3 = status
        out = dict(state)
        p = primary[:nc].copy()
        sw = primary[nc:2 * nc].copy()
        x = primary[2 * nc:3 * nc].copy()
        # This is exactly the ``sG = st{2}.*(1-sW) + st{3}.*x`` relation
        # in ThreePhaseBlackOilModel.initStateAD (only when dissolution or
        # vaporization is active; otherwise x is sG itself).
        if self.disgas or self.vapoil:
            sg = st2 * (1.0 - sw) + st3 * x
        else:
            sg = x
        rsmax = self._phase_pvt(p)['rs']
        old_rs = _np.asarray(state.get('rs', _np.zeros(nc)), dtype=float).ravel()
        old_rv = _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel()
        rs = (~st1) * rsmax + st1 * x if self.disgas else old_rs
        if self.vapoil:
            _, _, pg = self._phase_pressures(p, sw, sg)
            pvt = getattr(self, '_blackoil_pvt', None)
            if pvt is None or not hasattr(pvt, 'rv_sat'):
                raise NotImplementedError('Deck VAPOIL state reconstruction requires DeckBlackOilPVT.rv_sat')
            rv = (~st2) * pvt.rv_sat(pg) + st2 * x
        else:
            rv = old_rv
        out['pressure'] = p
        out['sW'] = sw
        out['sG'] = sg
        out['rs'] = rs
        out['rv'] = rv
        start = 3 * nc
        out['facility_qWs'] = primary[start:start + nw].copy(); start += nw
        out['facility_qOs'] = primary[start:start + nw].copy(); start += nw
        out['facility_qGs'] = primary[start:start + nw].copy(); start += nw
        out['facility_bhp'] = primary[start:start + nw].copy()
        return out

    def _primary_seed(self, values, nvar, offset, reverse_mode, group=None):
        """Seed one primary variable, or not.

        Port of the branch at the top of ``PhysicalModel.getEquations``::

            if reverseMode
                state = model.getStateAD(state, false);   % not seeded
            else
                [state, ~] = model.getStateAD(state, ~resOnly);

        Both ``reverseMode`` and ``resOnly`` mean the same thing here:
        the current state carries no derivative. What *does* carry one
        differs -- the previous state for the coupling term, the tuned
        parameters for ``partialWRTparam`` -- and the assembled system
        is therefore as wide as *that*, not as wide as the states.

        MATLAB gets this for free: ``getStateAD(state, false)`` leaves
        plain doubles and ``value()`` accepts them everywhere, so the AD
        side sets the width by itself. Here the ``_adi`` helpers reach
        for ``.val``, so the state becomes a zero-derivative constant
        instead -- which has to be given the right width, taken from
        whatever *is* seeded.

        ``group`` is the offsets of all the cell variables being seeded
        together, which the diagonal backend needs and the sparse one
        ignores; see :meth:`AutoDiffBackend.variable`.
        """
        backend = self.autodiff_backend
        if reverse_mode:
            return backend.constant(
                _np.asarray(values, dtype=float).ravel(), nvar)
        return backend.variable(values, nvar, offset, group)

    @property
    def autodiff_backend(self):
        """The representation this model assembles derivatives in.

        Read through ``AutoDiffBackend``, which is the MRST-facing name and
        the one ``init_eclipse_problem_ad`` already accepts as an option, so
        a model that was never told anything keeps the sparse
        representation every stored result was produced with.
        """
        backend = getattr(self, '_autodiff_backend', None)
        if backend is None:
            backend = _get_backend(getattr(self, 'AutoDiffBackend', None))
            self._autodiff_backend = backend
        return backend

    @autodiff_backend.setter
    def autodiff_backend(self, value):
        self._autodiff_backend = _get_backend(value)
        self.AutoDiffBackend = type(self._autodiff_backend)

    @staticmethod
    def _seeded_width(*candidates):
        """The AD width already in play, if anything carries one.

        ``reverseMode`` is entered from two places that mean different
        systems: the adjoint's coupling term, where the previous state
        is seeded over the state unknowns, and ``partialWRTparam``,
        where an operator is seeded over the parameters. Reading the
        width off whichever object is AD keeps one assembly serving
        both, instead of committing to the state-sized guess.
        """
        for candidate in candidates:
            if hasattr(candidate, 'nvar'):
                return int(candidate.nvar)
            if isinstance(candidate, dict):
                for value in candidate.values():
                    if hasattr(value, 'nvar'):
                        return int(value.nvar)
        return None

    #: Assemble the flux through the fixed-width face operators.  Turning
    #: it off restores the general SparseADI path, which is what the adjoint
    #: and the parameter sensitivities use anyway; the two are checked
    #: against each other.
    useFaceOperators = True

    def _face_flux_context(self, nc, nvar, c1, c2, ngroup=3):
        """Layout, neighbour table and divergence plan for the fast flux path.

        All three depend only on the grid and the width of the system, so
        they are built once and kept.  The divergence plan is the expensive
        one -- it works out the assembled matrix's sparsity -- and rebuilding
        it per Newton iteration would cost more than it saves.
        """
        key = (int(nc), int(nvar), int(c1.size), int(ngroup))
        cached = getattr(self, '_face_flux_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]

        layout = _CellVariableLayout(nc, ngroup, nvar)
        neighbours = _np.stack([_np.asarray(c1, dtype=_np.int64),
                                _np.asarray(c2, dtype=_np.int64)], axis=1)
        assembler = _DivergenceAssembler(neighbours, nc, layout)
        context = (layout, neighbours, assembler)
        self._face_flux_cache = (key, context)
        return context

    def _mrst_generic_adi_residual(self, state0, state, dt, drivingForces,
                                   wells, reverseMode=False):
        """Sparse ADI assembly for the direct generic black-oil residual.

        The component-mass and facility equations, evaluated with the
        direct port of MRST's ADI operators, so the cost scales with the
        TPFA stencil rather than with the number of unknowns.

        This is now the only expression of these equations. A numeric
        twin and its column-wise finite-difference Jacobian sat beside it
        from the first port; the ADI path superseded them, and by the
        time they were removed the twin referred to variables that no
        longer existed and raised on its first call. Nothing had noticed,
        because its only reachable caller was the ``ResOnly`` branch and
        no black-oil model sets ``stepFunctionIsLinear``. That branch now
        takes the value half of this assembly.
        """
        if not (self.water and self.oil and self.gas):
            raise NotImplementedError('Sparse ADI assembly currently requires all three active phases')
        nc = self._num_cells()
        nw = len(wells)
        status = self._mrst_blackoil_status(state)
        st1, st2, st3 = [_np.asarray(x, dtype=bool).ravel() for x in status]
        nvar = 3 * nc + 4 * nw
        if reverseMode:
            # Whatever is seeded sets the width: state0 for the adjoint's
            # coupling term, an operator for partialWRTparam.
            seeded = self._seeded_width(
                state0.get('pressure') if isinstance(state0, dict) else None,
                getattr(self, 'porevolume', None), self.operators)
            if seeded is not None:
                nvar = seeded
        seed = self._primary_seed
        cells = (0, nc, 2 * nc)
        p = seed(state['pressure'], nvar, 0, reverseMode, cells)
        sw = seed(state['sW'], nvar, nc, reverseMode, cells)
        x = seed(self._mrst_pack_primary(state, drivingForces)[0][2 * nc:3 * nc],
                 nvar, 2 * nc, reverseMode, cells)
        if self.disgas or self.vapoil:
            sg = (1.0 - sw) * st2 + x * st3
        else:
            # No dissolution/vaporization: the third primary variable is
            # gas saturation in every cell (MRST ``else: x = sG``), so the
            # gas equation keeps a full sg column instead of the status-
            # gated one which is identically zero where sG starts at 0.
            sg = x
        pW, pO, pG = self._phase_pressures_adi(p, sw, sg, state.get('pcowScale'))
        # ThreePhaseBlackOilModel.initStateAD reconstructs the cell-wise
        # x variable as Rs (oil only), Rv (gas only) or Sg (both phases).
        # RsMax is evaluated at the oil reference pressure; RvMax is a
        # PVTPropertyFunction and is evaluated at the gas phase pressure.
        pvt_sat = self._phase_pvt_adi(pO)
        # Every AD value built from here on follows the model's backend, so
        # the whole property chain stays in one representation. A hardcoded
        # SparseADI constant mixed into a diagonal chain does not fail
        # loudly -- it drags the rest of the expression back onto sparse
        # matrix algebra, which is the cost this was meant to avoid.
        AD = self.autodiff_backend.ad_class
        rs = (pvt_sat['rs'] * (~st1)) + x * st1 if self.disgas else AD.constant(state.get('rs', _np.zeros(nc)), nvar)
        if self.vapoil:
            deck_pvt = getattr(self, '_blackoil_pvt', None)
            if deck_pvt is None or not hasattr(deck_pvt, 'rv_sat_adi'):
                raise NotImplementedError('Deck AD VAPOIL assembly requires DeckBlackOilPVT.rv_sat_adi')
            rv_sat = deck_pvt.rv_sat_adi(pG)
            rv = rv_sat * (~st2) + x * st2
        else:
            rv = AD.constant(state.get('rv', _np.zeros(nc)), nvar)
        so = 1.0 - sw - sg
        pvt = self._phase_pvt_from_phase_pressures_adi(
            pW, pO, pG, rs_override=rs, rv_override=rv, sG_override=sg,
            oil_saturated_override=(sg.val > 0.0),
            gas_saturated_override=(so.val > 0.0),
        )
        bW, bO, bG = pvt['bw'], pvt['bo'], pvt['bg']
        muW, muO, muG = pvt['muw'], pvt['muo'], pvt['mug']
        krW, krO, krG = self._relative_perm_adi(sw, sg)
        lamW = krW / _ad_maximum(muW, 1.0e-30)
        lamO = krO / _ad_maximum(muO, 1.0e-30)
        lamG = krG / _ad_maximum(muG, 1.0e-30)

        rhoWS, rhoOS, rhoGS = self._mrst_surface_densities()
        rhoW = bW * rhoWS
        rhoO = bO * (rhoOS + rs * rhoGS)
        rhoG = bG * (rhoGS + rv * rhoOS)
        rhoG_component = bG * rhoGS
        p0 = self._state0_value(state0['pressure'])
        sw0 = self._state0_value(state0['sW'])
        sg0 = self._state0_value(state0['sG'])
        so0 = 1.0 - sw0 - sg0
        rs0 = self._state0_value(state0['rs'])
        rv0 = self._state0_value(state0.get('rv'), _np.zeros(nc))
        _pp0, _pvt0fn, _pv0fn = self._state0_fns(p0)
        pW0, pO0, pG0 = _pp0(p0, sw0, sg0, state0.get('pcowScale'))
        pvt0 = _pvt0fn(
            pW0, pO0, pG0, rs_override=rs0, rv_override=rv0, sG_override=sg0,
            oil_saturated_override=(self._flag_value(sg0) > 0.0),
            gas_saturated_override=(self._flag_value(so0) > 0.0),
        )
        rhoW0 = rhoWS * pvt0['bw']
        rhoO0 = pvt0['bo'] * (rhoOS + rs0 * rhoGS)
        rhoG_phase0 = pvt0['bg'] * (rhoGS + rv0 * rhoOS)
        rhoG_component0 = pvt0['bg'] * rhoGS

        ops = self.operators or {}
        c1, c2, T = self._internal_connections()
        nface = c1.size
        centroids = _np.asarray(self.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)
        z = centroids[:, 2] if centroids.ndim == 2 and centroids.shape[1] >= 3 else _np.zeros(nc)
        grav = _np.asarray(getattr(self, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()
        g = float(grav[-1]) if grav.size else 9.80665
        C = (_sp.csr_matrix((_np.r_[_np.ones(nface), -_np.ones(nface)],
                            (_np.r_[c1, c2], _np.r_[_np.arange(nface), _np.arange(nface)])),
                           shape=(nc, nface)) if nface else _sp.csr_matrix((nc, 0)))

        # The fixed-width face representation, when this assembly is the
        # ordinary forward one.  It reads a cell property's derivatives by
        # taking the diagonal of each variable group's block, which is only
        # the right thing while the primary variables are the cell ones at
        # their usual offsets.  ``reverseMode`` breaks exactly that: the
        # adjoint seeds the *previous* state and ``partialWRTparam`` seeds
        # the parameters, so the columns mean something else and the
        # extraction would be silently wrong rather than loud.
        # The guard has to cover the *inputs*, not just the mode flag.  A
        # fixed-width face value can only express dependence on its two
        # cells' primary variables, so a transmissibility that carries
        # derivatives of its own -- which is exactly what a sensitivity with
        # respect to transmissibility seeds, in forward mode -- has nowhere
        # to put them.  Checking reverseMode alone let that through, and the
        # face arithmetic met an AD operand it could not read.
        faces = None
        if (nface and not reverseMode and nvar == 3 * nc + 4 * nw
                and not _is_ad(T)
                and getattr(self, 'useFaceOperators', True)):
            faces = self._face_flux_context(nc, nvar, c1, c2)

        def phase_flux(phase_pressure, lam, rho, component_density):
            if not nface:
                return (AD.constant(_np.zeros(nc), nvar),
                        AD.constant(_np.zeros(0), nvar), _np.zeros(0, dtype=int))
            potential = (phase_pressure[c2] - phase_pressure[c1] -
                         (rho[c1] + rho[c2]) * (0.5 * g * (z[c2] - z[c1])))
            upstream = _np.where(potential.val <= 0.0, c1, c2)
            q = potential * (-T) * lam[upstream]
            flux = q * component_density[upstream]
            # linear_map leaves the diagonal representation: a cell's
            # divergence sums over all its faces, so its Jacobian row has as
            # many entries as the cell has neighbours.  That is genuine
            # sparse structure, and the flux term is where it starts.
            return flux.linear_map(C), q, upstream

        def phase_flux_fast(phase_pressure, lam, rho, component_density):
            """The same flux, through MRST's fixed-width face operators.

            Identical arithmetic, one array shape throughout: the potential
            is a two-point gradient minus a face average, the mobility and
            density are upstream gathers, and the divergence is assembled
            from the dense derivative array without a face-length matrix in
            between.
            """
            layout, neighbours, assembler = faces
            potential = (_FaceValue.gradient(phase_pressure, layout, neighbours)
                         - _FaceValue.average(rho, layout, neighbours) * (g * dz))
            flag = _upwind_flag(potential)
            upstream = _np.where(flag, c1, c2)
            q = (potential * (-T)) * _FaceValue.gather(lam, layout, neighbours, flag)
            flux = q * _FaceValue.gather(component_density, layout, neighbours, flag)
            return assembler.assemble(flux), q, (flag, upstream)

        if faces is not None:
            dz = z[c2] - z[c1]
            layout, neighbours, assembler = faces
            divW, qWface, upW = phase_flux_fast(pW, lamW, rhoW, bW * rhoWS)
            divO, qOface, upO = phase_flux_fast(pO, lamO, rhoO, bO * rhoOS)
            divG, qGface, upG = phase_flux_fast(pG, lamG, rhoG, bG * rhoGS)
            flagO, upO = upO
            flagG, upG = upG
            upW = upW[1]
            # The dissolved-gas and vaporised-oil cross terms ride on the
            # phase flux that carries them, gathered on that phase's own
            # upstream direction.
            divG = divG + assembler.assemble(
                qOface * _FaceValue.gather(rs * (rhoGS * bO), layout, neighbours, flagO))
            if self.vapoil:
                divO = divO + assembler.assemble(
                    qGface * _FaceValue.gather(rv * (rhoOS * bG), layout, neighbours, flagG))
        else:
            divW, qWface, upW = phase_flux(pW, lamW, rhoW, bW * rhoWS)
            divO, qOface, upO = phase_flux(pO, lamO, rhoO, bO * rhoOS)
            divG, qGface, upG = phase_flux(pG, lamG, rhoG, bG * rhoGS)
            if nface:
                divG = divG + (qOface * (rs[upO] * (rhoGS) * bO[upO])).linear_map(C)
                if self.vapoil:
                    divO = divO + (qGface * (rv[upG] * rhoOS * bG[upG])).linear_map(C)

        qws = seed(state.get('facility_qWs', _np.zeros(nw)), nvar,
                   3 * nc, reverseMode)
        qos = seed(state.get('facility_qOs', _np.zeros(nw)), nvar,
                   3 * nc + nw, reverseMode)
        qgs = seed(state.get('facility_qGs', _np.zeros(nw)), nvar,
                   3 * nc + 2 * nw, reverseMode)
        bhp = seed(state.get('facility_bhp', _np.zeros(nw)), nvar,
                   3 * nc + 3 * nw, reverseMode)

        def component_mass_fn(c, qph):
            qphW, qphO, qphG = qph
            cmassW = qphW * rhoWS * bW[c]
            cmassO = qphO * rhoOS * bO[c]
            if self.vapoil:
                cmassO = cmassO + qphG * rv[c] * rhoOS * bG[c]
            cmassG = qphG * rhoGS * bG[c] + qphO * rs[c] * rhoGS * bO[c]
            return [cmassW, cmassO, cmassG]

        (srcW, srcO, srcG), (surface_w, surface_o, surface_g), perf_phase_all, perf_component_all = \
            self.FacilityModel.compute_well_contributions(
                wells=wells, state=state, p=p, bhp=bhp,
                lam_phases=[lamW, lamO, lamG], rhoS_phases=[rhoWS, rhoOS, rhoGS],
                component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar, n_component_phases=3,
            )

        state['facility_perforation_phase_flux'] = perf_phase_all
        state['facility_perforation_component_flux'] = perf_component_all

        pv = self._mrst_pore_volume_adi(p)
        pv0 = _pv0fn(p0)
        invdt = 1.0 / max(float(dt), 1.0e-30)
        resW = (pv * sw * rhoW - (pv0 * sw0 * rhoW0)) * invdt + divW - srcW
        if self.vapoil:
            resO = (pv * (so * bO + rv * bG * sg) * rhoOS -
                    (pv0 * (so0 * pvt0['bo'] + rv0 * pvt0['bg'] * sg0) * rhoOS)) * invdt + divO - srcO
        else:
            resO = (pv * so * (rhoOS) * bO - (pv0 * so0 * rhoOS * pvt0['bo'])) * invdt + divO - srcO
        resG = (pv * (sg * rhoG_component + so * rs * rhoGS * bO) -
                (pv0 * (sg0 * rhoG_component0 + so0 * rs0 * rhoGS * pvt0['bo']))) * invdt + divG - srcG
        # A normalisation for the well equations, not a modelled
        # quantity -- MRST takes value() here too, so it carries no
        # derivative even when the previous state is an AD variable.
        rho_scale = _np.asarray([
            rhoWS / _np.mean(self._flag_value(rhoW0)),
            rhoOS / _np.mean(self._flag_value(rhoO0)),
            rhoGS / _np.mean(self._flag_value(rhoG_phase0))])
        fW = (qws - surface_w) * rho_scale[0]
        fO = (qos - surface_o) * rho_scale[1]
        fG = (qgs - surface_g) * rho_scale[2]
        closure = _FacilityModel.compute_control_equations(
            wells, qs_phases={'w': qws, 'o': qos, 'g': qgs}, bhp=bhp, phase_order=['w', 'o', 'g'],
        )
        # concat leaves the diagonal representation for good: stacking seven
        # equations of different heights has no single column per row.  That
        # is the right place for it -- the system is about to be solved, and
        # a solve needs a real sparse matrix anyway.
        closure_adi = AD.concat(closure) if closure else AD.constant(_np.zeros(0), nvar)
        residual = _SparseADI.concat((resW, resO, resG, fW, fO, fG, closure_adi))
        return residual, {'status': status, 'pvt': pvt}

    @staticmethod
    def _state0_value(value, default=None):
        """A state0 field, letting an AD value through.

        The adjoint needs dR_n/dx_{n-1}, which means evaluating the
        residual with the *previous* state seeded as the AD variable.
        Forcing state0 to plain floats -- which every assembly did --
        strips those derivatives silently, so the adjoint could never be
        built on top of these equations.

        For a plain array this is exactly the cast it replaces, so an
        ordinary forward evaluation is bit-identical.
        """
        if value is None:
            value = default
        if hasattr(value, 'val'):        # SparseADI: keep the derivative
            return value
        return _np.asarray(value, dtype=float).ravel()

    @staticmethod
    def _flag_value(x):
        """The numeric value behind a possibly-AD quantity.

        Phase-status flags -- is this cell saturated, is that phase
        present -- are discrete. MRST holds them fixed through a Newton
        iteration and does not differentiate them, so a comparison that
        forms one takes the value and drops the derivative.
        """
        return x.val if hasattr(x, 'val') else x

    def getStateAD(self, state, init=True, drivingForces=None, offset=0,
                   nvar=None, nw=None):
        """Port of MRST ``getStateAD``: a state whose primary variables
        carry derivatives.

        The three cell primary variables are the ones the forward
        assembly seeds -- pressure, water saturation, and the third
        variable ``x`` that stands for Sg, Rs or Rv depending on which
        phases are present in the cell. They are seeded at the same
        column offsets the forward path uses, so a Jacobian built from
        this state lines up with one built from the current state.

        ``offset`` shifts every column, which is what lets the adjoint
        hold the current and previous states in one system: seed the
        current state at 0 and the previous at ``3*nc`` and the two
        blocks sit side by side.

        With ``init=False`` the state comes back untouched, matching
        MRST's use of the flag to mean "values only, this time".
        """
        if not init:
            return state

        nc = int(self.G['cells']['num'])
        if nw is None:
            nw = (len(self._mrst_active_wells(drivingForces, state))
                  if drivingForces else 0)
        if nvar is None:
            nvar = 3 * nc + 4 * nw

        out = dict(state)
        out['pressure'] = _SparseADI.variable(
            _np.asarray(state['pressure'], dtype=float).ravel(), nvar,
            offset + 0)
        out['sW'] = _SparseADI.variable(
            _np.asarray(state['sW'], dtype=float).ravel(), nvar,
            offset + nc)

        packed = self._mrst_pack_primary(state, drivingForces)[0]
        x = _SparseADI.variable(packed[2 * nc:3 * nc], nvar,
                                offset + 2 * nc)
        out['x'] = x

        # Reconstruct the fields the assembly actually reads from the
        # third primary variable, exactly as the forward path does: x is
        # Sg where both oil and gas are present, Rs where the cell is
        # oil-only, and Rv where it is gas-only. Leaving these as plain
        # arrays would strip the derivative again one level down.
        st1, st2, st3 = [_np.asarray(f, dtype=bool).ravel()
                         for f in self._mrst_blackoil_status(state)]
        if self.disgas or self.vapoil:
            out['sG'] = (1.0 - out['sW']) * st2 + x * st3
        else:
            # No dissolution/vaporization: x is gas saturation itself.
            out['sG'] = x

        if self.disgas:
            pW, pO, pG = self._phase_pressures_adi(out['pressure'],
                                                   out['sW'], out['sG'],
                                                   state.get('pcowScale'))
            out['rs'] = self._phase_pvt_adi(pO)['rs'] * (~st1) + x * st1
        elif 'rs' in state:
            out['rs'] = _SparseADI.constant(
                _np.asarray(state['rs'], dtype=float).ravel(), nvar)

        # The well unknowns, seeded at the same offsets the forward
        # assembly uses: qWs, qOs, qGs, bhp in blocks of nw after the
        # 3*nc cell variables. An objective that matches rates or bhp has
        # its whole derivative here and none of it in the cells, so
        # leaving these as plain numbers gives a zero gradient that looks
        # like a converged one.
        #
        # ``nw`` has to be the count the *assembly* used, which is not
        # always a plain read of the driving forces -- the facility state
        # can resolve a different well list. A caller that knows the
        # assembled width should pass it; getting this wrong puts the
        # seeds at the wrong columns while every size still checks out.
        if nw and offset + 3 * nc + 4 * nw <= nvar:
            for i, name in enumerate(('qWs', 'qOs', 'qGs', 'bhp')):
                key = 'facility_' + name
                values = _np.asarray(state.get(key, _np.zeros(nw)),
                                     dtype=float).ravel()
                if values.size != nw:
                    values = _np.zeros(nw)
                out[key] = _SparseADI.variable(values, nvar,
                                               offset + 3 * nc + i * nw)
        return out

    def _state0_fns(self, p0):
        """The property evaluators to use for the previous state.

        The model keeps parallel value and AD stacks. A forward
        evaluation wants the value ones for state0 -- it is a constant
        there -- but the adjoint needs dR_n/dx_{n-1}, which means seeding
        state0 as the AD variable and routing it through the AD stack
        instead. Dispatching on what state0 actually is keeps the forward
        path exactly as it was.
        """
        if hasattr(p0, 'val'):
            return (self._phase_pressures_adi,
                    self._phase_pvt_from_phase_pressures_adi,
                    self._mrst_pore_volume_adi)
        return (self._phase_pressures,
                self._phase_pvt_from_phase_pressures,
                self._mrst_pore_volume)

    def _mrst_generic_adi_residual_ow(self, state0, state, dt, drivingForces,
                                      wells, reverseMode=False):
        """Two-phase branch of GenericBlackOilModel + GenericFacilityModel.

        This is the active water/oil subset of MRST's state-function chain
        (EGG): component masses, phase-potential TPFA fluxes and the three
        water/oil/bhp facility primary-variable groups.
        """
        nc, nw = self._num_cells(), len(wells)
        nvar = 2 * nc + 3 * nw
        # Through the model's backend, like the three-phase branch: this one
        # named SparseADI outright, so a two-phase deck could never use the
        # diagonal representation however the model was configured.
        if reverseMode:
            # Whatever is seeded sets the width: state0 for the adjoint's
            # coupling term, an operator for partialWRTparam.  Same rule the
            # three-phase branch follows.
            seeded = self._seeded_width(
                state0.get('pressure') if isinstance(state0, dict) else None,
                getattr(self, 'porevolume', None), self.operators)
            if seeded is not None:
                nvar = seeded
        AD = self.autodiff_backend.ad_class
        seed = self._primary_seed
        cells = (0, nc)
        p = seed(state['pressure'], nvar, 0, reverseMode, cells)
        sw = seed(state['sW'], nvar, nc, reverseMode, cells)
        zero = AD.constant(_np.zeros(nc), nvar)
        pW, pO, _ = self._phase_pressures_adi(p, sw, zero)
        pvt = self._phase_pvt_from_phase_pressures_adi(
            pW, pO, pO, rs_override=zero, rv_override=zero, sG_override=zero,
        )
        p0 = self._state0_value(state0['pressure'])
        sw0 = self._state0_value(state0['sW'])
        # Which property stack state0 goes through depends on what state0
        # is: a constant in a forward evaluation, an AD variable when the
        # adjoint wants dR_n/dx_{n-1}. The three-phase assembly has
        # dispatched on that since the adjoint was added; this branch was
        # left calling the value stack directly and referring to a
        # ``_pvt0fn`` that only ever existed in the other method, so it
        # raised NameError on its first call. Nothing caught it because
        # no test drives a two-phase model through this assembly.
        _pp0, _pvt0fn, _pv0fn = self._state0_fns(p0)
        pW0, pO0, _ = _pp0(p0, sw0, _np.zeros(nc))
        pvt0 = _pvt0fn(
            pW0, pO0, pO0, rs_override=_np.zeros(nc), rv_override=_np.zeros(nc),
            sG_override=_np.zeros(nc),
        )
        bW, bO = pvt['bw'], pvt['bo']
        rhoWS, rhoOS, _ = self._mrst_surface_densities()
        rhoW, rhoO = bW * rhoWS, bO * rhoOS
        rhoW0, rhoO0 = rhoWS * pvt0['bw'], rhoOS * pvt0['bo']
        krW, krO, _ = self._relative_perm_adi(sw, zero)
        lamW = krW / _ad_maximum(pvt['muw'], 1.0e-30)
        lamO = krO / _ad_maximum(pvt['muo'], 1.0e-30)

        ops = self.operators or {}
        c1, c2, T = self._internal_connections()
        nface = c1.size
        centroids = _np.asarray(self.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)
        z = centroids[:, 2] if centroids.ndim == 2 and centroids.shape[1] >= 3 else _np.zeros(nc)
        grav = _np.asarray(getattr(self, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()
        g = float(grav[-1]) if grav.size else 9.80665
        C = (_sp.csr_matrix((_np.r_[_np.ones(nface), -_np.ones(nface)],
                             (_np.r_[c1, c2], _np.r_[_np.arange(nface), _np.arange(nface)])),
                            shape=(nc, nface)) if nface else _sp.csr_matrix((nc, 0)))

        # Same guard as the three-phase branch: the face values read a cell
        # property's derivatives by taking the diagonal of each variable
        # group's block, which only means what it should while the primary
        # variables are the cell ones at their usual offsets.  reverseMode
        # seeds something else entirely.
        # The guard has to cover the *inputs*, not just the mode flag.  A
        # fixed-width face value can only express dependence on its two
        # cells' primary variables, so a transmissibility that carries
        # derivatives of its own -- which is exactly what a sensitivity with
        # respect to transmissibility seeds, in forward mode -- has nowhere
        # to put them.  Checking reverseMode alone let that through, and the
        # face arithmetic met an AD operand it could not read.
        faces = None
        if (nface and not reverseMode and nvar == 2 * nc + 3 * nw
                and not _is_ad(T)
                and getattr(self, 'useFaceOperators', True)):
            faces = self._face_flux_context(nc, nvar, c1, c2, ngroup=2)

        def phase_flux(phase_pressure, lam, rho, component_density):
            if not nface:
                return AD.constant(_np.zeros(nc), nvar)
            if faces is not None:
                layout, neighbours, assembler = faces
                potential = (_FaceValue.gradient(phase_pressure, layout, neighbours)
                             - _FaceValue.average(rho, layout, neighbours) * (g * dz))
                flag = _upwind_flag(potential)
                q = (potential * (-T)) * _FaceValue.gather(lam, layout, neighbours, flag)
                flux = q * _FaceValue.gather(component_density, layout, neighbours, flag)
                return assembler.assemble(flux)
            potential = phase_pressure[c2] - phase_pressure[c1] - (rho[c1] + rho[c2]) * (0.5 * g * (z[c2] - z[c1]))
            upstream = _np.where(potential.val <= 0.0, c1, c2)
            q = potential * (-T) * lam[upstream]
            return (q * component_density[upstream]).linear_map(C)

        dz = z[c2] - z[c1] if nface else _np.zeros(0)
        divW = phase_flux(pW, lamW, rhoW, bW * rhoWS)
        divO = phase_flux(pO, lamO, rhoO, bO * rhoOS)
        qws = seed(state.get('facility_qWs', _np.zeros(nw)), nvar, 2 * nc, reverseMode)
        qos = seed(state.get('facility_qOs', _np.zeros(nw)), nvar, 2 * nc + nw, reverseMode)
        bhp = seed(state.get('facility_bhp', _np.zeros(nw)), nvar, 2 * nc + 2 * nw, reverseMode)

        def component_mass_fn(c, qph):
            qW, qO = qph
            return [qW * rhoWS * bW[c], qO * rhoOS * bO[c]]

        (srcW, srcO), (surface_w, surface_o), perf_phase_all, perf_component_all = \
            self.FacilityModel.compute_well_contributions(
                wells=wells, state=state, p=p, bhp=bhp,
                lam_phases=[lamW, lamO], rhoS_phases=[rhoWS, rhoOS],
                component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar, n_component_phases=2,
            )
        # _mrst_update_connection_pressure_drop always evaluates 3-phase
        # mobility/density, so it expects a gas column (zero here) even on
        # this water/oil-only path.
        perf_phase_all = [_np.pad(a, ((0, 0), (0, 1))) if a.size else _np.zeros((0, 3)) for a in perf_phase_all]
        perf_component_all = [_np.pad(a, ((0, 0), (0, 1))) if a.size else _np.zeros((0, 3)) for a in perf_component_all]
        state['facility_perforation_phase_flux'] = perf_phase_all
        state['facility_perforation_component_flux'] = perf_component_all

        pv, pv0 = self._mrst_pore_volume_adi(p), _pv0fn(p0)
        invdt = 1.0 / max(float(dt), 1.0e-30)
        resW = (pv * sw * rhoW - pv0 * sw0 * rhoW0) * invdt + divW - srcW
        resO = (pv * (1.0 - sw) * rhoOS * bO - pv0 * (1.0 - sw0) * rhoOS * pvt0['bo']) * invdt + divO - srcO
        scaleW, scaleO = rhoWS / _np.mean(rhoW0), rhoOS / _np.mean(rhoO0)
        fW, fO = (qws - surface_w) * scaleW, (qos - surface_o) * scaleO
        closure = _FacilityModel.compute_control_equations(
            wells, qs_phases={'w': qws, 'o': qos}, bhp=bhp, phase_order=['w', 'o'],
        )
        residual = _SparseADI.concat((resW, resO, fW, fO, _SparseADI.concat(closure) if closure else _SparseADI.constant(_np.zeros(0), nvar)))
        return residual, {'pvt': pvt, 'rho0': (rhoW0, rhoO0)}

    def _mrst_generic_adi_residual_og(self, state0, state, dt, drivingForces,
                                      wells, reverseMode=False):
        """Two-phase oil/gas branch of GenericBlackOilModel + GenericFacilityModel
        (``assignRelPerm.m``'s ``relPermOG``; the ``model.water``-false
        analogue of :meth:`_mrst_generic_adi_residual_ow`, keeping the same
        Rs-driven status switching as the three-phase path since disgas is
        the common case for a gas-condensate/oil-gas deck).

        MRST's own ``equationsBlackOil.m`` keeps a ``water`` equation with
        ``sW`` pinned to the constant 0 even when ``~model.water`` (see that
        function's ``if ~model.water, [sW, sW0] = deal(0); end`` branch);
        taken literally that leaves an all-zero equation row with no
        corresponding primary variable, which is not square. This mirrors
        ``equationsOilWater.m``'s pattern instead: water is dropped
        entirely, not carried as an inert placeholder.
        """
        nc = self._num_cells()
        nw = len(wells)
        status = self._mrst_blackoil_status(state)
        st1, st2, st3 = [_np.asarray(x, dtype=bool).ravel() for x in status]
        nvar = 2 * nc + 3 * nw
        if reverseMode:
            seeded = self._seeded_width(
                state0.get('pressure') if isinstance(state0, dict) else None,
                getattr(self, 'porevolume', None), self.operators)
            if seeded is not None:
                nvar = seeded
        AD = self.autodiff_backend.ad_class
        seed = self._primary_seed
        cells = (0, nc)
        p = seed(state['pressure'], nvar, 0, reverseMode, cells)
        zero = AD.constant(_np.zeros(nc), nvar)
        if self.disgas:
            rs0_val = _np.asarray(state.get('rs', _np.zeros(nc)), dtype=float).ravel()
            sg0_val = _np.asarray(state.get('sG', _np.zeros(nc)), dtype=float).ravel()
            x0 = st1 * rs0_val + st3 * sg0_val
            x = seed(x0, nvar, nc, reverseMode, cells)
        else:
            x = seed(_np.asarray(state.get('sG', _np.zeros(nc)), dtype=float).ravel(),
                     nvar, nc, reverseMode, cells)
        if self.disgas or self.vapoil:
            sg = st2 * 1.0 + x * st3
        else:
            sg = x
        pW, pO, pG = self._phase_pressures_adi(p, zero, sg)
        pvt_sat = self._phase_pvt_adi(pO)
        rs = (pvt_sat['rs'] * (~st1)) + x * st1 if self.disgas else AD.constant(state.get('rs', _np.zeros(nc)), nvar)
        if self.vapoil:
            deck_pvt = getattr(self, '_blackoil_pvt', None)
            if deck_pvt is None or not hasattr(deck_pvt, 'rv_sat_adi'):
                raise NotImplementedError('Deck AD VAPOIL assembly requires DeckBlackOilPVT.rv_sat_adi')
            rv_sat = deck_pvt.rv_sat_adi(pG)
            rv = rv_sat * (~st2) + x * st2
        else:
            rv = _SparseADI.constant(state.get('rv', _np.zeros(nc)), nvar)
        so = 1.0 - sg
        pvt = self._phase_pvt_from_phase_pressures_adi(
            pW, pO, pG, rs_override=rs, rv_override=rv, sG_override=sg,
            oil_saturated_override=(sg.val > 0.0),
            gas_saturated_override=(so.val > 0.0),
        )
        bO, bG = pvt['bo'], pvt['bg']
        muO, muG = pvt['muo'], pvt['mug']
        _, krO, krG = self._relative_perm_adi(zero, sg)
        lamO = krO / _ad_maximum(muO, 1.0e-30)
        lamG = krG / _ad_maximum(muG, 1.0e-30)

        _, rhoOS, rhoGS = self._mrst_surface_densities()
        rhoO = bO * (rhoOS + rs * rhoGS)
        rhoG = bG * (rhoGS + rv * rhoOS)
        rhoG_component = bG * rhoGS
        p0 = self._state0_value(state0['pressure'])
        sg0 = self._state0_value(state0.get('sG'), _np.zeros(nc))
        so0 = 1.0 - sg0
        rs0 = self._state0_value(state0.get('rs'), _np.zeros(nc))
        rv0 = self._state0_value(state0.get('rv'), _np.zeros(nc))
        zeros_nc = _np.zeros(nc)
        # Same dispatch as the water/oil branch and the three-phase one:
        # state0 goes through the value stack when it is a constant and
        # the AD stack when the adjoint has seeded it.
        _pp0, _pvt0fn, _pv0fn = self._state0_fns(p0)
        pW0, pO0, pG0 = _pp0(p0, zeros_nc, sg0)
        pvt0 = _pvt0fn(
            pW0, pO0, pG0, rs_override=rs0, rv_override=rv0, sG_override=sg0,
            oil_saturated_override=(self._flag_value(sg0) > 0.0),
            gas_saturated_override=(self._flag_value(so0) > 0.0),
        )
        rhoO0 = pvt0['bo'] * (rhoOS + rs0 * rhoGS)
        rhoG_phase0 = pvt0['bg'] * (rhoGS + rv0 * rhoOS)
        rhoG_component0 = pvt0['bg'] * rhoGS

        ops = self.operators or {}
        c1, c2, T = self._internal_connections()
        nface = c1.size
        centroids = _np.asarray(self.G.get('cells', {}).get('centroids', _np.zeros((nc, 3))), dtype=float)
        z = centroids[:, 2] if centroids.ndim == 2 and centroids.shape[1] >= 3 else _np.zeros(nc)
        grav = _np.asarray(getattr(self, 'gravity', [0.0, 0.0, 9.80665]), dtype=float).ravel()
        g = float(grav[-1]) if grav.size else 9.80665
        C = (_sp.csr_matrix((_np.r_[_np.ones(nface), -_np.ones(nface)],
                            (_np.r_[c1, c2], _np.r_[_np.arange(nface), _np.arange(nface)])),
                           shape=(nc, nface)) if nface else _sp.csr_matrix((nc, 0)))

        # Same guard as the other two branches: the fixed-width face values
        # assume the cell variables sit at their usual offsets, which
        # reverseMode does not.
        # The guard has to cover the *inputs*, not just the mode flag.  A
        # fixed-width face value can only express dependence on its two
        # cells' primary variables, so a transmissibility that carries
        # derivatives of its own -- which is exactly what a sensitivity with
        # respect to transmissibility seeds, in forward mode -- has nowhere
        # to put them.  Checking reverseMode alone let that through, and the
        # face arithmetic met an AD operand it could not read.
        faces = None
        if (nface and not reverseMode and nvar == 2 * nc + 3 * nw
                and not _is_ad(T)
                and getattr(self, 'useFaceOperators', True)):
            faces = self._face_flux_context(nc, nvar, c1, c2, ngroup=2)
        dz = z[c2] - z[c1] if nface else _np.zeros(0)

        def phase_flux(phase_pressure, lam, rho, component_density):
            if not nface:
                return (AD.constant(_np.zeros(nc), nvar),
                        AD.constant(_np.zeros(0), nvar), _np.zeros(0, dtype=int))
            if faces is not None:
                layout, neighbours, assembler = faces
                potential = (_FaceValue.gradient(phase_pressure, layout, neighbours)
                             - _FaceValue.average(rho, layout, neighbours) * (g * dz))
                flag = _upwind_flag(potential)
                q = (potential * (-T)) * _FaceValue.gather(lam, layout, neighbours, flag)
                flux = q * _FaceValue.gather(component_density, layout, neighbours, flag)
                return assembler.assemble(flux), q, flag
            potential = (phase_pressure[c2] - phase_pressure[c1] -
                        (rho[c1] + rho[c2]) * (0.5 * g * (z[c2] - z[c1])))
            upstream = _np.where(potential.val <= 0.0, c1, c2)
            q = potential * (-T) * lam[upstream]
            flux = q * component_density[upstream]
            return flux.linear_map(C), q, upstream

        divO, qOface, upO = phase_flux(pO, lamO, rhoO, bO * rhoOS)
        divG, qGface, upG = phase_flux(pG, lamG, rhoG, bG * rhoGS)
        if nface and faces is not None:
            layout, neighbours, assembler = faces
            divG = divG + assembler.assemble(
                qOface * _FaceValue.gather(rs * (rhoGS * bO), layout, neighbours, upO))
            if self.vapoil:
                divO = divO + assembler.assemble(
                    qGface * _FaceValue.gather(rv * (rhoOS * bG), layout, neighbours, upG))
        elif nface:
            divG = divG + (qOface * (rs[upO] * rhoGS * bO[upO])).linear_map(C)
            if self.vapoil:
                divO = divO + (qGface * (rv[upG] * rhoOS * bG[upG])).linear_map(C)

        qos = seed(state.get('facility_qOs', _np.zeros(nw)), nvar, 2 * nc, reverseMode)
        qgs = seed(state.get('facility_qGs', _np.zeros(nw)), nvar, 2 * nc + nw, reverseMode)
        bhp = seed(state.get('facility_bhp', _np.zeros(nw)), nvar, 2 * nc + 2 * nw, reverseMode)

        def component_mass_fn(c, qph):
            qphO, qphG = qph
            cmassO = qphO * rhoOS * bO[c]
            if self.vapoil:
                cmassO = cmassO + qphG * rv[c] * rhoOS * bG[c]
            cmassG = qphG * rhoGS * bG[c] + qphO * rs[c] * rhoGS * bO[c]
            return [cmassO, cmassG]

        # SimpleWell.compute_contributions indexes ``w['compi']`` by
        # position against ``lam_phases`` (``compi[:nph]``); ``compi`` is
        # always stored in the deck's standard [water, oil, gas] order
        # (e.g. init_eclipse_problem_ad.py's injector defaults), which only
        # happens to line up with lam_phases=[lamW, lamO] on the OW path.
        # Here lam_phases is [oil, gas] (indices 1, 2), so compi must be
        # sliced to match or a water-fraction/oil-fraction pair would be
        # used as if it were the oil/gas mix.
        wells_og = []
        for w in wells:
            compi = _np.asarray(w.get('compi', [0.0, 1.0, 0.0]), dtype=float).ravel()
            if compi.size < 3:
                compi = _np.pad(compi, (0, 3 - compi.size))
            w = dict(w)
            w['compi'] = compi[[1, 2]]
            wells_og.append(w)

        (srcO, srcG), (surface_o, surface_g), perf_phase_all, perf_component_all = \
            self.FacilityModel.compute_well_contributions(
                wells=wells_og, state=state, p=p, bhp=bhp,
                lam_phases=[lamO, lamG], rhoS_phases=[rhoOS, rhoGS],
                component_mass_fn=component_mass_fn, nc=nc, nw=nw, nvar=nvar, n_component_phases=2,
            )
        # _mrst_update_connection_pressure_drop always evaluates 3-phase
        # [water, oil, gas] mobility/density; insert the missing water
        # column at the front (not appended) to match that ordering.
        perf_phase_all = [_np.pad(a, ((0, 0), (1, 0))) if a.size else _np.zeros((0, 3)) for a in perf_phase_all]
        perf_component_all = [_np.pad(a, ((0, 0), (1, 0))) if a.size else _np.zeros((0, 3)) for a in perf_component_all]
        state['facility_perforation_phase_flux'] = perf_phase_all
        state['facility_perforation_component_flux'] = perf_component_all

        pv, pv0 = self._mrst_pore_volume_adi(p), _pv0fn(p0)
        invdt = 1.0 / max(float(dt), 1.0e-30)
        if self.vapoil:
            resO = (pv * (so * bO + rv * bG * sg) * rhoOS -
                    (pv0 * (so0 * pvt0['bo'] + rv0 * pvt0['bg'] * sg0) * rhoOS)) * invdt + divO - srcO
        else:
            resO = (pv * so * rhoOS * bO - pv0 * so0 * rhoOS * pvt0['bo']) * invdt + divO - srcO
        resG = (pv * (sg * rhoG_component + so * rs * rhoGS * bO) -
                (pv0 * (sg0 * rhoG_component0 + so0 * rs0 * rhoGS * pvt0['bo']))) * invdt + divG - srcG
        scaleO = rhoOS / _np.mean(rhoO0)
        scaleG = rhoGS / _np.mean(rhoG_phase0)
        fO, fG = (qos - surface_o) * scaleO, (qgs - surface_g) * scaleG
        closure = _FacilityModel.compute_control_equations(
            wells, qs_phases={'o': qos, 'g': qgs}, bhp=bhp, phase_order=['o', 'g'],
        )
        residual = _SparseADI.concat((resO, resG, fO, fG,
                                      _SparseADI.concat(closure) if closure else _SparseADI.constant(_np.zeros(0), nvar)))
        return residual, {'status': status, 'pvt': pvt}

    def _get_equations_mrst_generic_og(self, state0, state, dt, drivingForces, **kwargs):
        if drivingForces is None:
            drivingForces = {}
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        self._mrst_write_wellsol(state, wells)
        # ``getStateAD(state, ~resOnly)``: resOnly means the state carries no
        # derivatives, which is the same condition reverseMode sets.  Both
        # were ignored here, so an adjoint or a parameter sensitivity built
        # on a oil-gas model differentiated with respect to the wrong
        # variables -- silently, because the assembly still returned a
        # perfectly well-formed system.
        res_only = bool(kwargs.get('ResOnly', False))
        reverse = bool(kwargs.get('reverseMode', False)) or res_only
        assembled, meta = self._mrst_generic_adi_residual_og(
            state0, state, dt, drivingForces, wells, reverseMode=reverse)
        residual = assembled.val
        if res_only:
            jacobian = (_sp.csr_matrix((residual.size, residual.size))
                        if _sp is not None
                        else _np.zeros((residual.size, residual.size)))
        else:
            jacobian = assembled.jac
        nc, nw = self._num_cells(), len(wells)
        status = meta['status']
        gvar = 'x' if self.disgas else 'sG'
        problem = {
            'Residuals': residual, 'Jacobian': jacobian,
            'State': state, 'State0': state0, 'dt': float(dt), 'drivingForces': drivingForces,
            'equationNames': ['oil'] * nc + ['gas'] * nc + ['oilWells'] * nw + ['gasWells'] * nw + ['closureWells'] * nw,
            'types': ['cell'] * (2 * nc) + ['perf'] * (2 * nw) + ['well'] * nw,
            'blackOilStatus': status,
            'primaryVariables': ['pressure', gvar, 'qOs', 'qGs', 'bhp'],
            'facilityPrimaryVariables': ['qOs', 'qGs', 'bhp'],
            '_assembled': assembled,
            'wellSol': state.get('wellSol', []),
            'rs': _np.asarray(state.get('rs', _np.zeros(nc)), dtype=float).ravel(),
            'rv': _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel(),
        }
        return problem, state

    def _get_equations_mrst_generic_ow(self, state0, state, dt, drivingForces, **kwargs):
        if drivingForces is None:
            drivingForces = {}
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        self._mrst_write_wellsol(state, wells)
        # ``getStateAD(state, ~resOnly)``: resOnly means the state carries no
        # derivatives, which is the same condition reverseMode sets.  Both
        # were ignored here, so an adjoint or a parameter sensitivity built
        # on a two-phase model differentiated with respect to the wrong
        # variables -- silently, because the assembly still returned a
        # perfectly well-formed system.
        res_only = bool(kwargs.get('ResOnly', False))
        reverse = bool(kwargs.get('reverseMode', False)) or res_only
        assembled, meta = self._mrst_generic_adi_residual_ow(
            state0, state, dt, drivingForces, wells, reverseMode=reverse)
        residual = assembled.val
        if res_only:
            jacobian = (_sp.csr_matrix((residual.size, residual.size))
                        if _sp is not None
                        else _np.zeros((residual.size, residual.size)))
        else:
            jacobian = assembled.jac
        nc, nw = self._num_cells(), len(wells)
        problem = {
            'Residuals': residual, 'Jacobian': jacobian,
            'State': state, 'State0': state0, 'dt': float(dt), 'drivingForces': drivingForces,
            'equationNames': ['water'] * nc + ['oil'] * nc + ['waterWells'] * nw + ['oilWells'] * nw + ['closureWells'] * nw,
            'types': ['cell'] * (2 * nc) + ['perf'] * (2 * nw) + ['well'] * nw,
            'primaryVariables': ['pressure', 'sW', 'qWs', 'qOs', 'bhp'],
            'facilityPrimaryVariables': ['qWs', 'qOs', 'bhp'],
            '_assembled': assembled,
            'wellSol': state.get('wellSol', []),
        }
        return problem, state

    def _get_equations_mrst_generic(self, state0, state, dt, drivingForces, **kwargs):
        if drivingForces is None:
            drivingForces = {}
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        # GenericFacilityModel keeps the active control in ``state.wellSol``.
        # A limit can be reached during Newton iterations (not only at the
        # mini-step boundary), in which case the next assembled closure
        # equation must use the switched target.  This is the
        # applyWellLimitsWellSol path used by MRST's facility assembly.
        self._mrst_write_wellsol(state, wells)
        # state0 does not carry primary facility variables on the first
        # report step, but its density/PV state is all that the reservoir
        # equations require.
        # ``getStateAD(state, ~resOnly)``: resOnly means the state is not
        # seeded, exactly as reverseMode does. Zeroing the Jacobian
        # afterwards is not the same thing -- the assembled object would
        # still carry derivatives with respect to the states, which is
        # what partialWRTparam relies on it *not* doing: there the only
        # derivatives are meant to be the parameters'.
        res_only = bool(kwargs.get('ResOnly', False))
        assembled, meta = self._mrst_generic_adi_residual(
            state0, state, dt, drivingForces, wells,
            reverseMode=bool(kwargs.get('reverseMode', False)) or res_only
        )
        residual = assembled.val
        if res_only:
            # MRST's resOnly skips the derivatives; here the residual is
            # the value half of the AD object, so it is assembled either
            # way and only the Jacobian is dropped. A second, numeric
            # expression of the same equations used to serve this branch
            # -- it had drifted into referring to variables that no
            # longer existed and raised on the first call, which nothing
            # noticed because no black-oil model sets
            # stepFunctionIsLinear. One expression cannot drift from
            # itself.
            jacobian = _sp.csr_matrix((residual.size, residual.size)) \
                if _sp is not None \
                else _np.zeros((residual.size, residual.size))
        else:
            jacobian = assembled.jac
        status = meta['status']
        nc = self._num_cells()
        nw = len(wells)
        names = (['water'] * nc + ['oil'] * nc + ['gas'] * nc +
                 ['waterWells'] * nw + ['oilWells'] * nw +
                 ['gasWells'] * nw + ['closureWells'] * nw)
        types = (['cell'] * (3 * nc) + ['perf'] * (3 * nw) + ['well'] * nw)
        self._mrst_write_wellsol(state, wells)
        problem = {
            'Residuals': residual, 'Jacobian': jacobian,
            # The AD object itself, not just its two halves. MRST's
            # problem.equations are ADI and computeSensitivitiesAdjointAD
            # forms ``lambda' * eqdth`` on them -- an inner product that
            # yields ``lambda^T dR/dtheta`` in one Jacobian, without ever
            # building dR/dtheta as a matrix. Splitting into value and
            # Jacobian up front throws that away.
            '_assembled': assembled,
            'State': state, 'State0': state0, 'dt': float(dt),
            'drivingForces': drivingForces, 'equationNames': names,
            'types': types, 'blackOilStatus': status,
            'primaryVariables': ['pressure', 'sW', 'x', 'qWs', 'qOs', 'qGs', 'bhp'],
            'facilityPrimaryVariables': ['qWs', 'qOs', 'qGs', 'bhp'],
            'wellSol': state.get('wellSol', []),
            'rs': _np.asarray(state['rs'], dtype=float).ravel(),
            'rv': _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel(),
        }
        return problem, state

    def _mrst_flash_blackoil(self, state, state_before, status):
        """Numerical port of ``computeFlashBlackOil.m`` for black oil."""
        sw = _np.asarray(state['sW'], dtype=float).ravel().copy()
        sg = _np.asarray(state['sG'], dtype=float).ravel().copy()
        so = 1.0 - sw - sg
        sw0 = _np.asarray(state_before['sW'], dtype=float).ravel()
        sg0 = _np.asarray(state_before['sG'], dtype=float).ravel()
        so0 = 1.0 - sw0 - sg0
        rs = _np.asarray(state['rs'], dtype=float).ravel().copy()
        rs0 = _np.asarray(state_before['rs'], dtype=float).ravel()
        rv = _np.asarray(state.get('rv', _np.zeros_like(rs)), dtype=float).ravel().copy()
        rv0 = _np.asarray(state_before.get('rv', _np.zeros_like(rs)), dtype=float).ravel()
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        p0 = _np.asarray(state_before['pressure'], dtype=float).ravel()
        etol = _np.sqrt(_np.finfo(float).eps)
        wat_only = sw > 1.0 - etol

        if self.disgas:
            st1 = _np.asarray(status[0], dtype=bool)
            rs_sat0 = self._phase_pvt(p0)['rs']
            rs_sat = self._phase_pvt(p)['rs']
            gas_present = (((sg > 0.0) | (rs == 0.0)) & ~st1) | wat_only
            ix1 = (sg < 0.0) & (sg0 > etol)
            gas_present = gas_present | ix1
            ix2 = ((rs > rs_sat * (1.0 + etol)) & st1 &
                   (rs0 > rs_sat0 * (1.0 - etol)))
            sg[ix2] = 0.0
            gas_present = gas_present | ix2
        else:
            rs_sat = rs0
            gas_present = _np.ones_like(sw, dtype=bool)

        ix = sg < 0.0
        if _np.any(ix):
            sw[ix] = sw[ix] / (1.0 - sg[ix])
            so[ix] = so[ix] / (1.0 - sg[ix])
            sg[ix] = 0.0

        if not self.vapoil:
            oil_present = _np.ones_like(sw, dtype=bool)
            rv_sat = rv0
        else:
            st2 = _np.asarray(status[1], dtype=bool)
            _, _, pg0 = self._phase_pressures(p0, sw0, sg0)
            _, _, pg = self._phase_pressures(p, sw, sg)
            pvt = getattr(self, '_blackoil_pvt', None)
            if pvt is None or not hasattr(pvt, 'rv_sat'):
                raise NotImplementedError('Deck VAPOIL flash requires DeckBlackOilPVT.rv_sat')
            # Literal computeFlashBlackOil.m phase-transition branch.
            rv_sat0 = pvt.rv_sat(pg0)
            rv_sat = pvt.rv_sat(pg)
            oil_present = (((so > 0.0) | (rv == 0.0)) & ~st2) | wat_only
            ix1 = (so < 0.0) & (so0 > etol)
            oil_present = oil_present | ix1
            ix2 = ((rv > rv_sat * (1.0 + etol)) & st2 &
                   (rv0 > rv_sat0 * (1.0 - etol)))
            so[ix2] = 0.0
            oil_present = oil_present | ix2
        ix = so < 0.0
        if _np.any(ix):
            sw[ix] = sw[ix] / (1.0 - so[ix])
            sg[ix] = sg[ix] / (1.0 - so[ix])
            so[ix] = 0.0
        ix = sw < 0.0
        if _np.any(ix):
            so[ix] = so[ix] / (1.0 - sw[ix])
            sg[ix] = sg[ix] / (1.0 - sw[ix])
            sw[ix] = 0.0

        rs[gas_present] = rs_sat[gas_present]
        rs[~gas_present] = _np.minimum(rs_sat[~gas_present], rs[~gas_present])
        rv[oil_present] = rv_sat[oil_present]
        total = sw + so + sg
        total = _np.where(_np.abs(total) > 1.0e-30, total, 1.0)
        state['sW'] = sw / total
        state['sG'] = sg / total
        state['rs'] = _np.maximum(rs, 0.0)
        state['rv'] = _np.maximum(rv, 0.0)
        state['status'] = oil_present.astype(int) + 2 * gas_present.astype(int)
        return state

    def _update_state_mrst_generic_ow(self, state, problem, dx, drivingForces):
        """Port the two-phase portion of ``ReservoirModel.updateSaturations``."""
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        nc, nw = self._num_cells(), len(wells)
        if dx.size != 2 * nc + 3 * nw:
            raise ValueError('Expected two-phase MRST generic update of length %d, got %d' % (2 * nc + 3 * nw, dx.size))
        p0 = _np.asarray(state['pressure'], dtype=float).ravel()
        dp = dx[:nc]
        state['pressure'] = self.limit_pressure_increment(p0, dp)
        sw0 = _np.asarray(state['sW'], dtype=float).ravel()
        dsw = dx[nc:2 * nc]
        # ReservoirModel.updateSaturations clips and renormalizes the two
        # phases after the simultaneous increment.
        state['sW'] = _np.clip(self.limit_saturation_increment(sw0, dsw), 0.0, 1.0)
        start = 2 * nc
        state['facility_qWs'] = _np.asarray(state['facility_qWs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_qOs'] = _np.asarray(state['facility_qOs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_bhp'] = _np.asarray(state['facility_bhp'], dtype=float) + dx[start:start + nw]
        # GenericBlackOilModel.updateState applies facility well limits
        # *after* the Newton increment.  Applying them while assembling an
        # equation changes a rate-control closure before it has been
        # solved, which is not the MRST execution order.
        self._mrst_apply_well_limits(state, wells)
        self._mrst_write_wellsol(state, wells)
        return state

    def _update_state_mrst_generic(self, state, problem, dx, drivingForces):
        """Port ThreePhaseBlackOilModel.updateState for the SPE1 variables."""
        self.validateState(state)
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        nc, nw = self._num_cells(), len(wells)
        expected = 3 * nc + 4 * nw
        if dx.size != expected:
            raise ValueError('Expected MRST generic update of length %d, got %d' % (expected, dx.size))
        before = {
            'pressure': _np.asarray(state['pressure'], dtype=float).ravel().copy(),
            'sW': _np.asarray(state['sW'], dtype=float).ravel().copy(),
            'sG': _np.asarray(state['sG'], dtype=float).ravel().copy(),
            'rs': _np.asarray(state['rs'], dtype=float).ravel().copy(),
            'rv': _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel().copy(),
            'status': _np.asarray(state.get('status', _np.zeros(nc)), dtype=int).ravel().copy(),
        }
        status = problem.get('blackOilStatus', self._mrst_blackoil_status(state))
        st1, st2, st3 = (_np.asarray(status[0], dtype=bool),
                         _np.asarray(status[1], dtype=bool),
                         _np.asarray(status[2], dtype=bool))
        p = before['pressure']
        dp = dx[:nc]
        state['pressure'] = self.limit_pressure_increment(p, dp)

        dsw = dx[nc:2 * nc]
        dr = dx[2 * nc:3 * nc]
        if self.disgas or self.vapoil:
            dsg = st3 * dr - st2 * dsw
        else:
            # x is gas saturation itself; its increment is dSg directly.
            dsg = dr
        dso = -(dsg + dsw)
        sscale = self.shared_saturation_scale(dsw, dsg, dso)
        state['sW'] = before['sW'] + dsw * sscale
        state['sG'] = before['sG'] + dsg * sscale
        if self.disgas:
            state['rs'] = _np.maximum(before['rs'] + st1 * dr, 0.0)
        if self.vapoil:
            state['rv'] = _np.maximum(before['rv'] + st2 * dr, 0.0)
        state = self._mrst_flash_blackoil(state, before, (st1, st2, st3))

        start = 3 * nc
        state['facility_qWs'] = _np.asarray(state['facility_qWs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_qOs'] = _np.asarray(state['facility_qOs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_qGs'] = _np.asarray(state['facility_qGs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_bhp'] = _np.asarray(state['facility_bhp'], dtype=float) + dx[start:start + nw]
        self._mrst_apply_well_limits(state, wells)
        self._mrst_write_wellsol(state, wells)
        return state

    def _update_state_mrst_generic_og(self, state, problem, dx, drivingForces):
        """Port of the oil/gas-only branch of ``ThreePhaseBlackOilModel.
        updateState`` (``model.water`` false, mirroring
        :meth:`_update_state_mrst_generic` with no ``sW`` increment, so
        ``dso = -dsg`` instead of ``-(dsg + dsw)``)."""
        self.validateState(state)
        wells = self._ensure_mrst_facility_state(state, drivingForces)
        nc, nw = self._num_cells(), len(wells)
        expected = 2 * nc + 3 * nw
        if dx.size != expected:
            raise ValueError('Expected two-phase oil/gas MRST generic update of length %d, got %d' % (expected, dx.size))
        before = {
            'pressure': _np.asarray(state['pressure'], dtype=float).ravel().copy(),
            'sW': _np.zeros(nc),
            'sG': _np.asarray(state.get('sG', _np.zeros(nc)), dtype=float).ravel().copy(),
            'rs': _np.asarray(state.get('rs', _np.zeros(nc)), dtype=float).ravel().copy(),
            'rv': _np.asarray(state.get('rv', _np.zeros(nc)), dtype=float).ravel().copy(),
            'status': _np.asarray(state.get('status', _np.zeros(nc)), dtype=int).ravel().copy(),
        }
        status = problem.get('blackOilStatus', self._mrst_blackoil_status(state))
        st1, st2, st3 = (_np.asarray(status[0], dtype=bool),
                         _np.asarray(status[1], dtype=bool),
                         _np.asarray(status[2], dtype=bool))
        p = before['pressure']
        dp = dx[:nc]
        state['pressure'] = self.limit_pressure_increment(p, dp)

        dr = dx[nc:2 * nc]
        if self.disgas or self.vapoil:
            dsg = st3 * dr
        else:
            dsg = dr
        dso = -dsg
        dsw = _np.zeros(nc)
        sscale = self.shared_saturation_scale(dsw, dsg, dso)
        state['sW'] = dsw
        state['sG'] = before['sG'] + dsg * sscale
        if self.disgas:
            state['rs'] = _np.maximum(before['rs'] + st1 * dr, 0.0)
        if self.vapoil:
            state['rv'] = _np.maximum(before['rv'] + st2 * dr, 0.0)
        state = self._mrst_flash_blackoil(state, before, (st1, st2, st3))

        start = 2 * nc
        state['facility_qOs'] = _np.asarray(state['facility_qOs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_qGs'] = _np.asarray(state['facility_qGs'], dtype=float) + dx[start:start + nw]; start += nw
        state['facility_bhp'] = _np.asarray(state['facility_bhp'], dtype=float) + dx[start:start + nw]
        self._mrst_apply_well_limits(state, wells)
        self._mrst_write_wellsol(state, wells)
        return state

    def _get_equations_ow(self, state0, state, dt, drivingForces, **kwargs):
        """Oil-water 2-equation model (pressure + water saturation)."""
        nc = self._num_cells()
        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sW = _np.asarray(state['sW'], dtype=float).ravel()
        p0 = self._state0_value(state0['pressure'])
        sW0 = self._state0_value(state0['sW'])

        if drivingForces is None:
            drivingForces = {}
        # MRST equationsBlackOil uses cell-wise model.getProps(..., 'PoreVolume').
        pv = self._porevolume_vector()

        inv_dt = 1.0 / max(float(dt), 1e-30)
        pvt = self._phase_pvt(p, rs_override=state.get('rs'), rv_override=state.get('rv'))
        lamW, lamO, _, pvt = self._three_phase_mobility(
            p, sW, _np.zeros_like(sW),
            rs_override=state.get('rs'), rv_override=state.get('rv'),
        )
        div_w, div_o, _, Lw, Lo, _ = self._assemble_flux_divergence(
            p, lamW, lamO, _np.zeros_like(lamW), pvt=pvt,
        )
        src_w, src_o, _, dsw_dp, dso_dp, _, well_sol = \
            self._well_sources(p, sW, _np.zeros_like(sW), drivingForces, lamW, lamO, _np.zeros_like(lamW), pvt)

        # MRST equations use shrinkage factors b = 1/B in accumulation.
        bW = 1.0 / _np.maximum(pvt['bw'], 1e-30)
        bO = 1.0 / _np.maximum(pvt['bo'], 1e-30)
        bW0 = 1.0 / _np.maximum(self._phase_pvt(p0, rs_override=state0.get('rs'), rv_override=state0.get('rv'))['bw'], 1e-30)
        bO0 = 1.0 / _np.maximum(self._phase_pvt(p0, rs_override=state0.get('rs'), rv_override=state0.get('rv'))['bo'], 1e-30)
        # 2-equation: pressure (total balance) + water
        comp = self.totalCompressibility * pv * (p - p0) * inv_dt
        acc_w = (pv * bW * sW - pv * bW0 * sW0) * inv_dt
        acc_o = (pv * bO * (1.0 - sW) - pv * bO0 * (1.0 - sW0)) * inv_dt
        res_p = comp + (div_w + div_o) - (src_w + src_o)
        res_w = acc_w + div_w - src_w
        residuals = _np.concatenate([res_p, res_w])

        if _sp is not None:
            rows_w, cols_w, vals_w = Lw
            rows_o, cols_o, vals_o = Lo
            Jwp = _sp.csr_matrix((vals_w, (rows_w, cols_w)), shape=(nc, nc))
            Jop = _sp.csr_matrix((vals_o, (rows_o, cols_o)), shape=(nc, nc))
            if _np.any(dsw_dp):
                Jwp = Jwp - _sp.diags(dsw_dp, 0, shape=(nc, nc), format='csr')
            if _np.any(dso_dp):
                Jop = Jop - _sp.diags(dso_dp, 0, shape=(nc, nc), format='csr')
            Jpp = Jwp + Jop + _sp.diags(self.totalCompressibility * pv * inv_dt, 0, shape=(nc, nc), format='csr')
            Jps = _sp.csr_matrix((nc, nc), dtype=float)
            Jsp = Jwp
            Jss = _sp.diags(pv * bW * inv_dt, 0, shape=(nc, nc), format='csr')
            jacobian = _sp.bmat([[Jpp, Jps], [Jsp, Jss]], format='csr')
        else:
            jacobian = _np.eye(2 * nc, dtype=float)

        names = ['pressure'] * nc + ['water'] * nc
        types = ['cell'] * (2 * nc)

        oil_present = (1.0 - sW0 - _np.zeros_like(sW0)) > _np.sqrt(_np.finfo(float).eps)
        gas_present = _np.zeros_like(sW0, dtype=bool)
        black_oil_status = (oil_present & ~gas_present, ~oil_present & gas_present, oil_present & gas_present)
        residuals, jacobian, names, types, state = self._augment_facility_system(
            residuals, jacobian, names, types, state, well_sol, drivingForces, nc,
        )
        problem = {
            'Residuals': residuals, 'Jacobian': jacobian,
            'State': state, 'State0': state0, 'dt': float(dt),
            'drivingForces': drivingForces,
            'equationNames': names, 'types': types,
            'blackOilStatus': black_oil_status,
            'facilityPrimaryVariables': state.get('facility_primary_variables', []),
            'wellSol': well_sol,
            'rs': pvt['rs'], 'rv': pvt['rv'],
            'bw': pvt['bw'], 'bo': pvt['bo'], 'bg': pvt['bg'],
        }
        state['rs'] = _np.asarray(pvt['rs'], dtype=float).ravel()
        state['rv'] = _np.asarray(pvt['rv'], dtype=float).ravel()
        return problem, state

    def _get_equations_3ph(self, state0, state, dt, drivingForces, **kwargs):
        """Three-phase black-oil conservation equations with Rs/Rv mass transfer.

        Primary variables: pressure, sW, sG.
        Equations: water, oil, gas conservation at surface conditions.
        """
        state = self.validateState(state)
        state0 = self.validateState(state0)
        nc = self._num_cells()

        p = _np.asarray(state['pressure'], dtype=float).ravel()
        sW = _np.asarray(state['sW'], dtype=float).ravel()
        sG = _np.asarray(state['sG'], dtype=float).ravel()
        p0 = self._state0_value(state0['pressure'])
        sW0 = self._state0_value(state0['sW'])
        sG0 = self._state0_value(state0['sG'])

        sO = _np.clip(1.0 - sW - sG, 0.0, 1.0)
        sO0 = _np.clip(1.0 - sW0 - sG0, 0.0, 1.0)

        if drivingForces is None:
            drivingForces = {}

        # MRST equationsBlackOil uses cell-wise PoreVolume in all accumulation terms.
        pv = self._porevolume_vector()

        inv_dt = 1.0 / max(float(dt), 1e-30)

        # ---- PVT evaluation at current and previous pressure ----
        pvt = self._phase_pvt(
            p,
            rs_override=state.get('rs'),
            rv_override=state.get('rv'),
        )
        pvt0 = self._phase_pvt(
            p0,
            rs_override=state0.get('rs'),
            rv_override=state0.get('rv'),
        )

        bw, bo, bg = pvt['bw'], pvt['bo'], pvt['bg']
        bw0, bo0, bg0 = pvt0['bw'], pvt0['bo'], pvt0['bg']
        # DeckBlackOilPVT.eval now returns MRST shrinkage factors b directly.
        bW, bO, bG = bw, bo, bg
        bW0, bO0, bG0 = bw0, bo0, bg0
        rs, rv = pvt['rs'], pvt['rv']
        rs0, rv0 = pvt0['rs'], pvt0['rv']

        # ---- Mobilities and flux divergence ----
        lamW, lamO, lamG, _ = self._three_phase_mobility(
            p, sW, sG,
            rs_override=state.get('rs'), rv_override=state.get('rv'),
        )
        div_w, div_o, div_g, Lw, Lo, Lg = self._assemble_flux_divergence(
            p, lamW, lamO, lamG, pvt=pvt,
        )

        # Upstream-weight B-factors on fluxes (simplified: use cell B for now)
        # Full upstream weighting would use face-upstream B values.
        # For surface-condition conservation, flux = B * v_reservoir
        # Here div_* already represents reservoir-condition flux divergence.
        # We approximate the B * divergence as the divergence scaled by cell B.

        # ---- Well sources ----
        src_w, src_o, src_g, dsw_dp, dso_dp, dsg_dp, well_sol = \
            self._well_sources(p, sW, sG, drivingForces, lamW, lamO, lamG, pvt)

        # ---- Accumulation terms at surface conditions (MRST b=1/B) ----
        # Water: (pv * bW * sW - pv0 * bW0 * sW0) / dt
        acc_w = (pv * bW * sW - pv * bW0 * sW0) * inv_dt

        # Oil: pv*(bO*sO + rv*bG*sG), with vaporized oil if enabled.
        if self.vapoil:
            acc_o = (pv * (bO * sO + rv * bG * sG) -
                     pv * (bO0 * sO0 + rv0 * bG0 * sG0)) * inv_dt
        else:
            acc_o = (pv * bO * sO - pv * bO0 * sO0) * inv_dt

        # Gas: pv*(bG*sG + rs*bO*sO), with dissolved gas if enabled.
        if self.disgas:
            acc_g = (pv * (bG * sG + rs * bO * sO) -
                     pv * (bG0 * sG0 + rs0 * bO0 * sO0)) * inv_dt
        else:
            acc_g = (pv * bG * sG - pv * bG0 * sG0) * inv_dt

        # ---- Residuals = Accumulation + Flux - Source ----
        res_w = acc_w + div_w - src_w
        res_o = acc_o + div_o - src_o
        res_g = acc_g + div_g - src_g

        residuals = _np.concatenate([res_w, res_o, res_g])

        # MRST getCellStatusVO: x is Rs in oil-only cells, Rv in gas-only
        # cells, and Sg only where both hydrocarbon phases are present.
        oil_present0 = (1.0 - sW0 - sG0) > _np.sqrt(_np.finfo(float).eps)
        gas_present0 = sG0 > _np.sqrt(_np.finfo(float).eps)
        st1 = oil_present0 & ~gas_present0
        st2 = ~oil_present0 & gas_present0
        st3 = oil_present0 & gas_present0

        # ---- Sparse Jacobian assembly ----
        if _sp is not None:
            rows_w, cols_w, vals_w = Lw
            rows_o, cols_o, vals_o = Lo
            rows_g, cols_g, vals_g = Lg

            # Flux Jacobian blocks (dFlux/dp)
            Jwp = _sp.csr_matrix((vals_w, (rows_w, cols_w)), shape=(nc, nc))
            Jop = _sp.csr_matrix((vals_o, (rows_o, cols_o)), shape=(nc, nc))
            Jgp = _sp.csr_matrix((vals_g, (rows_g, cols_g)), shape=(nc, nc))

            # Well source pressure derivatives
            if _np.any(dsw_dp):
                Jwp = Jwp - _sp.diags(dsw_dp, 0, shape=(nc, nc), format='csr')
            if _np.any(dso_dp):
                Jop = Jop - _sp.diags(dso_dp, 0, shape=(nc, nc), format='csr')
            if _np.any(dsg_dp):
                Jgp = Jgp - _sp.diags(dsg_dp, 0, shape=(nc, nc), format='csr')

            # Accumulation Jacobian blocks (diagonal approximations)
            # Water: d(acc_w)/dp ≈ pv * bw * sW * cw, d(acc_w)/dsW ≈ pv * bw
            # Add minimum pressure coupling for all cells to avoid singular blocks
            min_ct = max(self.totalCompressibility, 1e-8)
            bW = bw; bO = bo; bG = bg
            bW0 = bw0; bO0 = bo0; bG0 = bg0
            Jwp = Jwp + _sp.diags(pv * bW * _np.maximum(sW, 0.01) * min_ct * inv_dt, 0,
                                  shape=(nc, nc), format='csr')
            Jws = _sp.diags(pv * bW * inv_dt, 0, shape=(nc, nc), format='csr')
            Jwsg = _sp.csr_matrix((nc, nc), dtype=float)

            # Oil: d(acc_o)/dp
            Jop = Jop + _sp.diags(pv * bO * _np.maximum(sO, 0.01) * min_ct * inv_dt, 0,
                                  shape=(nc, nc), format='csr')
            Jos = _sp.diags(-pv * bO * inv_dt, 0, shape=(nc, nc), format='csr')
            Josg = _sp.diags(-pv * bO * inv_dt, 0, shape=(nc, nc), format='csr')
            if self.vapoil:
                Josg = Josg + _sp.diags(pv * rv * bg * inv_dt, 0, shape=(nc, nc), format='csr')

            # Gas: d(acc_g)/dp
            Jgp = Jgp + _sp.diags(pv * bG * _np.maximum(sG, 0.01) * min_ct * inv_dt, 0,
                                  shape=(nc, nc), format='csr')
            Jgs = _sp.diags(-pv * rs * bO * inv_dt, 0, shape=(nc, nc), format='csr') if self.disgas else _sp.csr_matrix((nc, nc), dtype=float)
            # d(gas)/dSg, valid only where x denotes Sg (st3).
            Jg_sg = _sp.diags(pv * bG * inv_dt, 0, shape=(nc, nc), format='csr')
            if self.disgas:
                Jg_sg = Jg_sg + _sp.diags(pv * rs * bO * inv_dt, 0, shape=(nc, nc), format='csr')

            # MRST x-column: only status-switches between Rs/Rv/Sg (masked
            # by st1/st2/st3) when a hydrocarbon phase can actually change
            # identity, i.e. when disgas or vapoil is active
            # (equationsBlackOil.m: ``x = st1.*rs + st2.*rv + st3.*sG``).
            # When *neither* is active, MRST's own model takes a separate,
            # simpler branch instead (``else: x = sG; gvar = 'sG'``): x is
            # directly sG in every cell, not status-gated at all. Applying
            # the st3 mask unconditionally here made the gas-Jacobian
            # column identically zero (structurally singular) for any
            # no-disgas/no-vapoil deck whose cells start with sG=0
            # everywhere -- st3 (oil AND gas both present) is then false
            # everywhere, with no Rs/Rv term to compensate since disgas and
            # vapoil are both off.
            Jw_x = Jwsg
            if self.disgas or self.vapoil:
                Jo_x = Josg.multiply(_sp.diags(st3.astype(float), format='csr'))
                if self.vapoil:
                    Jo_x = Jo_x + _sp.diags(st2.astype(float) * pv * bG * sG * inv_dt, 0, format='csr')
                Jg_x = Jg_sg.multiply(_sp.diags(st3.astype(float), format='csr'))
                if self.disgas:
                    Jg_x = Jg_x + _sp.diags(st1.astype(float) * pv * bO * sO * inv_dt, 0, format='csr')
            else:
                Jo_x = Josg
                Jg_x = Jg_sg

            jacobian = _sp.bmat([[Jwp, Jws, Jw_x],
                                 [Jop, Jos, Jo_x],
                                 [Jgp, Jgs, Jg_x]], format='csr')
        else:
            jacobian = _np.eye(3 * nc, dtype=float)

        names = ['water'] * nc + ['oil'] * nc + ['gas'] * nc
        types = ['cell'] * (3 * nc)

        black_oil_status = (st1, st2, st3)
        residuals, jacobian, names, types, state = self._augment_facility_system(
            residuals, jacobian, names, types, state, well_sol, drivingForces, nc,
        )
        problem = {
            'Residuals': residuals,
            'Jacobian': jacobian,
            'State': state,
            'State0': state0,
            'dt': float(dt),
            'drivingForces': drivingForces,
            'equationNames': names,
            'types': types,
            'blackOilStatus': black_oil_status,
            'facilityPrimaryVariables': state.get('facility_primary_variables', []),
            'wellSol': well_sol,
            'rs': rs, 'rv': rv,
            'bw': bw, 'bo': bo, 'bg': bg,
        }
        state['rs'] = _np.asarray(rs, dtype=float).ravel()
        state['rv'] = _np.asarray(rv, dtype=float).ravel()
        return problem, state

    def _num_cells(self):
        if isinstance(self.G, dict) and 'cells' in self.G and 'num' in self.G['cells']:
            return int(self.G['cells']['num'])
        return 1

    @staticmethod
    def _param_value(value, default=None):
        """An operator the parameters live in, letting an AD value through.

        Pore volume and transmissibility are what a history match tunes,
        and the adjoint needs dR/dp for each. Both enter the residual
        linearly -- pv scales the accumulation, T scales the flux -- so
        seeding them as AD variables and letting the existing arithmetic
        carry the derivative gives dR/dp exactly, with no term written
        out by hand to drift from the assembly it mirrors.

        For a plain array this is the cast it replaces, so an ordinary
        evaluation is unchanged.
        """
        if value is None:
            value = default
        if hasattr(value, 'val'):
            return value
        return _np.asarray(value, dtype=float).ravel()

    def _internal_connections(self):
        """Return ``(c1, c2, T)`` as 0-based cell indices for the TPFA stencil.

        ``operators['N']`` arrives in either convention: the logical
        Cartesian ``setup_operators`` emits 1-based indices (MRST's own
        convention), while ``setup_operators_tpfa`` and the nwm hybrid-grid
        assembly emit 0-based ones.  Deciding between them with
        ``min(N) >= 1`` -- as this used to -- misreads a 0-based list whose
        cell 0 happens to take part in no connection, silently shifting
        *every* index by one.  Prefer an explicit marker, and otherwise
        discriminate on the maximum: a 0-based index can never reach ``nc``.
        """
        ops = self.operators or {}
        N = _np.asarray(ops.get('N', _np.zeros((0, 2))), dtype=int)
        T = self._param_value(ops.get('T'), _np.zeros(0))
        if N.ndim != 2 or N.shape[1] < 2:
            N = _np.zeros((0, 2), dtype=int)
        one_based = ops.get('oneBased')
        if one_based is None:
            one_based = bool(N.size) and int(_np.max(N)) >= self._num_cells()
        if one_based:
            c1, c2 = N[:, 0] - 1, N[:, 1] - 1
        else:
            c1, c2 = N[:, 0], N[:, 1]
        nface = min(T.val.size if hasattr(T, 'val') else T.size, c1.size)
        return c1[:nface], c2[:nface], T[:nface]

    def _average_porevolume(self):
        """Return the legacy scalar pore volume used by compatibility paths."""
        return float(_np.mean(self._porevolume_vector()))

    def _porevolume_vector(self):
        """Port of MRST ``poreVolume(G, rock)``: ``poro .* volumes .* ntg``.

        ``setupOperatorsTPFA`` stores this as ``operators.pv``; PRSTCore
        keeps it on the model instead, so this is the single place the
        accumulation terms obtain it.

        Cell volumes live under two different keys depending on which
        constructor built the grid: ``init_eclipse_grid`` writes a
        top-level ``cell_volumes``, whereas ``compute_geometry`` (and
        therefore ``process_grdecl``/``tensor_grid``/``cart_grid``, i.e.
        every grid the nwm pipeline builds) writes ``cells.volumes``.
        Only the first was consulted here, so an entire class of grids
        silently fell through to a unit pore volume -- an accumulation
        term wrong by whatever ``poro*volume*ntg`` happens to be, which
        converges quietly to the wrong answer rather than failing.
        """
        nc = self._num_cells()

        # ``setupOperatorsTPFA`` stores the pore volume in
        # ``operators.pv`` and MRST reads it from there -- which is why
        # ``ModelParameter``'s location for 'porevolume' is
        # ``{'operators','pv'}``. Consulting it first makes that the one
        # source, so a parameter seeded where MRST seeds it actually
        # reaches the accumulation term. Without this the seeding lands
        # somewhere nothing reads and the sensitivity comes back exactly
        # zero -- which is indistinguishable from a converged gradient.
        operators = self.operators if isinstance(self.operators, dict) else {}
        for candidate in (operators.get('pv'), self.porevolume):
            if candidate is None:
                continue
            pv = self._param_value(candidate)
            if (pv.val.size if hasattr(pv, 'val') else pv.size) == nc:
                return pv

        vols = None
        if isinstance(self.G, dict):
            raw = self.G.get('cell_volumes')
            if raw is None:
                raw = self.G.get('cells', {}).get('volumes')
            if raw is not None:
                candidate = _np.asarray(raw, dtype=float).ravel()
                if candidate.size == nc:
                    vols = candidate
        if vols is None:
            return _np.ones(nc, dtype=float)

        rock = self.rock if isinstance(self.rock, dict) else {}
        pv = vols
        poro = rock.get('poro')
        if poro is not None:
            poro = _np.asarray(poro, dtype=float).ravel()
            if poro.size == nc:
                pv = pv * poro
        # poreVolume.m applies NTG only when the field is present.
        ntg = rock.get('ntg')
        if ntg is not None:
            ntg = _np.asarray(ntg, dtype=float).ravel()
            if ntg.size == 1:
                pv = pv * float(ntg[0])
            elif ntg.size == nc:
                pv = pv * ntg
        return pv

    def _cart_dims(self):
        if isinstance(self.G, dict):
            if 'cartDims' in self.G and self.G['cartDims'] is not None:
                dims = self.G['cartDims']
                if len(dims) >= 3:
                    return int(dims[0]), int(dims[1]), int(dims[2])
            if 'cells' in self.G and 'num' in self.G['cells']:
                n = int(self.G['cells']['num'])
                return n, 1, 1
        return 1, 1, 1


def make_generic_black_oil_model(G, rock, fluid):
    return GenericBlackOilModel(G=G, rock=rock, fluid=fluid)


def _krscale_fingerprint(rock):
    """A cheap identity for the endpoint table the scaling was built from.

    Sums are enough: the question is only whether ``rock.krscale`` is the
    same data as last time, and a tuned endpoint moves every entry of the
    column it owns.
    """
    if not isinstance(rock, dict):
        return None
    krscale = rock.get('krscale')
    if not isinstance(krscale, dict):
        return None
    key = []
    for table in sorted(krscale):
        entry = krscale[table]
        if not isinstance(entry, dict):
            continue
        for phase in sorted(entry):
            values = entry[phase]
            if values is None:
                continue
            arr = _np.asarray(values, dtype=float)
            key.append((table, phase, arr.shape,
                        float(_np.nansum(arr)),
                        float(_np.nansum(arr * arr))))
    return tuple(key)
