"""Tests for the ``PRSTCore.ad_eor`` port of MRST's ``autodiff/ad-eor``
module (polymer/surfactant EOR).

Two layers are covered:

- Unit tests for the pure ``properties/`` functions and the
  ``computeShearMult(Log)`` solvers, checked against hand-derived limiting
  cases (e.g. zero concentration -> unit multiplier) rather than against a
  reference MRST run (MATLAB is not runnable on this machine -- see prior
  session notes).
- A shape/mass-balance smoke test of ``equationsOilWaterPolymer`` on a tiny
  synthetic two-cell model, verifying the assembled Jacobian is finite,
  correctly shaped, and that the polymer equation's Jacobian is consistent
  with its residual under a finite-difference check.
"""

import numpy as np
import pytest

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_eor.properties.PolymerAdsorption import PolymerAdsorption
from PRSTCore.ad_eor.properties.PolymerEffViscMult import PolymerEffViscMult
from PRSTCore.ad_eor.properties.PolymerPermReduction import PolymerPermReduction
from PRSTCore.ad_eor.properties.PolymerPhaseFlux import PolymerPhaseFlux
from PRSTCore.ad_eor.properties.PolymerViscMult import PolymerViscMult
from PRSTCore.ad_eor.properties.SurfactantAdsorption import SurfactantAdsorption
from PRSTCore.ad_eor.properties.SurfactantCapillaryPressure import SurfactantCapillaryPressure
from PRSTCore.ad_eor.properties.SurfactantPhasePressures import SurfactantPhasePressures
from PRSTCore.ad_eor.properties.SurfactantRelativePermeability import SurfactantRelativePermeability
from PRSTCore.ad_eor.properties.SurfactantViscMultiplier import SurfactantViscMultiplier
from PRSTCore.ad_eor.utils.private.computeShearMult import computeShearMult
from PRSTCore.ad_eor.utils.private.computeShearMultLog import computeShearMultLog


def _polymer_fluid():
    return {
        'ads': lambda c: 0.5 * c / (1.0 + 0.1 * c),
        'adsInx': 1,
        'adsMax': 1.0,
        'rrf': 2.0,
        'muWMult': lambda c: 1.0 + 4.0 * c,
        'mixPar': 1.0,
        'cpmax': 2.0,
        'dps': 0.0,
        'rhoR': 0.0,
    }


class TestPolymerAdsorption:
    def test_zero_concentration_zero_adsorption(self):
        fluid = _polymer_fluid()
        ads = PolymerAdsorption(fluid, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        assert np.allclose(ads, 0.0)

    def test_monotone_in_concentration(self):
        fluid = _polymer_fluid()
        cp = np.array([0.0, 0.5, 1.0, 2.0])
        ads = PolymerAdsorption(fluid, cp, cp)
        assert np.all(np.diff(ads) >= 0.0)

    def test_hysteresis_uses_max(self):
        fluid = _polymer_fluid()
        fluid['adsInx'] = 2
        ads = PolymerAdsorption(fluid, np.array([0.1]), np.array([1.0]))
        expected = fluid['ads'](np.array([1.0]))
        assert np.allclose(ads, expected)


class TestPolymerViscosity:
    def test_zero_concentration_unit_multiplier(self):
        fluid = _polymer_fluid()
        cp = np.array([0.0, 0.0])
        assert np.allclose(PolymerViscMult(fluid, cp), 1.0)
        assert np.allclose(PolymerEffViscMult(fluid, cp), 1.0)

    def test_eff_visc_mult_matches_full_mult_at_cpmax(self):
        fluid = _polymer_fluid()
        cpmax = np.full(3, fluid['cpmax'])
        eff = PolymerEffViscMult(fluid, cpmax)
        full = fluid['muWMult'](cpmax)
        assert np.allclose(eff, full)

    def test_perm_reduction_floor_is_one(self):
        fluid = _polymer_fluid()
        permRed = PolymerPermReduction(fluid, np.array([0.0, 0.0]))
        assert np.allclose(permRed, 1.0)
        permRed_full = PolymerPermReduction(fluid, np.array([fluid['adsMax']]))
        assert np.allclose(permRed_full, fluid['rrf'])

    def test_phase_flux_zero_concentration_gives_zero_polymer_flux(self):
        fluid = _polymer_fluid()
        vW = np.array([1.0, -2.0])
        vP = PolymerPhaseFlux(fluid, vW, np.zeros(2))
        assert np.allclose(vP, 0.0)


def _surfactant_fluid():
    return {
        'surfads': lambda c: 0.2 * c,
        'adsInxSft': 1,
        'muWSft': lambda c: 1.0 + 2.0 * c,
        'muWr': 1.0,
        'ift': lambda c: 30.0 / (1.0 + 100.0 * c),
        'miscfact': lambda logNc: 1.0 / (1.0 + np.exp(-(logNc + 5.0))),
        'krW': lambda s: np.clip(s, 0.0, 1.0) ** 2,
        'krOW': lambda s: np.clip(s, 0.0, 1.0) ** 2,
    }


class TestSurfactantProperties:
    def test_adsorption_zero_at_zero_concentration(self):
        fluid = _surfactant_fluid()
        ads = SurfactantAdsorption(fluid, np.array([0.0]), np.array([0.0]))
        assert np.allclose(ads, 0.0)

    def test_visc_multiplier_is_one_at_zero_concentration(self):
        fluid = _surfactant_fluid()
        m = SurfactantViscMultiplier(fluid, np.array([0.0]))
        assert np.allclose(m, 1.0)

    def test_capillary_pressure_scales_by_ift_ratio(self):
        fluid = _surfactant_fluid()
        pcow = np.array([1000.0, 2000.0])
        cs = np.array([0.0, 0.01])
        scaled = SurfactantCapillaryPressure(fluid, pcow, cs)
        assert np.isclose(scaled[0], pcow[0])  # cs=0 -> ift ratio = 1
        assert scaled[1] < pcow[1]  # surfactant lowers IFT -> lowers pcow

    def test_capillary_pressure_none_passthrough(self):
        assert SurfactantCapillaryPressure(_surfactant_fluid(), None, np.array([0.0])) is None

    def test_phase_pressures_passthrough_for_empty_pc(self):
        p = np.array([100.0, 200.0])
        out = SurfactantPhasePressures(p, [None, np.array([5.0, 5.0])])
        assert np.allclose(out[0], p)
        assert np.allclose(out[1], p + 5.0)

    def test_relative_permeability_reduces_to_base_at_zero_capillary_number(self):
        fluid = _surfactant_fluid()
        sW = np.array([0.5])
        sO = np.array([0.5])
        cs = np.array([0.0])
        Nc = np.array([1.0e-30])
        krPts_base = {'w': 0.2, 'ow': 0.2}
        krPts_surf = {'w': 0.0, 'ow': 0.0}
        krW, krO = SurfactantRelativePermeability(fluid, sW, sO, None, cs, Nc, krPts_base, krPts_surf, False)
        krW_direct = fluid['krW']((sW - 0.2) / (1.0 - 0.4) * (1.0 - 0.4) + 0.2)
        assert np.allclose(krW, krW_direct, atol=1e-8)


class TestComputeShearMult:
    def test_trivial_shear_table_gives_unit_multiplier(self):
        # plyshearMult == 1 everywhere -> shFunc(x) = P*(x - Vw), whose root
        # is x = Vw regardless of P, and the returned shear multiplier
        # v = (1 + (P-1)*1)/P = 1 for any muWMultf.
        fluid = {'plyshearMult': lambda x: np.ones_like(x)}
        Vw = np.array([1.0, 2.0, 3.0])
        muWMultf = np.array([2.0, 5.0, 0.5])
        v = computeShearMult(fluid, Vw, muWMultf)
        assert np.allclose(v, 1.0, atol=1e-8)

    def test_shear_multiplier_matches_independent_root_find(self):
        """Solve MRST's EQ 52.12 (``Vsh*(1+(P-1)*M(Vsh))/P = Vw``) for
        ``Vsh`` independently via ``scipy.optimize.brentq`` per cell, then
        check the returned shear multiplier ``v = (1+(P-1)*M(Vsh))/P``
        matches -- an end-to-end check of ``computeShearMult`` against a
        solver with no shared code."""
        from scipy.optimize import brentq

        fluid = {'plyshearMult': lambda x: 1.0 / (1.0 + 0.1 * np.abs(x))}
        Vw = np.array([0.5, 5.0, 50.0])
        muWMultf = np.array([3.0, 3.0, 3.0])
        v = computeShearMult(fluid, Vw, muWMultf)

        expected = np.zeros_like(v)
        for i in range(Vw.size):
            def shFunc(x, i=i):
                M = fluid['plyshearMult'](np.array([x]))[0]
                return x * (1.0 + (muWMultf[i] - 1.0) * M) - muWMultf[i] * Vw[i]
            Vsh = brentq(shFunc, 1e-9, 10.0 * Vw[i] + 10.0)
            M = fluid['plyshearMult'](np.array([Vsh]))[0]
            expected[i] = (1.0 + (muWMultf[i] - 1.0) * M) / muWMultf[i]

        assert np.allclose(v, expected, atol=1e-6)


class TestComputeShearMultLog:
    def test_no_shear_below_min_table_velocity(self):
        fluid = {
            'plyshlog': {
                'refcondition': np.array([1.0]),
                'data': [np.array([[1.0, 1.0], [10.0, 0.5], [100.0, 0.2]])],
            },
            'muWMult': lambda c: np.full_like(np.atleast_1d(c), 3.0, dtype=float),
        }
        vW = np.array([0.5])  # below table min (1.0)
        muWMultf = np.array([3.0])
        zSh = computeShearMultLog(fluid, vW, muWMultf)
        assert np.allclose(zSh, 1.0)

    def test_no_shear_when_multiplier_not_above_one(self):
        fluid = {
            'plyshlog': {
                'refcondition': np.array([1.0]),
                'data': [np.array([[1.0, 1.0], [10.0, 0.5], [100.0, 0.2]])],
            },
            'muWMult': lambda c: np.full_like(np.atleast_1d(c), 3.0, dtype=float),
        }
        vW = np.array([50.0])
        muWMultf = np.array([1.0])  # not > 1 -> no shear cell selected
        zSh = computeShearMultLog(fluid, vW, muWMultf)
        assert np.allclose(zSh, 1.0)


def _two_cell_polymer_model():
    from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel
    from PRSTCore.ad_core.operators import setup_operators
    from PRSTCore.ad_core.initialization.pvt_tables import DeckBlackOilPVT
    from PRSTCore.ad_eor.utils.addPolymerProperties import addPolymerProperties

    G = {
        'type': 'tensor',
        'xfaces': np.array([0.0, 1.0, 2.0]),
        'yfaces': np.array([0.0, 1.0]),
        'zfaces': np.array([0.0, 1.0]),
        'cells': {'num': 2, 'volumes': np.array([1.0, 1.0]),
                  'centroids': np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])},
        'cartDims': (2, 1, 1),
    }
    rock = {'perm': np.array([100.0, 100.0]), 'poro': np.array([0.2, 0.2])}
    props = {
        'PVCDO': [100.0, 1.0, 0.0, 3.0, 0.0],
        'PVTW': [100.0, 1.0, 0.0, 1.0, 0.0],
        'DENSITY': [800.0, 1000.0, 1.0],  # ECLIPSE order: oil, water, gas
        'SWOF': [0.2, 0.0, 1.0, 0.0,
                 1.0, 1.0, 0.0, 0.0],
    }
    pvt = DeckBlackOilPVT(props)
    # getFluxAndPropsWaterPolymer_BO calls fluid['muW']/fluid['bW'] with a
    # SparseADI pressure during residual assembly, so these must be plain
    # elementwise arithmetic (AD-compatible via operator overloading), not
    # a numpy-only DeckBlackOilPVT.eval() call -- these are exact
    # assignPVCDO.m/assignPVTW.m constant-compressibility formulas with the
    # same PVCDO/PVTW coefficients as ``props`` above.
    por, bor, co, muor, vbo = props['PVCDO']
    p_ref, bw_ref, cw, mu_ref, cmu = props['PVTW']
    fluid = {
        'bO': lambda p: (co * (p - por)).exp() / bor if hasattr(p, 'exp') else np.exp(co * (p - por)) / bor,
        'muO': lambda p: muor * ((vbo * (p - por)).exp() if hasattr(p, 'exp') else np.exp(vbo * (p - por))),
        'bW': lambda p: (1.0 + cw * (p - p_ref) + 0.5 * (cw * (p - p_ref)) ** 2) / bw_ref,
        'muW': lambda p: mu_ref / (1.0 - cmu * (p - p_ref) + 0.5 * (cmu * (p - p_ref)) ** 2),
        'pcOW': None,
        'rhoWS': 1000.0,
        'rhoOS': 800.0,
    }
    fluid = addPolymerProperties(fluid, _polymer_fluid())
    model = GenericBlackOilModel(G=G, rock=rock, fluid=fluid, gas=False, mrst_generic_assembly=True,
                                  disgas=False, vapoil=False)
    model.operators = setup_operators(G, rock)
    model.inputdata = {'PROPS': props}
    model._blackoil_pvt = pvt
    return model


class TestEquationsOilWaterPolymerSmoke:
    def _states(self):
        state0 = {'pressure': np.array([100.0, 90.0]), 'sW': np.array([0.3, 0.3]),
                  'sG': np.zeros(2), 'polymer': np.array([0.0, 0.0]),
                  'polymermax': np.array([0.0, 0.0]), 'time': 0.0, 'wellSol': []}
        state = {'pressure': np.array([100.0, 90.0]), 'sW': np.array([0.32, 0.28]),
                 'sG': np.zeros(2), 'polymer': np.array([0.5, 0.1]),
                 'polymermax': np.array([0.5, 0.1]), 'time': 1.0, 'wellSol': []}
        return state0, state

    def test_residual_and_jacobian_shape(self):
        from PRSTCore.ad_eor.utils.equationsOilWaterPolymer import equationsOilWaterPolymer
        model = _two_cell_polymer_model()
        state0, state = self._states()
        residual, aux = equationsOilWaterPolymer(model, state0, state, 1.0, {'W': []}, [])
        assert residual.val.shape == (6,)  # 3 eqns x 2 cells, no wells
        assert residual.jac.shape == (6, 6)
        assert np.all(np.isfinite(residual.val))

    def test_polymer_row_jacobian_matches_finite_difference(self):
        """Spot-check d(resP)/d(cp) against a finite difference, cell by
        cell, since a wrong sign/scale in the polymer accumulation or
        adsorption term would be invisible from shape checks alone."""
        from PRSTCore.ad_eor.utils.equationsOilWaterPolymer import equationsOilWaterPolymer
        model = _two_cell_polymer_model()
        state0, state = self._states()

        def resP_at(cp_vals):
            s = dict(state)
            s['polymer'] = np.asarray(cp_vals, dtype=float)
            s['polymermax'] = np.maximum(state0['polymermax'], s['polymer'])
            residual, _ = equationsOilWaterPolymer(model, state0, s, 1.0, {'W': []}, [])
            return residual.val[4:6]  # polymer block, nc=2

        residual, _ = equationsOilWaterPolymer(model, state0, state, 1.0, {'W': []}, [])
        jac_polymer_block = residual.jac[4:6, 2 * 2:3 * 2].toarray()

        eps = 1e-6
        cp0 = state['polymer']
        fd = np.zeros((2, 2))
        base = resP_at(cp0)
        for j in range(2):
            bump = cp0.copy()
            bump[j] += eps
            fd[:, j] = (resP_at(bump) - base) / eps

        assert np.allclose(jac_polymer_block, fd, atol=1e-4, rtol=1e-3)


class TestGenericSurfactantPolymerModelDispatch:
    def test_polymer_only_dispatches_to_oil_water_polymer_model(self):
        from PRSTCore.ad_eor.models.GenericSurfactantPolymerModel import GenericSurfactantPolymerModel
        from PRSTCore.ad_eor.models.OilWaterPolymerModel import OilWaterPolymerModel

        fluid = _polymer_fluid()
        fluid.update({'bW': lambda p: np.ones_like(np.atleast_1d(p), dtype=float),
                      'bO': lambda p: np.ones_like(np.atleast_1d(p), dtype=float),
                      'muW': lambda p: np.ones_like(np.atleast_1d(p), dtype=float),
                      'muO': lambda p: np.ones_like(np.atleast_1d(p), dtype=float),
                      'pcOW': None, 'rhoWS': 1000.0, 'rhoOS': 800.0})
        m = GenericSurfactantPolymerModel(fluid=fluid, water=True, oil=True, gas=False, polymer=True)
        assert isinstance(m, OilWaterPolymerModel)

    def test_combined_polymer_surfactant_raises(self):
        from PRSTCore.ad_eor.models.GenericSurfactantPolymerModel import GenericSurfactantPolymerModel
        with pytest.raises(NotImplementedError):
            GenericSurfactantPolymerModel(polymer=True, surfactant=True)

    def test_surfactant_plus_gas_raises(self):
        from PRSTCore.ad_eor.models.GenericSurfactantPolymerModel import GenericSurfactantPolymerModel
        with pytest.raises(NotImplementedError):
            GenericSurfactantPolymerModel(surfactant=True, gas=True)
