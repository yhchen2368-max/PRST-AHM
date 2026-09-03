"""Parity tests for the wellbore cross-flow mixture.

``crossFlowMixture.m`` is transcribed here directly from the MATLAB and used
as the oracle, so the tests state the reference rule rather than restating
the port.  The gate is the one its caller ``WellComponentPhaseFlux.m`` uses:
``if any(perfIsInjector)`` -- notably *not* restricted to producers.
"""

import numpy as np
import pytest

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.ad_core.models.well_model import SimpleWell


def mrst_cross_flow_mixture(flux, compi):
    """Direct transcription of ``crossFlowMixture.m`` for a single well."""
    flux_in = -np.minimum(flux, 0.0)
    if np.all(flux_in == 0.0):
        return np.asarray(compi, dtype=float).copy()
    net_injection = max(float(np.sum(flux)), 0.0)
    comp = np.sum(flux_in, axis=0) + net_injection * np.asarray(compi, dtype=float)
    total = float(np.sum(comp))
    return comp / total if total > 0.0 else np.asarray(compi, dtype=float).copy()


def run_well(*, drawdown, mobility, compi, sign):
    """Drive one well through SimpleWell and return its per-perforation
    phase fluxes.

    ``drawdown[i] > 0`` makes perforation ``i`` inject (MRST's
    ``perfIsInjector``); ``mobility[i][k]`` is phase ``k``'s mobility in
    perforation ``i``'s cell.
    """
    mobility = np.asarray(mobility, dtype=float)
    nperf, nph = mobility.shape
    nc, nvar = nperf, nperf + 1

    # tdp = (p - bhp)*(-WI); with bhp = 0 and WI = 1, tdp = -p, so p = -drawdown.
    p = SparseADI.variable(-np.asarray(drawdown, dtype=float), nvar, 0)
    bhp = SparseADI.variable(np.zeros(1), nvar, nc)
    lam = [SparseADI.constant(mobility[:, k], nvar) for k in range(nph)]

    w = {'WI': np.ones(nperf), 'cstatus': np.ones(nperf, dtype=bool),
         'compi': np.asarray(compi, dtype=float), 'sign': sign}

    _, _, perf_phase_flux, _ = SimpleWell().compute_contributions(
        w=w, cells=list(range(nperf)), p=p, bhp=bhp, cdp=np.zeros(nperf),
        lam_phases=lam, rhoS_phases=[1.0] * nph,
        component_mass_fn=lambda c, qph: list(qph),
        nc=nc, nvar=nvar, n_component_phases=nph)
    return perf_phase_flux


def test_transcription_matches_the_matlab_on_a_crossflowing_injector():
    """Produced fluid [2,1] blends with 2 units of net injection carrying
    compi=[1,0]: comp = [2,1] + 2*[1,0] = [4,1] -> [0.8, 0.2]."""
    flux = np.array([[-2.0, -1.0], [5.0, 0.0]])
    assert np.allclose(mrst_cross_flow_mixture(flux, np.array([1.0, 0.0])),
                       [0.8, 0.2])


def test_no_inflow_leaves_compi_untouched():
    """``all(flux_in == 0)`` returns compi unchanged, whatever the sign."""
    flux = np.array([[3.0, 0.0], [5.0, 0.0]])
    compi = np.array([1.0, 0.0])
    assert np.allclose(mrst_cross_flow_mixture(flux, compi), compi)


def test_pure_production_blends_to_the_produced_composition():
    flux = np.array([[-2.0, -6.0]])
    assert np.allclose(mrst_cross_flow_mixture(flux, np.array([1.0, 0.0])),
                       [0.25, 0.75])


@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_injecting_perforation_of_a_crossflowing_well_uses_the_blend(sign):
    """The regression: the port gated the blend on ``sign < 0``, so an
    injector (sign > 0) with cross-flow reinjected raw compi.

    Perforation 0 produces into the wellbore, perforation 1 injects. The
    injecting perforation must deliver the *blended* composition, and the
    blend must not depend on the well's overall sign.
    """
    flux = run_well(drawdown=[-1.0, 1.0],
                    mobility=[[2.0, 1.0], [3.0, 2.0]],
                    compi=[1.0, 0.0], sign=sign)

    # Perforation 0 is producing, so its split is mobility-weighted.
    assert np.allclose(flux[0], [-2.0, -1.0])

    # Perforation 1 injects the blend of the back-produced fluid and compi.
    expected_mix = mrst_cross_flow_mixture(flux, np.array([1.0, 0.0]))
    total = float(np.sum(flux[1]))
    assert np.allclose(flux[1], total * expected_mix)
    # It is genuinely a blend, not raw compi.
    assert not np.allclose(expected_mix, [1.0, 0.0])


def test_sign_does_not_change_the_result():
    """An injector and a producer with identical perforation states must
    produce identical fluxes -- MRST's gate never consults the well sign."""
    kwargs = dict(drawdown=[-1.0, 1.0], mobility=[[2.0, 1.0], [3.0, 2.0]],
                  compi=[1.0, 0.0])
    assert np.allclose(run_well(sign=1.0, **kwargs),
                       run_well(sign=-1.0, **kwargs))


def test_ordinary_injector_without_crossflow_injects_compi():
    """With no inflow the blend is a no-op, so the unrestricted gate is
    safe for a plain injector."""
    flux = run_well(drawdown=[1.0, 1.0], mobility=[[2.0, 1.0], [3.0, 2.0]],
                    compi=[1.0, 0.0], sign=1.0)
    for perf in flux:
        total = float(np.sum(perf))
        assert np.allclose(perf, total * np.array([1.0, 0.0]))


def test_unnormalised_compi_is_normalised_before_mixing():
    """WellComponentPhaseFlux divides compi by max(sum(compi), 1e-10), so a
    deck compi of [2, 0] must behave exactly like [1, 0]."""
    kwargs = dict(drawdown=[-1.0, 1.0], mobility=[[2.0, 1.0], [3.0, 2.0]],
                  sign=1.0)
    assert np.allclose(run_well(compi=[2.0, 0.0], **kwargs),
                       run_well(compi=[1.0, 0.0], **kwargs))
