"""Tests for the well-rate and well-placement helpers in the ``hm`` port."""

import numpy as np
import pytest
import scipy.sparse as sp

from PRSTCore.hm.utils.calculatePhaseRateBlackOil import calculatePhaseRateBlackOil
from PRSTCore.hm.utils.perturbWellPosition import (_checkStatus,
                                                   perturbWellPosition)


class _Res:
    @staticmethod
    def getPhaseIndex(*names):
        return tuple({'W': 0, 'O': 1, 'G': 2}[n] for n in names)


class _Model:
    ReservoirModel = _Res()
    verbose = 0


def _map(perf2well, is_injector, compi):
    return {'perf2well': np.asarray(perf2well, dtype=int),
            'isInjector': np.asarray(is_injector, dtype=bool),
            'W': [{'compi': np.asarray(c, dtype=float)} for c in compi]}


# ------------------------------------------------- calculatePhaseRateBlackOil --

def test_producing_perforations_keep_their_own_mobility():
    """With no injection the mobilities pass through untouched."""
    Tdp = np.array([-1.0, -2.0])
    mobw = [np.array([0.4, 0.5]), np.array([0.3, 0.2]), np.array([0.1, 0.1])]
    bw = np.ones((2, 3))
    q_ph = calculatePhaseRateBlackOil(
        Tdp, mobw, bw, np.zeros((1, 3)), np.zeros(2), np.zeros(2),
        _map([0, 0], [False], [[1.0, 0.0, 0.0]]), False, _Model())
    for i in range(3):
        assert np.allclose(q_ph[i], mobw[i] * Tdp)


def test_injecting_perforations_are_split_along_compi():
    """An injector delivers its declared mixture, not the mobility ratio."""
    Tdp = np.array([2.0])
    mobw = [np.array([0.4]), np.array([0.3]), np.array([0.3])]
    bw = np.ones((1, 3))
    q_ph = calculatePhaseRateBlackOil(
        Tdp, mobw, bw, np.zeros((1, 3)), np.zeros(1), np.zeros(1),
        _map([0], [True], [[1.0, 0.0, 0.0]]), False, _Model())
    total = sum(float(q[0]) for q in q_ph)
    # Total rate is conserved and goes entirely into water.
    assert total == pytest.approx(1.0 * 2.0)
    assert float(q_ph[0][0]) == pytest.approx(2.0)
    assert float(q_ph[1][0]) == pytest.approx(0.0)
    assert float(q_ph[2][0]) == pytest.approx(0.0)


def test_injection_split_honours_a_mixed_composition():
    Tdp = np.array([2.0])
    mobw = [np.array([0.4]), np.array([0.3]), np.array([0.3])]
    q_ph = calculatePhaseRateBlackOil(
        Tdp, mobw, np.ones((1, 3)), np.zeros((1, 3)), np.zeros(1), np.zeros(1),
        _map([0], [True], [[0.25, 0.75, 0.0]]), False, _Model())
    assert float(q_ph[0][0]) == pytest.approx(0.5)
    assert float(q_ph[1][0]) == pytest.approx(1.5)


def test_shrinkage_factors_reweight_the_split():
    """F = compi/b, so a smaller b gives that phase a larger share."""
    Tdp = np.array([2.0])
    mobw = [np.array([0.5]), np.array([0.5]), np.array([0.0])]
    bw = np.array([[1.0, 2.0, 1.0]])
    q_ph = calculatePhaseRateBlackOil(
        Tdp, mobw, bw, np.zeros((1, 3)), np.zeros(1), np.zeros(1),
        _map([0], [True], [[0.5, 0.5, 0.0]]), False, _Model())
    # Fw = 0.5/1, Fo = 0.5/2 -> water takes two thirds.
    assert float(q_ph[0][0]) == pytest.approx(2.0 * 2.0 / 3.0)
    assert float(q_ph[1][0]) == pytest.approx(2.0 * 1.0 / 3.0)


def test_total_injected_rate_is_conserved_across_compositions():
    Tdp = np.array([3.0])
    mobw = [np.array([0.2]), np.array([0.5]), np.array([0.1])]
    for compi in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]):
        q_ph = calculatePhaseRateBlackOil(
            Tdp, [m.copy() for m in mobw], np.ones((1, 3)), np.zeros((1, 3)),
            np.zeros(1), np.zeros(1), _map([0], [True], [compi]), False, _Model())
        assert sum(float(q[0]) for q in q_ph) == pytest.approx(0.8 * 3.0)


def test_crossflow_recomputes_the_composition_from_the_inflow():
    """A producer with one injecting perforation reinjects what came in."""
    Tdp = np.array([-1.0, 1.0])
    mobw = [np.array([1.0, 0.5]), np.array([1.0, 0.5]), np.array([0.0, 0.0])]
    plain = calculatePhaseRateBlackOil(
        Tdp, [m.copy() for m in mobw], np.ones((2, 3)), np.zeros((1, 3)),
        np.zeros(2), np.zeros(2),
        _map([0, 0], [False], [[1.0, 0.0, 0.0]]), False, _Model())
    crossed = calculatePhaseRateBlackOil(
        Tdp, [m.copy() for m in mobw], np.ones((2, 3)), np.zeros((1, 3)),
        np.zeros(2), np.zeros(2),
        _map([0, 0], [False], [[1.0, 0.0, 0.0]]), True, _Model())
    # Without cross-flow the injecting perforation follows compi (all
    # water); with it, the back-produced 50/50 mixture is reinjected.
    assert float(plain[1][1]) == pytest.approx(0.0)
    assert float(crossed[1][1]) > 0.0


# ------------------------------------------------------ perturbWellPosition --

def test_check_status_pads_a_shut_well_with_zero():
    W = [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}]
    out = _checkStatus(np.array([1.0, 2.0]), W, 1)
    assert np.allclose(out, [1.0, 0.0, 2.0])


def test_check_status_passes_a_full_vector_through():
    W = [{'name': 'A'}, {'name': 'B'}]
    assert np.allclose(_checkStatus(np.array([1.0, 2.0]), W, 0), [1.0, 2.0])


def test_check_status_rejects_an_unexplained_length():
    W = [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}]
    with pytest.raises(AssertionError):
        _checkStatus(np.array([1.0]), W, 0)


def test_perturb_well_position_differences_a_linear_residual():
    """A residual linear in the control point must give an exact gradient."""
    nc = 6
    cells = np.array([1, 2])

    class _Traj:
        def __init__(self):
            self.controlPoints = np.array([[0.0, 0.0, 0.0]])
            self.parameters = {'pointIx': np.array([0]),
                               'perturbation': np.array([[1e-3, 0.0, 0.0]])}
            self.w = {'name': 'P1'}

        def getTrajectory(self):
            return float(self.controlPoints[0, 0])

    posControl = _Traj()

    def getResiduals(state, W):
        # r = 3 * x on the perforated cells, 5 * x on the well equation.
        x = state['_x']
        cell_eq = np.zeros(nc)
        cell_eq[cells] = 3.0 * x
        return {'equations': [cell_eq, np.array([5.0 * x])],
                'types': ['cell', 'well']}

    import PRSTCore.hm.utils.perturbWellPosition as mod

    def fake_update(model, w, ws, traj):
        w = dict(w)
        w['cells'] = cells
        w['status'] = True
        return w, dict(ws)

    state = {'pressure': np.zeros(nc), 'wellSol': [{}], '_x': 0.0}
    W = [{'name': 'P1', 'cells': cells, 'status': True}]

    # Route the trajectory update through a stub and let the residual read
    # the control point the stub was handed.
    original = mod._evaluate

    def patched(model, st, Wl, wno, w0, ws0, pc, points0, point, pert, gr, ncc):
        points = np.array(points0, dtype=float, copy=True)
        points[point, :] = points0[point, :] + pert
        pc.controlPoints = points
        st['_x'] = float(points[0, 0])
        w_p, ws_p = fake_update(model, w0, ws0, pc.getTrajectory())
        mask = np.zeros(ncc, dtype=bool)
        mask[w_p['cells']] = True
        Wl[wno].update(w_p)
        return mask, gr(st, Wl)

    mod._evaluate = patched
    try:
        dFdU = perturbWellPosition(None, state, W, getResiduals, posControl)
    finally:
        mod._evaluate = original

    # d(cell eq)/dx = 3 on the perforated cells, d(well eq)/dx = 5.
    cell_block = np.asarray(sp.csc_matrix(dFdU[0]).todense()).ravel()
    assert cell_block[cells] == pytest.approx([3.0, 3.0])
    assert np.asarray(dFdU[1]).ravel()[0] == pytest.approx(5.0)


def test_perturb_well_position_restores_the_control_points():
    """The original geometry must survive the finite differencing."""
    nc = 4

    class _Traj:
        def __init__(self):
            self.controlPoints = np.array([[1.0, 2.0, 3.0]])
            self.parameters = {'pointIx': np.array([0]),
                               'perturbation': np.array([[1e-3, 0.0, 0.0]])}
            self.w = {'name': 'P1'}

        def getTrajectory(self):
            return self.controlPoints

    posControl = _Traj()
    state = {'pressure': np.zeros(nc), 'wellSol': [{}]}
    W = [{'name': 'P1', 'cells': np.array([0]), 'status': True}]

    import PRSTCore.hm.utils.perturbWellPosition as mod
    original = mod._evaluate

    def patched(model, st, Wl, wno, w0, ws0, pc, points0, point, pert, gr, ncc):
        points = np.array(points0, dtype=float, copy=True)
        points[point, :] = points0[point, :] + pert
        pc.controlPoints = points
        return np.zeros(ncc, dtype=bool), gr(st, Wl)

    mod._evaluate = patched
    try:
        perturbWellPosition(None, state, W, lambda s, w: {
            'equations': [np.zeros(nc)], 'types': ['cell']}, posControl)
    finally:
        mod._evaluate = original

    assert np.allclose(posControl.controlPoints, [[1.0, 2.0, 3.0]])
