"""Tests for the port of MRST ``hm/utils`` (self-contained helpers)."""

import numpy as np
import pytest

from PRSTCore.hm.utils.argmaxQuadratic import argmaxQuadratic
from PRSTCore.hm.utils.getCapPressScalingPoints import getCapPressScalingPoints
from PRSTCore.hm.utils.getRelpermScalingPoints import (as_dict,
                                                       getRelpermScalingPoints)
from PRSTCore.hm.utils.initCapPressScaling import initCapPressScaling
from PRSTCore.hm.utils.isPointOnLine import isPointOnLine


# --------------------------------------------------------------- argmax --

def test_argmax_recovers_a_known_quadratic_maximum():
    """f(x) = -(x-2)^2 + 5 peaks at x = 2, f'(0) = 4, f(0) = 1, f(6) = -11."""
    x_opt, poly = argmaxQuadratic({'a': 0.0, 'v': 1.0, 'dv': 4.0},
                                  {'a': 6.0, 'v': -11.0})
    assert x_opt == pytest.approx(2.0)
    assert np.allclose(poly, [-1.0, 4.0, 1.0])


def test_argmax_is_shift_invariant():
    """The fit is formed in a coordinate shifted by p1.a, then shifted back."""
    x_opt, _ = argmaxQuadratic({'a': 10.0, 'v': 1.0, 'dv': 4.0},
                               {'a': 16.0, 'v': -11.0})
    assert x_opt == pytest.approx(12.0)


def test_argmax_rejects_an_extremum_behind_the_first_point():
    """f(x) = -(x+1)^2: the peak is at x = -1, left of p1.a = 0."""
    x_opt, _ = argmaxQuadratic({'a': 0.0, 'v': -1.0, 'dv': -2.0},
                               {'a': 1.0, 'v': -4.0})
    assert x_opt == -np.inf


def test_argmax_handles_a_degenerate_fit():
    """A straight line has no interior stationary point."""
    x_opt, _ = argmaxQuadratic({'a': 0.0, 'v': 0.0, 'dv': 1.0},
                               {'a': 2.0, 'v': 2.0})
    assert x_opt == -np.inf


def test_argmax_accepts_attribute_objects():
    class P:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    x_opt, _ = argmaxQuadratic(P(a=0.0, v=1.0, dv=4.0), P(a=6.0, v=-11.0))
    assert x_opt == pytest.approx(2.0)


# ---------------------------------------------------------- point/line --

def test_point_on_line_detects_membership_in_3d():
    P1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    P2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    Q = np.array([[0.5, 0.0, 0.0],   # on line 0
                  [0.0, 0.5, 0.0],   # on line 1
                  [0.5, 0.5, 0.0]])  # on neither
    on, idx = isPointOnLine(Q, P1, P2)
    assert list(on) == [True, True, False]
    assert list(idx[0]) == [0]
    assert list(idx[1]) == [1]
    assert idx[2].size == 0


def test_point_on_line_works_in_2d():
    P1 = np.array([[0.0, 0.0]])
    P2 = np.array([[2.0, 0.0]])
    on, _ = isPointOnLine(np.array([[1.0, 0.0], [1.0, 1.0]]), P1, P2)
    assert list(on) == [True, False]


def test_degenerate_segment_becomes_a_coincidence_test():
    """|P2 - P1| < tol collapses the segment; only P1 itself matches."""
    P1 = np.array([[1.0, 1.0, 1.0]])
    P2 = np.array([[1.0, 1.0, 1.0]])
    on, _ = isPointOnLine(np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]), P1, P2)
    assert list(on) == [True, False]


def test_dimension_mismatch_is_rejected():
    with pytest.raises(AssertionError):
        isPointOnLine(np.zeros((1, 3)), np.zeros((1, 2)), np.zeros((1, 2)))


# ------------------------------------------------------- scaling points --

class _Model:
    def __init__(self, nc, krPts=None, props=None, regions=None, parent=None):
        self.G = {'cells': {'num': nc, 'indexMap': np.arange(nc)}}
        if parent is not None:
            self.G['parent'] = parent
        self.fluid = {'krPts': krPts or {}}
        self.rock = {'regions': regions or {}}
        self.inputdata = {'PROPS': props} if props is not None else None


def test_relperm_points_come_from_the_tabulated_krpts():
    krPts = {'w': np.array([[0.1, 0.2, 0.9, 0.8]]),
             'g': np.array([[0.0, 0.05, 0.8, 0.7]])}
    got = as_dict(getRelpermScalingPoints(_Model(3, krPts=krPts)))
    assert np.allclose(got['SWL'], 0.1) and np.allclose(got['SWCR'], 0.2)
    assert np.allclose(got['SWU'], 0.9) and np.allclose(got['KRW'], 0.8)
    assert np.allclose(got['SGCR'], 0.05) and np.allclose(got['KRG'], 0.7)


def test_kro_is_taken_from_ow_and_not_duplicated_by_og():
    krPts = {'ow': np.array([[0.0, 0.15, 1.0, 0.85]]),
             'og': np.array([[0.0, 0.25, 1.0, 0.55]])}
    scaling = getRelpermScalingPoints(_Model(2, krPts=krPts))
    names = [n for n, _ in scaling]
    assert names.count('KRO') == 1
    got = as_dict(scaling)
    assert np.allclose(got['KRO'], 0.85)      # the ow value wins
    assert np.allclose(got['SOGCR'], 0.25)


def test_saturation_regions_select_the_row_per_cell():
    krPts = {'w': np.array([[0.1, 0.2, 0.9, 0.8],
                            [0.3, 0.4, 0.7, 0.6]])}
    model = _Model(3, krPts=krPts, regions={'saturation': np.array([1, 2, 1])})
    got = as_dict(getRelpermScalingPoints(model))
    assert np.allclose(got['SWL'], [0.1, 0.3, 0.1])


def test_deck_keywords_override_only_where_finite():
    krPts = {'w': np.array([[0.1, 0.2, 0.9, 0.8]])}
    props = {'SWL': np.array([0.15, np.nan, 0.25])}
    got = as_dict(getRelpermScalingPoints(_Model(3, krPts=krPts, props=props)))
    assert np.allclose(got['SWL'], [0.15, 0.1, 0.25])


def test_unknown_deck_keywords_are_ignored():
    props = {'NOTASCALINGKEYWORD': np.zeros(3)}
    scaling = getRelpermScalingPoints(_Model(3, krPts={}, props=props))
    assert scaling == []


def test_coarse_grid_keeps_only_the_tabulated_points():
    krPts = {'w': np.array([[0.1, 0.2, 0.9, 0.8]])}
    props = {'SWL': np.full(3, 0.5)}
    model = _Model(3, krPts=krPts, props=props, parent={'cells': {'num': 9}})
    got = as_dict(getRelpermScalingPoints(model))
    assert np.allclose(got['SWL'], 0.1)


def test_cap_press_points_read_only_their_four_keywords():
    props = {'SWLPC': np.full(3, 0.11), 'PCW': np.full(3, 2.0),
             'SWL': np.full(3, 0.9)}
    got = as_dict(getCapPressScalingPoints(_Model(3, props=props)))
    assert set(got) == {'SWLPC', 'PCW'}
    assert np.allclose(got['PCW'], 2.0)


def test_cap_press_points_are_empty_without_a_deck():
    assert getCapPressScalingPoints(_Model(3)) == []


# ------------------------------------------------------ initCapPressScaling --

def test_init_cap_press_scaling_fills_both_branches():
    deck = {'PROPS': {'SWLPC': np.full(4, 0.12), 'PCW': np.full(4, 3.0),
                      'ISWLPC': np.full(4, 0.20)}}
    out = initCapPressScaling(deck, 4)
    assert np.allclose(out['drainage']['w'][:, 0], 0.12)
    assert np.allclose(out['drainage']['w'][:, 1], 3.0)
    assert np.allclose(out['imbibition']['w'][:, 0], 0.20)
    # An absent keyword stays NaN, the "use the tabulated value" marker.
    assert np.all(np.isnan(out['imbibition']['w'][:, 1]))
    assert np.all(np.isnan(out['drainage']['g']))


def test_init_cap_press_scaling_broadcasts_a_scalar():
    out = initCapPressScaling({'PROPS': {'PCG': np.array([5.0])}}, 3)
    assert np.allclose(out['drainage']['g'][:, 1], 5.0)


# ------------------------------------------------------------- Corey WO --

from PRSTCore.ad_core.adi import SparseADI
from PRSTCore.hm.utils.assignCOREYGO import assignCOREYGO
from PRSTCore.hm.utils.assignCOREYWO import assignCOREYWO

# SWL SWU SWCR SOWCR krOLW krORW krWR krWU pcOW nOW nW np SpcO
_WO = np.array([0.15, 1.0, 0.20, 0.25, 0.9, np.nan, np.nan, 0.6,
                0.0, 2.0, 2.0, 0.0, 0.0])
# SGL SGU SGCR SOGCR krOLG krORG krGR krGU pcOG nOG nG np SpcG
_GO = np.array([0.05, 1.0, 0.10, 0.20, 0.85, np.nan, np.nan, 0.7,
                0.0, 2.0, 2.0, 0.0, 0.0])


def _matlab_krW(SW, c=_WO, SGL=0.0):
    """Direct transcription of CoreyKrW's two-segment branch."""
    SWn = (SW - c[2]) / (1 - c[2] - c[3] - SGL)
    out = np.zeros_like(SW)
    ix1 = (SW >= c[0]) & (SW < c[2])
    ix2 = (SW >= c[2]) & (SW <= c[1])
    out[ix1] = 0.0
    out[ix2] = c[7] * SWn[ix2] ** c[10]
    out[SW > c[1]] = c[7]
    return out


def test_corey_krw_matches_the_matlab_across_all_branches():
    f = assignCOREYWO({}, _WO[None, :], [0.0], {'sat': 1})
    sw = np.array([0.10, 0.15, 0.20, 0.50, 0.75, 0.99, 1.0, 1.05])
    assert np.allclose(f['krW'][0](sw), _matlab_krW(sw))


def test_corey_krw_clamps_only_strictly_above_swu():
    """krW(S > SWU) = krWU -- at S == SWU the Corey branch still applies,
    and with SOWCR > 0 the normalised saturation exceeds one there."""
    f = assignCOREYWO({}, _WO[None, :], [0.0], {'sat': 1})
    assert f['krW'][0](np.array([1.05]))[0] == pytest.approx(0.6)
    assert f['krW'][0](np.array([1.0]))[0] > 0.6


def test_corey_krow_is_max_below_connate_water():
    f = assignCOREYWO({}, _WO[None, :], [0.0], {'sat': 1})
    # SW = 1 - SO < SWL  =>  krOW = krOLW
    assert f['krOW'][0](np.array([0.95]))[0] == pytest.approx(0.9)


def test_corey_scaling_points_follow_getpoints():
    f = assignCOREYWO({}, _WO[None, :], [0.0], {'sat': 1})
    assert np.allclose(f['krPts']['w'][0], [0.15, 0.20, 1.0, 0.6])
    assert np.allclose(f['krPts']['ow'][0], [0.0, 0.25, 1.0, 0.9])


def test_corey_krw_derivative_matches_finite_differences():
    f = assignCOREYWO({}, _WO[None, :], [0.0], {'sat': 1})
    krW, h = f['krW'][0], 1e-7
    u = SparseADI.variable(np.array([0.5]), 1, 0)
    fd = (krW(np.array([0.5 + h]))[0] - krW(np.array([0.5]))[0]) / h
    assert krW(u).jac.toarray()[0, 0] == pytest.approx(fd, abs=1e-4)


def test_corey_pc_is_absent_when_the_exponent_is_zero():
    f = assignCOREYWO({}, _WO[None, :], [0.0], {'sat': 1})
    assert 'pcOW' not in f


def test_corey_pc_is_attached_when_the_exponent_is_nonzero():
    row = _WO.copy()
    row[8], row[11], row[12] = 1.0e5, 2.0, 0.8   # pcOW, np, SpcO
    f = assignCOREYWO({}, row[None, :], [0.0], {'sat': 1})
    assert 'pcOW' in f
    # Pc vanishes above SpcO and is positive below it.
    assert f['pcOW'][0](np.array([0.9]))[0] == pytest.approx(0.0)
    assert f['pcOW'][0](np.array([0.3]))[0] > 0.0


def test_corey_multiple_regions_get_their_own_curves():
    rows = np.vstack([_WO, _WO.copy()])
    rows[1, 7] = 0.3                              # a different krWU
    f = assignCOREYWO({}, rows, [0.0, 0.0], {'sat': 2})
    assert len(f['krW']) == 2
    assert f['krW'][0](np.array([1.05]))[0] == pytest.approx(0.6)
    assert f['krW'][1](np.array([1.05]))[0] == pytest.approx(0.3)


# ------------------------------------------------------------- Corey GO --

def test_corey_krg_mirrors_the_water_construction():
    f = assignCOREYGO({}, _GO[None, :], [0.15], {'sat': 1})
    krG = f['krG'][0]
    assert krG(np.array([0.02]))[0] == pytest.approx(0.0)   # below SGL
    assert krG(np.array([0.10]))[0] == pytest.approx(0.0)   # at SGCR
    assert krG(np.array([1.05]))[0] == pytest.approx(0.7)   # above SGU


def test_corey_go_scaling_points_subtract_connate_water():
    """pts_o(2) = SOGCR - SWL."""
    f = assignCOREYGO({}, _GO[None, :], [0.15], {'sat': 1})
    assert np.allclose(f['krPts']['g'][0], [0.05, 0.10, 1.0, 0.7])
    assert np.allclose(f['krPts']['og'][0], [0.0, 0.20 - 0.15, 1.0, 0.85])


def test_corey_pcog_rises_with_gas_saturation():
    """CoreyPcOG uses (SG - SpcG), the opposite sense to CoreyPcOW."""
    row = _GO.copy()
    row[8], row[11], row[12] = 1.0e5, 2.0, 0.10   # pcOG, np, SpcG
    f = assignCOREYGO({}, row[None, :], [0.15], {'sat': 1})
    assert f['pcOG'][0](np.array([0.5]))[0] == pytest.approx(0.0)   # >= SpcG
    below = f['pcOG'][0](np.array([0.07]))[0]
    assert below > 0.0


# ------------------------------------------------------- well index ADI --

from PRSTCore.hm.utils.evaluate.computeWellIndexADI import computeWellIndexADI
from PRSTCore.hm.utils.imposeCapPressScaling import imposeCapPressScaling
from PRSTCore.hm.utils.recomputeWellIndex import (_dedupe_wpimult,
                                                  _grid_logical_indices,
                                                  recomputeWellIndex)

_G3 = {'cells': {'num': 4}, 'griddim': 3}
_DIMS = np.tile([10.0, 20.0, 5.0], (4, 1))
_PERM = np.tile([1e-13, 2e-13, 3e-13], (4, 1))


def _peaceman(d1, d2, ell, k1, k2, rw, skin=0.0, wc=0.14):
    """The formula computeWellIndex.m implements, written out."""
    k21, k12 = k2 / k1, k1 / k2
    re = (2 * wc * np.sqrt(d1 ** 2 * np.sqrt(k21) + d2 ** 2 * np.sqrt(k12))
          / (k21 ** 0.25 + k12 ** 0.25))
    return 2 * np.pi * (ell * np.sqrt(k1 * k2)) / (np.log(re / rw) + skin)


def _wi(rock=None, **kw):
    kw.setdefault('Dir', 'z')
    kw.setdefault('cellDims', _DIMS)
    return computeWellIndexADI(_G3, rock or {'perm': _PERM}, np.full(4, 0.1),
                               np.arange(4), **kw)


def test_well_index_matches_the_peaceman_formula():
    assert np.allclose(_wi(), _peaceman(10.0, 20.0, 5.0, 1e-13, 2e-13, 0.1))


def test_well_index_direction_selects_the_cross_flow_axes():
    """Dir='x' uses (dy, dz, dx) and (ky, kz)."""
    assert np.allclose(_wi(Dir='x'),
                       _peaceman(20.0, 5.0, 10.0, 2e-13, 3e-13, 0.1))


def test_well_index_skin_enters_the_denominator():
    assert np.allclose(_wi(Skin=np.full(4, 2.0)),
                       _peaceman(10.0, 20.0, 5.0, 1e-13, 2e-13, 0.1, 2.0))


def test_supplied_kh_overrides_the_computed_one():
    got = _wi(Kh=np.full(4, 1e-12))
    k21, k12 = 2.0, 0.5
    re = (2 * 0.14 * np.sqrt(100 * np.sqrt(k21) + 400 * np.sqrt(k12))
          / (k21 ** 0.25 + k12 ** 0.25))
    assert np.allclose(got, 2 * np.pi * 1e-12 / np.log(re / 0.1))


def test_zero_permeability_gives_zero_not_nan():
    assert np.allclose(_wi(rock={'perm': np.tile([0.0, 2e-13, 3e-13], (4, 1))}), 0.0)


def test_well_index_derivative_matches_finite_differences():
    kx = SparseADI.variable(np.full(4, 1e-13), 4, 0)
    ad = _wi(rock={'perm': [kx, np.full(4, 2e-13), np.full(4, 3e-13)]})
    h = 1e-18
    bumped = _wi(rock={'perm': np.tile([1e-13 + h, 2e-13, 3e-13], (4, 1))})
    fd = (bumped[0] - _wi()[0]) / h
    assert ad.jac.toarray()[0, 0] == pytest.approx(fd, rel=1e-4)


def test_subset_selects_only_the_requested_perforations():
    assert _wi(Subset=np.array([True, False, True, False])).size == 2


def test_mismatched_celldims_is_rejected():
    with pytest.raises(ValueError, match='cellDims'):
        _wi(cellDims=np.zeros((7, 3)))


def test_negative_well_index_is_reported():
    """A large negative skin drives WI negative; the MATLAB errors."""
    with pytest.raises(ValueError, match='skin'):
        _wi(Skin=np.full(4, -100.0))


# ----------------------------------------------------- recomputeWellIndex --

class _WIModel:
    def __init__(self):
        self.G = {'cells': {'num': 4, 'DX': np.full(4, 10.0),
                            'DY': np.full(4, 20.0), 'DZ': np.full(4, 5.0)},
                  'griddim': 3}
        self.rock = {'perm': _PERM}
        self.inputdata = None


def _well(wi, defaulted_wi):
    return [{'name': 'P1', 'cells': np.arange(4), 'r': np.full(4, 0.1),
             'dir': np.array(['z'] * 4), 'cstatus': np.ones(4, bool),
             'WI': np.asarray(wi, dtype=float).copy(),
             'defaulted': {'WI': np.asarray(defaulted_wi, dtype=float).copy(),
                           'Kh': np.full(4, -1.0), 'Skin': np.zeros(4)}}]


def test_grid_logical_indices_are_one_based():
    G = {'cartDims': [3, 2, 2], 'cells': {'num': 12, 'indexMap': np.arange(12)}}
    i, j, k = _grid_logical_indices(G)
    assert list(i[:4]) == [1, 2, 3, 1]
    assert list(j[:4]) == [1, 1, 1, 2]
    assert k[0] == 1 and k[6] == 2


def test_wpimult_keeps_only_the_last_blanket_record():
    out = _dedupe_wpimult([['W1', 2.0, -1, -1, -1], ['W1', 3.0, -1, -1, -1],
                           ['W1', 5.0, 1, 1, 1]])
    assert ['W1', 5.0, 1, 1, 1] in out
    blanket = [r for r in out if r[2] == -1]
    assert len(blanket) == 1 and blanket[0][1] == 3.0


def test_recompute_well_index_refreshes_defaulted_perforations():
    schedule = {'control': [{'W': _well(np.zeros(4), np.zeros(4))}]}
    out = recomputeWellIndex(_WIModel(), schedule)
    assert np.allclose(out['control'][0]['W'][0]['WI'],
                       _peaceman(10.0, 20.0, 5.0, 1e-13, 2e-13, 0.1))


def test_recompute_skips_perforations_the_deck_supplied():
    """compWI requires defaulted.WI <= 0; a positive one is left alone."""
    schedule = {'control': [{'W': _well(np.full(4, 7.0), np.full(4, 7.0))}]}
    out = recomputeWellIndex(_WIModel(), schedule)
    assert np.allclose(out['control'][0]['W'][0]['WI'], 7.0)


# --------------------------------------------------- imposeCapPressScaling --

class _ScaleModel:
    def __init__(self, nc):
        self.G = {'cells': {'num': nc, 'indexMap': np.arange(nc)}}
        self.rock = {}
        self.inputdata = None


def test_impose_cap_press_creates_the_scaling_and_enables_endscale():
    model = imposeCapPressScaling(_ScaleModel(3), SWLPC=0.12, PCW=np.full(3, 4.0))
    assert np.allclose(model.rock['pcscale']['drainage']['w'][:, 0], 0.12)
    assert np.allclose(model.rock['pcscale']['drainage']['w'][:, 1], 4.0)
    assert model.inputdata['RUNSPEC']['ENDSCALE'][0] == 'NODIR'
    assert model.inputdata['PROPS']['SCALECRS'] == ['NO']


def test_impose_cap_press_updates_an_existing_pcscale():
    model = _ScaleModel(3)
    imposeCapPressScaling(model, SWLPC=0.10)
    imposeCapPressScaling(model, PCW=np.full(3, 9.0))
    assert np.allclose(model.rock['pcscale']['drainage']['w'][:, 0], 0.10)
    assert np.allclose(model.rock['pcscale']['drainage']['w'][:, 1], 9.0)


def test_impose_cap_press_rejects_a_wrong_length():
    with pytest.raises(AssertionError):
        imposeCapPressScaling(_ScaleModel(3), PCW=np.zeros(5))


def test_impose_cap_press_warns_on_an_unknown_keyword():
    with pytest.warns(RuntimeWarning, match='unrecognized'):
        imposeCapPressScaling(_ScaleModel(3), NOTAKEYWORD=1.0, PCW=1.0)


def test_impose_cap_press_without_arguments_is_a_no_op():
    model = _ScaleModel(3)
    assert imposeCapPressScaling(model) is model
    assert 'pcscale' not in model.rock
