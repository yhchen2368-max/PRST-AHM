import numpy as np
import pytest
from PRSTCore.ad_core.plotting import plot_well_sols
from PRSTCore.ad_core.upscale import upscale_model_tpfa, upscale_schedule, upscale_state
from PRSTCore.ad_core.utils import simple_schedule
from PRSTCore.coarsegrid import compress_partition, partition_ui, process_partition
from PRSTCore.optimization import evaluate_match, unit_box_bfgs, unit_box_lm
from PRSTCore.optimization.objectives import match_observed_ow
from PRSTCore.optimization.utils.parameters import add_parameter, get_scaled_parameter_vector


def test_partition_compress():
    p = np.array([3, 3, 5, 5, 1, 1])
    pc = compress_partition(p)
    assert pc.min() == 1
    assert pc.max() == 3
    assert np.array_equal(np.unique(pc), [1, 2, 3])


def test_partition_ui():
    G = {
        "cells": {
            "num": 12,
            "centroids": np.column_stack([
                np.tile(np.linspace(0, 300, 4), 3),
                np.repeat(np.linspace(0, 200, 3), 4),
                np.zeros(12),
            ]),
        }
    }
    q = partition_ui(G, [4, 3, 1])
    assert q.min() == 1
    assert q.max() == 12


def test_upscale_model():
    # A minimal but topologically complete 1D chain of 6 unit cells at
    # x=0..5 (upscale_model_tpfa needs real face connectivity/geometry --
    # process_partition/generate_coarse_grid/coarsen_geometry all read
    # G.faces.neighbors/areas/normals/centroids, not just G.cells).
    # Faces (1-based cell ids, 0 = boundary), ordered left to right:
    #   f0: (0,1) f1: (1,2) f2: (2,3) f3: (3,4) f4: (4,5) f5: (5,6) f6: (6,0)
    nbrs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]])
    G = {
        "cells": {
            "num": 6,
            "centroids": np.column_stack([np.arange(6, dtype=float), np.zeros(6), np.zeros(6)]),
            "volumes": np.ones(6),
        },
        "faces": {
            "num": 7,
            "neighbors": nbrs,
            "areas": np.ones(7),
            "normals": np.tile([1.0, 0.0, 0.0], (7, 1)),
            "centroids": np.column_stack([np.arange(-0.5, 6.5, 1.0), np.zeros(7), np.zeros(7)]),
        },
        "griddim": 3,
    }
    rock = {"poro": np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
            "perm": np.array([100, 200, 300, 100, 200, 300]).reshape(-1, 1)}
    model = {"G": G, "rock": rock, "operators": {"T": np.ones(18), "pv": np.ones(6)}}
    partition = np.array([1, 1, 2, 2, 3, 3])
    cm = upscale_model_tpfa(model, partition)
    assert cm["G"]["cells"]["num"] == 3
    assert cm["rock"]["poro"].size == 3
    # Coarse blocks 1 and 2 (cells {1,2} and {3,4}) are adjacent -> exactly
    # one internal coarse face should connect them.
    internal = np.all(cm["G"]["faces"]["neighbors"] != 0, axis=1)
    assert internal.sum() == 2  # block1-block2 and block2-block3


def test_upscale_state():
    cmodel = {"G": {"cells": {"num": 2}, "partition": np.array([1, 1, 2, 2])}}
    fmodel = {"operators": {"pv": np.array([1, 1, 2, 2])}}
    state = {"pressure": np.array([100, 110, 200, 210]),
             "s": np.column_stack([np.array([0.2, 0.3, 0.4, 0.5]),
                                    np.array([0.8, 0.7, 0.6, 0.5])])}
    cs = upscale_state(cmodel, fmodel, state)
    assert cs["pressure"].size == 2
    assert cs["s"].shape == (2, 2)


def test_get_scaled_parameter_vector():
    # A parameter is read through its ``location`` -- ``setupByName`` puts
    # porevolume at ``model.operators.pv``, not at a field of its own
    # name -- so the fixture has to be shaped like a model.
    model = {"operators": {"pv": np.array([0.1, 0.2, 0.3])},
             "conntrans": np.array([1.0, 2.0, 3.0])}
    setup = {"model": model, "schedule": {"step": {"val": [1]}, "control": []}}
    params = add_parameter([], setup, name="porevolume", scaling="linear",
                           relative_limits=[0.5, 2.0])
    pvec = get_scaled_parameter_vector(setup, params)
    assert pvec.size == 3
    assert np.all((pvec >= 0) & (pvec <= 1))


def test_evaluate_match_workflow():
    schedule = simple_schedule([1], [{"W": [{"type": "rate", "val": 0.5, "sign": -1, "status": True}]}])
    model = {"porevolume": np.array([1.0, 2.0]), "operators": {"T": np.ones(6), "pv": np.array([1.0, 2.0])}}
    setup = {"state0": {}, "model": model, "schedule": schedule}
    params = add_parameter([], setup, name="porevolume",
                           scaling="linear", relative_limits=[0.5, 2.0])
    states_ref = [{"wellSol": [{"status": True, "qWs": 0.1, "qOs": 0.4, "bhp": 50.0}]}]

    def obj(model, states, schedule, ref, cp, ts, s):
        return match_observed_ow(model, states, schedule, ref, compute_partials=cp)

    m, g, ws, ss = evaluate_match(np.array([0.5, 0.5]), obj, setup, params, states_ref)
    assert m is not None
    assert ws is not None


def test_unit_box_lm():
    pinit = np.array([0.5, 0.5])

    def residual_func(u):
        r = np.array([u[0] - 0.3, u[1] - 0.7])
        return r, np.eye(2), {}

    popt = unit_box_lm(pinit, residual_func, max_iter=10)
    assert popt.shape == (2,)
    assert np.all((popt >= 0) & (popt <= 1))


def test_plot_well_sols():
    ws1 = [[{"qWs": 0.5, "qOs": 0.3, "bhp": 100.0}]]
    ws2 = [[{"qWs": 0.4, "qOs": 0.4, "bhp": 110.0}]]
    fig = plot_well_sols([ws1, ws2],
                         [np.array([1]), np.array([1])],
                         dataset_names=["a", "b"],
                         field="qWs",
                         selected_wells=[0])
    assert fig is not None


# ------------------------------------------- MRST-0 upscaling behaviour --

def _chain_model(pv_fine=None):
    """The 1D chain above, with pore volumes that can be set apart from
    porosity -- which is what NTG or a PORV override produces."""
    nbrs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]])
    G = {
        "cells": {"num": 6,
                  "centroids": np.column_stack([np.arange(6, dtype=float),
                                                np.zeros(6), np.zeros(6)]),
                  "volumes": np.ones(6)},
        "faces": {"num": 7, "neighbors": nbrs, "areas": np.ones(7),
                  "normals": np.tile([1.0, 0.0, 0.0], (7, 1)),
                  "centroids": np.column_stack([np.arange(-0.5, 6.5, 1.0),
                                                np.zeros(7), np.zeros(7)])},
        "griddim": 3,
    }
    rock = {"poro": np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
            "perm": np.array([100, 200, 300, 100, 200, 300]).reshape(-1, 1)}
    if pv_fine is None:
        pv_fine = np.ones(6)
    return {"G": G, "rock": rock,
            "operators": {"T": np.ones(18), "pv": np.asarray(pv_fine)}}


def test_coarse_pore_volume_is_summed_from_the_fine_model():
    """MRST-0 sums the fine pore volumes by partition rather than
    recomputing porosity times coarse bulk volume; a coarse block that
    holds a different amount of fluid than the cells it stands for
    cannot reproduce their material balance."""
    pv = np.array([0.05, 0.25, 0.30, 0.05, 0.25, 0.30])
    cm = upscale_model_tpfa(_chain_model(pv), np.array([1, 1, 2, 2, 3, 3]))
    assert np.allclose(cm["operators"]["pv"], [0.30, 0.35, 0.55])


def test_the_geometric_product_would_have_given_a_different_split():
    """Stated explicitly so the difference cannot be dismissed as
    rounding: block 2 would hold 0.40 instead of 0.35."""
    pv = np.array([0.05, 0.25, 0.30, 0.05, 0.25, 0.30])
    cm = upscale_model_tpfa(_chain_model(pv), np.array([1, 1, 2, 2, 3, 3]))
    geometric = np.asarray(cm["G"]["cells"]["volumes"]) * \
        np.asarray(cm["rock"]["poro"])
    assert not np.allclose(cm["operators"]["pv"], geometric)


def test_total_pore_volume_is_conserved():
    pv = np.array([0.05, 0.25, 0.30, 0.05, 0.25, 0.30])
    cm = upscale_model_tpfa(_chain_model(pv), np.array([1, 1, 2, 2, 3, 3]))
    assert cm["operators"]["pv"].sum() == pytest.approx(pv.sum())


def test_an_explicit_coarse_pore_volume_is_used_as_given():
    cm = upscale_model_tpfa(_chain_model(), np.array([1, 1, 2, 2, 3, 3]),
                            pv_coarse=[1.0, 2.0, 3.0])
    assert list(cm["operators"]["pv"]) == [1.0, 2.0, 3.0]


def test_zero_transmissibility_connections_are_dropped():
    """A zero-weight connection contributes nothing but still costs a
    Jacobian entry."""
    model = _chain_model()
    cm = upscale_model_tpfa(model, np.array([1, 1, 2, 2, 3, 3]))
    assert np.all(cm["operators"]["T"] != 0)


# ------------------------------------------------ upscaleState (MRST-0) --

def _fine_state(nph=3):
    s = np.array([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2],
                  [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])[:, :nph]
    return {"pressure": np.array([1.0, 2.0, 3.0, 4.0]), "s": s}


def _coarse_pair():
    coarse = {"G": {"cells": {"num": 2}, "partition": np.array([1, 1, 2, 2])}}
    fine = {"operators": {"pv": np.array([1.0, 3.0, 2.0, 2.0])}}
    return coarse, fine


def test_upscaled_saturations_sum_to_one():
    """The last phase is set by closure. Pore-volume averages need not
    close on their own, and a state whose saturations do not sum to one
    is not a valid state."""
    coarse, fine = _coarse_pair()
    out = upscale_state(coarse, fine, _fine_state())
    assert np.allclose(out["s"].sum(axis=1), 1.0)


def test_upscaled_saturations_are_pore_volume_weighted():
    coarse, fine = _coarse_pair()
    out = upscale_state(coarse, fine, _fine_state())
    # Block 1 is cells 0 and 1 with pore volumes 1 and 3.
    assert out["s"][0, 0] == pytest.approx((0.2 * 1 + 0.4 * 3) / 4)


def test_a_single_phase_state_is_left_alone():
    """Closure has nothing to do with one phase, so it must not fire."""
    coarse, fine = _coarse_pair()
    out = upscale_state(coarse, fine, _fine_state(nph=1))
    assert out["s"].shape[1] == 1
    assert out["s"][0, 0] == pytest.approx((0.2 * 1 + 0.4 * 3) / 4)


def test_surfactant_concentration_is_upscaled():
    """MRST-0 adds this; concentration travels with the fluid, so it is
    pore-volume weighted like the saturations."""
    coarse, fine = _coarse_pair()
    fine["surfactant"] = True
    state = _fine_state()
    state["cs"] = np.array([1.0, 3.0, 2.0, 6.0])
    out = upscale_state(coarse, fine, state)
    assert out["cs"][0] == pytest.approx((1.0 * 1 + 3.0 * 3) / 4)


def test_polymer_concentration_is_upscaled():
    coarse, fine = _coarse_pair()
    fine["polymer"] = True
    state = _fine_state()
    state["cp"] = np.array([1.0, 3.0, 2.0, 6.0])
    out = upscale_state(coarse, fine, state)
    assert out["cp"][1] == pytest.approx((2.0 * 2 + 6.0 * 2) / 4)


def test_concentration_is_left_out_when_the_model_has_no_such_phase():
    coarse, fine = _coarse_pair()
    state = _fine_state()
    state["cs"] = np.array([1.0, 3.0, 2.0, 6.0])
    out = upscale_state(coarse, fine, state)
    assert out["cs"].size == 4          # untouched, still the fine array


# --------------------------------- schedule step multipliers (MRST-0) --

def test_step_multipliers_accumulate():
    """MRST's getCurrentMultipliers takes indices 1:step, so a MULTPV
    applied at step 2 is still applied at step 5."""
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
        _current_multiplier
    m = [1.0, 0.5, 2.0]
    assert _current_multiplier(m, 0, 1)[0] == pytest.approx(1.0)
    assert _current_multiplier(m, 1, 1)[0] == pytest.approx(0.5)
    assert _current_multiplier(m, 2, 1)[0] == pytest.approx(1.0)


def test_a_step_multiplier_may_be_per_cell():
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
        _current_multiplier
    m = [np.array([1.0, 2.0]), np.array([3.0, 1.0])]
    assert list(_current_multiplier(m, 1, 2)) == [3.0, 2.0]


def test_no_multipliers_means_nothing_to_apply():
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
        _current_multiplier
    assert _current_multiplier(None, 0, 2) is None


def test_each_step_multiplies_the_original_not_the_previous_result():
    """Compounding on the previous step's operators would apply the same
    multiplier again every step."""
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import (
        _apply_step_multiplier, _base_operator)

    class _M:
        operators = {'pv': np.array([10.0, 20.0])}

    model = _M()
    base = _base_operator(model, 'pv')
    _apply_step_multiplier(model, 'pv', base, [1.0, 0.5], 1)
    assert list(model.operators['pv']) == [5.0, 10.0]
    _apply_step_multiplier(model, 'pv', base, [1.0, 0.5], 0)
    assert list(model.operators['pv']) == [10.0, 20.0]


def test_a_schedule_without_multipliers_is_untouched():
    """The ordinary path must not change: base operators are only read
    when the schedule actually carries multpv/multipliers."""
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
        _base_operator

    class _M:
        operators = {'pv': np.array([1.0])}

    assert _base_operator(_M(), 'nosuch') is None
