import numpy as np

from PRSTCore.network_models.network import Network
from PRSTCore.network_models.gpsnet import GPSNet
from PRSTCore.visualization.diagnostics import (
    computePressureAndDiagnostics,
    computeTOFandTracer,
    computeWellPairs,
)


def _one_dimensional_problem():
    G = {
        "cells": {
            "num": 3,
            "centroids": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            "volumes": np.ones(3),
        },
        "faces": {"neighbors": np.array([[0, 1], [1, 2]])},
    }
    rock = {"poro": np.ones(3)}
    rock["perm"] = np.ones((3, 1))
    model = {
        "G": G,
        "rock": rock,
        "operators": {
            "N": np.array([[0, 1], [1, 2]]),
            "T_all": np.ones(2),
            "T": np.ones(2),
            "pv": np.ones(3),
        },
    }
    W = [
        {"name": "I1", "cells": [0], "sign": 1, "val": 1.0, "status": True, "dZ": np.array([0.0]), "refDepth": 0.0},
        {"name": "P1", "cells": [2], "sign": -1, "val": 1.0, "status": True, "dZ": np.array([0.0]), "refDepth": 0.0},
    ]
    return G, rock, model, W


def test_compute_tof_and_well_pairs_match_mrst_field_layout():
    G, rock, model, W = _one_dimensional_problem()
    state, diagnostics = computePressureAndDiagnostics(model, wells=W, firstArrival=False)

    D = diagnostics.D
    WP = diagnostics.WP
    assert D.inj.tolist() == [0]
    assert D.prod.tolist() == [1]
    assert D.tof.shape == (3, 2)
    assert D.itracer.shape == (3, 1)
    assert D.ptracer.shape == (3, 1)
    assert np.allclose(D.itracer, 1.0)
    assert np.allclose(D.ptracer, 1.0)
    assert WP.pairIx.tolist() == [[0, 0]]
    assert WP.pairs == ["I1, P1"]
    assert np.allclose(WP.vols, [3.0])
    assert np.allclose(diagnostics.wellCommunication, [[1.0]])

    D2 = computeTOFandTracer(state, G, rock, wells=W, model=model)
    WP2 = computeWellPairs(state, G, rock, W, D2)
    assert np.allclose(WP2.vols, WP.vols)


def test_flow_diagnostics_network_uses_communication_edges():
    G, _, model, W = _one_dimensional_problem()
    problem = {
        "SimulatorSetup": {
            "model": model,
            "schedule": {"step": {"val": [1.0], "control": [1]}, "control": [{"W": W}]},
        }
    }

    network = Network(W, G, type="fd_preprocessor", problem=problem, flow_filter=0.0)

    assert network.num_nodes == 2
    assert network.num_edges == 1
    assert list(network.network.edges()) == [(0, 1)]
    T, pv = network.get_edge_data()
    assert T.shape == (1,)
    assert pv.shape == (1,)
    assert np.isfinite(T[0])
    assert np.allclose(pv, [3.0])


def test_gpsnet_uses_fd_edge_data_for_initial_parameters():
    G, _, model, W = _one_dimensional_problem()
    problem = {
        "SimulatorSetup": {
            "model": model,
            "schedule": {"step": {"val": [1.0], "control": [1]}, "control": [{"W": W}]},
        }
    }
    network = Network(W, G, type="fd_preprocessor", problem=problem, flow_filter=0.0)
    gps = GPSNet(model, network, W, nc=4)
    edge_T, edge_pv = network.get_edge_data()
    setup = {
        "model": {
            "porevolume": gps.model["operators"]["pv"],
            "transmissibility": gps.model["operators"]["T"],
        }
    }

    class IdentityParam:
        def __init__(self, name, n_param):
            self.name = name
            self.n_param = n_param

        def scale(self, value):
            return np.asarray(value, dtype=float)

    pv_param = IdentityParam("porevolume", gps.model["operators"]["pv"].size)
    t_param = IdentityParam("transmissibility", gps.model["operators"]["T"].size)
    pvec = gps.get_scaled_parameter_vector(setup, [pv_param, t_param])

    assert np.allclose(pvec[:4], edge_pv[0] / 4)
    assert np.any(np.isclose(pvec[4:], edge_T[0]))
