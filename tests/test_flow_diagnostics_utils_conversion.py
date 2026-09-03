from pathlib import Path
import importlib

import numpy as np

from PRSTCore.visualization.diagnostics import (
    computeFandPhi,
    computePressureAndDiagnostics,
    computeRTD,
    computeTOFandTracer,
    computeWellPairs,
    estimateRTD,
)


def test_all_mrst_diagnostics_utils_have_python_counterparts():
    mrst = Path("mrst-2026a/visualization/diagnostics/utils")
    py = Path("PRSTCore/visualization/diagnostics/utils")
    missing = []
    for matlab_file in sorted(mrst.rglob("*.m")):
        rel = matlab_file.relative_to(mrst).with_suffix(".py")
        if not (py / rel).exists():
            missing.append(str(rel))
    assert missing == []


def test_all_python_counterpart_modules_import():
    root = Path("PRSTCore/visualization/diagnostics/utils")
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(Path("PRSTCore")).with_suffix("")
        parts = ["PRSTCore"] + list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        importlib.import_module(".".join(parts))


def test_estimate_and_compute_rtd_small_case():
    G = {
        "cells": {
            "num": 3,
            "centroids": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "volumes": np.ones(3),
        },
        "faces": {"neighbors": np.array([[0, 1], [1, 2]])},
    }
    rock = {"poro": np.ones(3), "perm": np.ones((3, 1))}
    model = {"G": G, "rock": rock, "operators": {"N": np.array([[0, 1], [1, 2]]), "T": np.ones(2), "T_all": np.ones(2), "pv": np.ones(3)}}
    W = [
        {"name": "I1", "cells": [0], "sign": 1, "val": 1.0, "status": True, "dZ": np.array([0.0]), "refDepth": 0.0},
        {"name": "P1", "cells": [2], "sign": -1, "val": 1.0, "status": True, "dZ": np.array([0.0]), "refDepth": 0.0},
    ]
    state, diagnostics = computePressureAndDiagnostics(model, wells=W, firstArrival=False)
    state["s"] = np.column_stack([np.full(3, 0.2), np.full(3, 0.8)])
    D = computeTOFandTracer(state, G, rock, wells=W, computeWellTOFs=True, firstArrival=False, model=model)
    WP = computeWellPairs(state, G, rock, W, D)

    estimated = estimateRTD(np.ones(3), D, WP, nbins=8)
    assert estimated.values.shape == (8, 1)
    assert estimated.pairIx.tolist() == [[0.0, 0.0]]
    F, Phi = computeFandPhi(estimated)
    assert F.shape == Phi.shape

    computed = computeRTD(state, G, np.ones(3), D, WP, W, nsteps=2, nbase=2)
    assert computed.values.shape[1] == 1
    assert computed.pairIx.tolist() == [[0.0, 0.0]]

