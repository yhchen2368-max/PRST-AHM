"""Self-consistency checks for PRSTCore.solvers.incomp (compute_trans / incomp_tpfa),
independent of an MRST reference (see the note in test_gridprocessing_selfcheck.py
about the local MATLAB install being unavailable for a true numeric MRST parity run).

Checks exercised:
  - compute_trans matches the textbook two-point flux formula T = k*A/d on a
    uniform Cartesian grid (an exact, hand-computable reference).
  - incomp_tpfa reproduces the analytic linear pressure profile / constant
    Darcy flux for 1D flow between two Dirichlet-pressure boundaries.
  - Local mass conservation: for every cell, the signed sum of face fluxes
    (outward positive) equals that cell's net source/well rate -- this is
    the fundamental correctness property incompTPFA is built to guarantee,
    and a bug in the matrix assembly or flux reconstruction breaks it
    immediately.
  - A rate-controlled injector's realized well flux exactly equals its
    target rate (the augmented well-equation block must be solved exactly,
    not approximately).
"""

from __future__ import annotations

import numpy as np
import pytest

from PRSTCore.gridprocessing import cart_grid, compute_geometry
from PRSTCore.solvers.incomp import compute_trans, incomp_tpfa


def _cell_flux_balance(G: dict, flux: np.ndarray) -> np.ndarray:
    """Signed sum of face fluxes per cell (outward positive); should equal
    net source/well rate for that cell under steady incompressible flow."""
    nc = G["cells"]["num"]
    face_pos = G["cells"]["facePos"]
    cell_faces = G["cells"]["faces"]
    cellNo = np.repeat(np.arange(nc), np.diff(face_pos))
    cf = cell_faces[:, 0]
    neighbors = G["faces"]["neighbors"]
    sign = np.where(neighbors[cf, 0] == cellNo, 1.0, -1.0)
    return np.bincount(cellNo, weights=sign * flux[cf], minlength=nc)


def test_compute_trans_matches_textbook_tpfa_on_uniform_grid():
    G = compute_geometry(cart_grid([5, 4, 3], [10.0, 8.0, 6.0]))
    k = 2.5e-13  # m^2
    rock = {"perm": np.full(G["cells"]["num"], k)}
    T = compute_trans(G, rock)

    dx, dy, dz = 2.0, 2.0, 2.0  # cell size for this grid
    cell_faces = G["cells"]["faces"]
    # tag: 1=W,2=E,3=S,4=N,5=T,6=B (MRST direction codes, kept as-is)
    area_by_axis = {1: dy * dz, 2: dy * dz, 3: dx * dz, 4: dx * dz, 5: dx * dy, 6: dx * dy}
    dist_by_axis = {1: dx / 2, 2: dx / 2, 3: dy / 2, 4: dy / 2, 5: dz / 2, 6: dz / 2}

    expected = np.array([k * area_by_axis[tag] / dist_by_axis[tag] for tag in cell_faces[:, 1]])
    assert np.allclose(T, expected, rtol=1e-10)


def test_incomp_tpfa_1d_linear_pressure_profile():
    n = 20
    L = 100.0
    G = compute_geometry(cart_grid([n], [L]))
    k = 1e-13
    mu = 1e-3
    rock = {"perm": np.full(n, k)}
    T = compute_trans(G, rock)
    fluid = {"mu": mu}

    p_left, p_right = 2.0e5, 1.0e5
    faces = G["faces"]["neighbors"]
    left_face = int(np.nonzero(faces[:, 0] < 0)[0][0])
    right_face = int(np.nonzero(faces[:, 1] < 0)[0][0])
    bc = {
        "face": np.array([left_face, right_face]),
        "type": ["pressure", "pressure"],
        "value": np.array([p_left, p_right]),
    }

    state = incomp_tpfa(G, T, fluid, bc=bc)

    x = G["cells"]["centroids"].reshape(-1)  # cell centroids already include the half-cell offset
    expected_p = p_left - (p_left - p_right) * x / L
    assert np.allclose(state["pressure"], expected_p, rtol=2e-3)

    # 1D grids carry unit cross-sectional area (G.faces.areas == 1, see geom_1d),
    # so Darcy flux here is k/mu * dP/dx * A with A=1.
    expected_flux = k / mu * (p_left - p_right) / L
    internal = (faces[:, 0] >= 0) & (faces[:, 1] >= 0)
    assert np.allclose(state["flux"][internal], expected_flux, rtol=2e-3)


@pytest.mark.parametrize("celldim,physdim", [([6, 5, 4], [6.0, 5.0, 4.0])])
def test_mass_conservation_with_wells_and_noflow_boundary(celldim, physdim):
    G = compute_geometry(cart_grid(celldim, physdim))
    nc = G["cells"]["num"]
    rock = {"perm": np.full(nc, 1e-13)}
    T = compute_trans(G, rock)
    fluid = {"mu": 1e-3}

    inj_cell = 0
    prod_cell = nc - 1
    wells = [
        {"cells": np.array([inj_cell]), "WI": np.array([1e-11]), "type": "rate", "val": 1e-4},
        {"cells": np.array([prod_cell]), "WI": np.array([1e-11]), "type": "bhp", "val": 1.0e5},
    ]

    state = incomp_tpfa(G, T, fluid, wells=wells)

    inj_flux = float(np.sum(state["wellSol"][0]["flux"]))
    prod_flux = float(np.sum(state["wellSol"][1]["flux"]))

    net_rate = np.zeros(nc)
    net_rate[inj_cell] += inj_flux
    net_rate[prod_cell] += prod_flux

    balance = _cell_flux_balance(G, state["flux"])
    assert np.allclose(balance, net_rate, atol=1e-14, rtol=1e-8)

    # Rate-controlled injector must realize exactly its target rate.
    assert np.isclose(inj_flux, 1e-4, rtol=1e-10)

    # Closed reservoir, steady state: everything injected must be produced.
    assert np.isclose(inj_flux, -prod_flux, rtol=1e-8)


def test_mass_conservation_with_source_and_dirichlet_bc():
    G = compute_geometry(cart_grid([8, 6], [8.0, 6.0]))
    nc = G["cells"]["num"]
    rock = {"perm": np.full(nc, 5e-13)}
    T = compute_trans(G, rock)
    fluid = {"mu": 1e-3}

    faces = G["faces"]["neighbors"]
    left_faces = np.nonzero(faces[:, 0] < 0)[0]
    bc = {
        "face": left_faces,
        "type": ["pressure"] * left_faces.size,
        "value": np.full(left_faces.size, 1.5e5),
    }
    src = {"cell": np.array([5, 17]), "rate": np.array([2e-5, -1e-5])}

    state = incomp_tpfa(G, T, fluid, bc=bc, src=src)

    net_rate = np.zeros(nc)
    net_rate[5] += 2e-5
    net_rate[17] += -1e-5

    # state['flux'] is defined on every half-face, including boundary ones, so
    # per-cell mass conservation must hold exactly regardless of face type.
    balance = _cell_flux_balance(G, state["flux"])
    assert np.allclose(balance, net_rate, atol=1e-12, rtol=1e-6)
