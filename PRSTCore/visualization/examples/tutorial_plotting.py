"""Port of MRST's ``tutorialPlotting.m`` (mrst-2026a/core/examples): a tour
of the core visualization routines (``plotGrid``/``plotCellData``/
``plotFaces``/``plotFaceData``/``plotWell``/``plotGridVolumes``), driven by
a small two-phase incompressible injector/producer simulation.

Headless (matplotlib ``Agg``): every section that would normally display
interactively is instead saved as a PNG under ``OUT_DIR``. The animated
saturation-front section (MRST's ``pause``/``drawnow`` loop over 20
timesteps) is run in full -- all 20 steps are solved and stored -- but only
a handful of representative frames are saved as images rather than all 20,
since a headless run has no interactive display to animate into.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from PRSTCore.gridprocessing.cart_grid import cart_grid
from PRSTCore.gridprocessing.compute_geometry import compute_geometry
from PRSTCore.solvers.incomp.compute_trans import compute_trans
from PRSTCore.solvers.incomp.incomp_tpfa import incomp_tpfa
from PRSTCore.solvers.incomp.init_simple_fluid import init_simple_fluid
from PRSTCore.solvers.incomp.init_state import init_res_sol
from PRSTCore.solvers.incomp.make_rock import make_rock
from PRSTCore.solvers.incomp.transport import implicit_transport, total_mobility
from PRSTCore.solvers.incomp.vertical_well import vertical_well
from PRSTCore.visualization import (plot_cell_data, plot_face_data, plot_faces,
                                     plot_grid, plot_grid_volumes, plot_well)

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    day = 86400.0
    centi_poise = 1.0e-3
    kilogram_per_m3 = 1.0
    barsa = 1.0e5
    milli_darcy = 9.869233e-16

    # ---- Plotting grids ----
    G = compute_geometry(cart_grid([10, 10, 3]))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plot_grid(G, ax=ax)
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("plotGrid(G)")
    savefig(fig, "01_plot_grid.png")

    # ---- MRST and patch: transparent edges/faces, custom color ----
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plot_grid(G, ax=ax, facecolor="blue", alpha=0.3, edgecolor="k", linewidth=0.1)
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("plotGrid(G, EdgeAlpha=0.1, FaceColor=blue)")
    savefig(fig, "02_plot_grid_style.png")

    # ---- plotGrid and subsets ----
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    equal_index = (np.arange(G["cells"]["num"]) + 1) % 2 == 0
    plot_grid(G, np.flatnonzero(equal_index), ax=ax, facecolor="red")
    plot_grid(G, np.flatnonzero(~equal_index), ax=ax, facecolor="blue")
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("plotGrid subsets")
    savefig(fig, "03_plot_grid_subsets.png")

    # ---- Set up a simple two-phase flow problem (incomp module) ----
    rock = make_rock(G, 100 * milli_darcy, 0.5)
    fluid = init_simple_fluid(
        mu=(1 * centi_poise, 10 * centi_poise),
        rho=(1014 * kilogram_per_m3, 859 * kilogram_per_m3),
        n=(2.0, 2.0),
    )
    nx, ny, nz = (int(x) for x in G["cartDims"])
    # MRST's tutorial injects composition [0,1] (pure oil) into an
    # all-water reservoir; this port's transport solver (explicit_/
    # implicit_transport's source term) only models injection of the
    # *tracked* phase (water, s = water saturation) -- so the scenario is
    # flipped here to a water injector into an all-oil reservoir, which
    # this solver models correctly and is the more standard waterflood
    # demo besides.
    W = vertical_well([], G, rock, 0, 0, type="bhp", val=1 * barsa, radius=0.1,
                       comp_i=[1.0, 0.0], name="Injector")
    W = vertical_well(W, G, rock, nx - 1, ny - 1, type="bhp", val=0 * barsa, radius=0.1,
                       comp_i=[0.0, 1.0], name="Producer")
    sol = init_res_sol(G, 0.0, s0=0.0)  # 'sol[\"s\"]' is the (1D) water saturation per cell

    T = compute_trans(G, rock)

    def psolve(state):
        mob = total_mobility(fluid, state["s"])
        return incomp_tpfa(G, T, fluid={"mu": fluid.mu[0]}, wells=W, mob=mob)

    def tsolve(state, dt):
        return implicit_transport(G, state, rock, fluid, dt, wells=W)

    # ---- Plot the pressure distribution ----
    sol = {**sol, **psolve(sol)}
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plot_cell_data(G, sol["pressure"], ax=ax)
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("Pressure distribution")
    savefig(fig, "04_pressure.png")

    # ---- Plot subset of data (middle j-slice) + wells ----
    I, J, K = np.unravel_index(np.arange(G["cells"]["num"]), (nx, ny, nz), order="F")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plot_grid(G, ax=ax, alpha=0.0, edgecolor="k", linewidth=0.1)
    mid_slice = np.flatnonzero(J == ny // 2)
    plot_cell_data(G, sol["pressure"], mid_slice, ax=ax)
    plot_well(G, W, ax=ax)
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("Pressure (mid j-slice) + wells")
    savefig(fig, "05_pressure_slice_wells.png")

    # ---- plotFaces: faces with positive z-normal ----
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    pos_z = np.flatnonzero(np.asarray(G["faces"]["normals"])[:, 2] > 0)
    plot_faces(G, pos_z, ax=ax)
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("plotFaces: positive z-normal faces")
    savefig(fig, "06_plot_faces.png")

    # ---- plotFaceData: color by face-centroid z ----
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plot_face_data(G, np.asarray(G["faces"]["centroids"])[:, 2], ax=ax)
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("plotFaceData: face-centroid z")
    savefig(fig, "07_plot_face_data.png")

    # ---- Animated transport: solve all 20 steps, save representative frames ----
    dT = 10 * day
    solutions = []
    for i in range(20):
        sol = {**sol, **tsolve(sol, dT)}
        sol = {**sol, **psolve(sol)}
        solutions.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sol.items()})
        print(f"  transport step {i + 1}/20 done")

    for i in (4, 9, 19):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plot_grid(G, ax=ax, alpha=0.0, edgecolor="k", linewidth=0.1)
        sw = solutions[i]["s"]
        front = np.flatnonzero(sw > 0.05)
        if front.size:
            plot_cell_data(G, sw, front, ax=ax)
        plot_well(G, W, ax=ax)
        ax.view_init(elev=30, azim=50)
        ax.invert_zaxis()
        ax.set_title(f"Water saturation front, step {i + 1}")
        savefig(fig, f"08_saturation_step{i + 1:02d}.png")

    # ---- plotGridVolumes: same data, isosurface rendering ----
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sw_last = solutions[-1]["s"]
    plot_grid_volumes(G, sw_last, ax=ax, basealpha=2.0)
    plot_grid(G, ax=ax, alpha=0.0, edgecolor="k", linewidth=0.1)
    plot_well(G, W, ax=ax)
    ax.view_init(elev=30, azim=50)
    ax.invert_zaxis()
    ax.set_title("plotGridVolumes: final water saturation")
    savefig(fig, "09_grid_volumes.png")

    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
