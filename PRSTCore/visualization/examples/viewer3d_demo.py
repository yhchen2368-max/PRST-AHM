"""Open the 3D viewer on a small synthetic model.

Nothing here needs a deck or a simulation run: the grid comes from
``cart_grid`` and the fields are made up, so this is the quickest way to
check that VTK, Qt and the viewer itself are working before pointing them at
a real model.

Run it with the interpreter that has ``vtk`` and ``PySide6`` -- on this
machine that is anaconda3 (3.13), not ``.conda`` (3.14), which has no VTK
wheel available::

    python PRSTCore/visualization/examples/viewer3d_demo.py

``--screenshot PATH`` renders one frame to a PNG and exits instead of
entering the event loop, which is what makes the viewer testable from a
script.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from PRSTCore.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from PRSTCore.visualization import ReservoirScene  # noqa: E402


def build_model(nsteps=12):
    """A box grid, two wells, a static field and one that moves in time."""
    G = compute_geometry(cart_grid([40, 30, 8], [800.0, 600.0, 80.0]))
    centroids = np.asarray(G["cells"]["centroids"])
    x, y, z = centroids.T

    poro = 0.12 + 0.08 * np.sin(x / 120.0) * np.cos(y / 90.0)
    permx = 1e-13 * np.exp(3.0 * (poro - 0.12) / 0.08)

    injector = np.array([60.0, 60.0])
    producer = np.array([740.0, 540.0])
    W = [
        {"name": "INJ1",
         "cells": np.flatnonzero((np.abs(x - injector[0]) < 12)
                                 & (np.abs(y - injector[1]) < 12))},
        {"name": "PROD1",
         "cells": np.flatnonzero((np.abs(x - producer[0]) < 12)
                                 & (np.abs(y - producer[1]) < 12))},
    ]

    # A front sweeping from the injector towards the producer, so stepping
    # through time actually shows something moving.
    distance = np.hypot(x - injector[0], y - injector[1])
    reach = np.linspace(0.0, 1.4 * distance.max(), nsteps)
    saturation = np.array([1.0 / (1.0 + np.exp((distance - r) / 40.0))
                           for r in reach])
    pressure = np.array([250.0 - 40.0 * t / max(nsteps - 1, 1) + 0.01 * z
                         for t in range(nsteps)])

    return G, W, {"PORO": poro, "PERMX": permx}, {"SW": saturation,
                                                  "PRESSURE": pressure}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", metavar="PATH",
                        help="render one frame to PATH and exit")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    G, W, static, dynamic = build_model(args.steps)

    if args.screenshot:
        import vtk

        scene = ReservoirScene(G, W=W, static_fields=static)
        for name, values in dynamic.items():
            scene.add_field(name, values)

        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.SetSize(1200, 800)
        scene.attach(window)
        scene.set_active_field("SW")
        scene.set_colormap("turbo")
        scene.set_step(args.steps // 2)
        scene.reset_camera()
        window.Render()

        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(window)
        grab.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(args.screenshot)
        writer.SetInputConnection(grab.GetOutputPort())
        writer.Write()
        print("wrote", args.screenshot)
        return

    from PRSTCore.visualization import view_reservoir

    view_reservoir(G, W=W, static_fields=static, fields=dynamic,
                   title="PRSTCore 3D -- synthetic model")


if __name__ == "__main__":
    main()
