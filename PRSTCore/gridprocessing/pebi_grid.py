"""Python port of MRST's ``pebi.m`` (mrst-2026a/core/gridprocessing): the PEBI
(Perpendicular Bisector, i.e. Voronoi) dual of a set of points/triangle grid.

Where MRST's ``pebi.m`` bounds the bordering Voronoi cells by adding
auxiliary triangles derived from the input triangulation's own boundary
edges, this port uses the equivalent, more standard "ghost point" technique
-- a ring of distant points added before computing the Voronoi diagram --
so every real input point's cell is bounded, then delegates the actual
diagram construction to ``scipy.spatial.Voronoi`` (a robust, independently
validated implementation of the same underlying geometric construction)
rather than hand-porting MRST's triangulation-specific boundary-closing
algorithm. As with MRST's own docstring caveat, a non-Delaunay/degenerate
input triangulation is not meaningful here since only the point *set*
(not a triangle list) drives the dual; this function is called on points,
matching ``pebi(triangleGrid(p))``'s effective input.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

from .tessellation_grid import tessellation_grid


def pebi_grid(points, *, ghost_ring_points: int = 24, ghost_radius_factor: float = 10.0) -> dict:
    """Compute the PEBI (Voronoi) dual grid of a 2D point set.

    Parameters
    ----------
    points : dict or (n, 2) array
        Either raw point coordinates, or a triangle-grid dict (as returned
        by :func:`PRSTCore.gridprocessing.triangle_grid.triangle_grid`) --
        in the latter case its node coordinates are used, matching MRST's
        ``pebi(triangleGrid(p))`` idiom.
    ghost_ring_points, ghost_radius_factor : int, float
        Number of, and distance-multiplier for, the ring of auxiliary
        points added around the point cloud to keep every real point's
        Voronoi cell bounded.
    """
    p = np.asarray(points["nodes"]["coords"] if isinstance(points, dict) else points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("pebi_grid is only supported in two space dimensions")
    n = p.shape[0]

    center = p.mean(axis=0)
    radius = float(np.max(np.linalg.norm(p - center, axis=1))) * ghost_radius_factor + 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, ghost_ring_points, endpoint=False)
    ghosts = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])

    vor = Voronoi(np.vstack([p, ghosts]))

    polygons = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            raise RuntimeError(
                f"pebi_grid: unbounded Voronoi cell for point {i}; increase "
                f"ghost_ring_points/ghost_radius_factor"
            )
        region = np.asarray(region, dtype=np.int64)
        # scipy does not guarantee a consistent (CCW) winding across
        # different cells' regions; tessellation_grid's neighbor assignment
        # relies on adjacent cells traversing their shared edge in opposite
        # directions, so sort each polygon's vertices by angle around its
        # own site to force that consistently.
        verts = vor.vertices[region]
        angles = np.arctan2(verts[:, 1] - p[i, 1], verts[:, 0] - p[i, 0])
        polygons.append(region[np.argsort(angles)])

    G = tessellation_grid(vor.vertices, polygons)
    G["type"] = ["pebi"]
    return G
