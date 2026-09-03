"""Parity tests for MRST ``poreVolume(G, rock)`` = ``poro .* volumes .* ntg``.

Cell volumes are stored under two different keys depending on which
constructor built the grid (``init_eclipse_grid`` writes a top-level
``cell_volumes``; ``compute_geometry`` writes ``cells.volumes``), and the
accumulation terms silently used a unit pore volume for the latter.
"""

import numpy as np
import pytest

from PRSTCore.ad_core.models.generic_black_oil_model import GenericBlackOilModel
from PRSTCore.gridprocessing.compute_geometry import compute_geometry
from PRSTCore.gridprocessing.tensor_grid import tensor_grid


def _grid():
    return compute_geometry(tensor_grid(np.arange(4.) * 10.0,
                                        np.arange(3.) * 10.0,
                                        np.arange(3.) * 5.0))


def _model(G, rock):
    return GenericBlackOilModel(G, rock, {}, water=True, oil=True, gas=True,
                                mrst_generic_assembly=True)


def test_compute_geometry_grid_uses_cells_volumes():
    G = _grid()
    nc = G['cells']['num']
    rock = {'poro': np.full(nc, 0.2), 'perm': np.full((nc, 3), 1e-13)}
    expected = G['cells']['volumes'] * 0.2
    assert np.allclose(_model(G, rock)._porevolume_vector(), expected)
    # The regression this guards: a unit pore volume for every cell.
    assert not np.allclose(_model(G, rock)._porevolume_vector(), 1.0)


def test_ntg_is_applied_when_present():
    G = _grid()
    nc = G['cells']['num']
    rock = {'poro': np.full(nc, 0.2), 'perm': np.full((nc, 3), 1e-13),
            'ntg': np.full(nc, 0.5)}
    expected = G['cells']['volumes'] * 0.2 * 0.5
    assert np.allclose(_model(G, rock)._porevolume_vector(), expected)


def test_scalar_ntg_is_broadcast():
    G = _grid()
    nc = G['cells']['num']
    rock = {'poro': np.full(nc, 0.2), 'perm': np.full((nc, 3), 1e-13),
            'ntg': np.array([0.5])}
    expected = G['cells']['volumes'] * 0.2 * 0.5
    assert np.allclose(_model(G, rock)._porevolume_vector(), expected)


def test_top_level_cell_volumes_key_still_honoured():
    """init_eclipse_grid's key must keep working alongside cells.volumes."""
    G = _grid()
    nc = G['cells']['num']
    volumes = G['cells']['volumes'].copy()
    G['cell_volumes'] = volumes
    rock = {'poro': np.full(nc, 0.2), 'perm': np.full((nc, 3), 1e-13)}
    assert np.allclose(_model(G, rock)._porevolume_vector(), volumes * 0.2)


def test_explicit_porevolume_overrides_grid():
    G = _grid()
    nc = G['cells']['num']
    rock = {'poro': np.full(nc, 0.2), 'perm': np.full((nc, 3), 1e-13)}
    model = _model(G, rock)
    model.porevolume = np.full(nc, 7.0)
    assert np.allclose(model._porevolume_vector(), 7.0)


@pytest.mark.parametrize('cr, pref', [(0.0, 0.0), (4.35e-10, 2.76e7)])
def test_rock_compressibility_matches_pvmult(cr, pref):
    """assignROCK.m pvMult: ``1 + x + 0.5*x^2`` with ``x = cR*(p-pRef)``."""
    G = _grid()
    nc = G['cells']['num']
    rock = {'poro': np.full(nc, 0.2), 'perm': np.full((nc, 3), 1e-13),
            'cr': cr, 'pref': pref}
    model = _model(G, rock)
    p = np.full(nc, 3.0e7)
    x = cr * (p - pref)
    expected = G['cells']['volumes'] * 0.2 * (1.0 + x + 0.5 * x * x)
    assert np.allclose(model._mrst_pore_volume(p), expected)
