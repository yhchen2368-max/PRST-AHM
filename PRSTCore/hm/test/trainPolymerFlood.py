"""Port of MRST ``trainPolymerFlood.m`` (mrst-2026a/hm/test).

Calibrates a polymer-flood model against observed well data. The
parameters come in three groups -- rock, saturation-function endpoints,
and polymer -- and ``ParameterType`` selects which combination is tuned.

The optimiser is Levenberg-Marquardt on the mismatch *summands* rather
than their sum, since LM needs the residual vector, not a scalar.
"""

import numpy as _np

from PRSTCore.optimization import evaluate_match_summands
from PRSTCore.optimization.optim.unit_box_lm import unit_box_lm
from PRSTCore.optimization.utils.parameters import (
    add_parameter, get_scaled_parameter_vector,
    update_setup_from_scaled_parameters)

_TYPES = ('rock', 'rock+fluid', 'rock+polymer', 'rock+fluid+polymer')


def trainPolymerFlood(trainSetup, mismatchFn, observed, NonLinearSolver=None,
                      ParameterType='rock', lumping=False):
    """Return ``(history, trainParms, wellSols, states)``."""
    if ParameterType not in _TYPES:
        raise ValueError('ParameterType must be one of %s, got %r'
                         % (', '.join(_TYPES), ParameterType))
    nls = NonLinearSolver

    nc = int(trainSetup['model'].G['cells']['num'])
    # lumping=True ties every cell to one value; otherwise each cell is
    # its own parameter.
    lump = _np.ones(nc, dtype=int) if lumping else _np.arange(nc)

    trainRock = []
    trainRock = add_parameter(trainRock, trainSetup, name='porevolume',
                              relative_limits=[0.01, 1.5], scaling='linear',
                              uniform_limits=False)
    trainRock = add_parameter(trainRock, trainSetup, name='conntrans',
                              relative_limits=[0.01, 100], scaling='log')
    trainRock = add_parameter(trainRock, trainSetup, name='transmissibility',
                              relative_limits=[0.01, 100], scaling='log')

    trainFluid = []
    for name, lims in (('swl', [0.0, 0.4]), ('swcr', [0.0, 0.4]),
                       ('swu', [0.7, 1.0]), ('sowcr', [0.0, 0.4]),
                       ('krw', [0.3, 0.8]), ('kro', [0.3, 0.8])):
        trainFluid = add_parameter(trainFluid, trainSetup, name=name,
                                   box_lims=lims, scaling='linear',
                                   lumping=lump)

    trainPoly = []
    # 'vsuply' is added twice in the MATLAB, once redundantly; the second
    # is dropped here because a duplicate parameter would double-count
    # the same quantity in the residual and in the Jacobian.
    for name, lims in (('aduply', [1e-7, 5e-5]), ('vsuply', [1.0, 100.0])):
        trainPoly = add_parameter(trainPoly, trainSetup, name=name,
                                  box_lims=lims, scaling='linear',
                                  lumping=lump)
    trainPoly = add_parameter(trainPoly, trainSetup, name='mixPar',
                              box_lims=[0.0, 1.0], scaling='linear')

    groups = {'rock': trainRock, 'fluid': trainFluid, 'polymer': trainPoly}
    trainParms = []
    for part in ParameterType.split('+'):
        trainParms = trainParms + groups[part]

    pvec = get_scaled_parameter_vector(trainSetup, trainParms)

    def objh(p):
        return evaluate_match_summands(p, mismatchFn, trainSetup, trainParms,
                                       observed, nonlinear_solver=nls)

    _, _, history = unit_box_lm(pvec, objh, max_it=50, update_tol=1e-10,
                                res_tol_abs=1e-6, update_strategy='TR')

    trained = update_setup_from_scaled_parameters(trainSetup, trainParms,
                                                  history['u'][-1])
    from PRSTCore.ad_core.simulators.simulate_schedule_ad import \
        simulate_schedule_ad
    wellSols, states = simulate_schedule_ad(
        trained['state0'], trained['model'], trained['schedule'],
        nonlinear_solver=nls)[:2]
    return history, trainParms, wellSols, states
