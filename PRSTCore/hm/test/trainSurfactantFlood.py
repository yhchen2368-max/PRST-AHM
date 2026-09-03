"""Port of MRST ``trainSurfactantFlood.m`` (mrst-2026a/hm/test).

The surfactant counterpart of :mod:`trainPolymerFlood`. Two things
differ: the fluid group also carries the *surfactant-state* endpoints
(the ``ss*``/``skr*`` names, the saturation functions that apply at full
surfactant concentration), and the surfactant group tunes the capillary
desaturation curve's limits rather than adsorption or viscosity.

Adsorption, viscosity and surface-tension parameters are commented out
in the MATLAB. They are listed in :data:`DISABLED_SURFACTANT_PARAMETERS`
here rather than dropped, so what MRST chose not to tune stays visible.
"""

import numpy as _np

from PRSTCore.optimization import evaluate_match_summands
from PRSTCore.optimization.optim.unit_box_lm import unit_box_lm
from PRSTCore.optimization.utils.parameters import (
    add_parameter, get_scaled_parameter_vector,
    update_setup_from_scaled_parameters)

_TYPES = ('rock', 'rock+fluid', 'rock+surfactant', 'rock+fluid+surfactant')

#: Commented out in MRST's trainSurfactantFlood.m -- adsorption,
#: viscosity and surface-tension tables, with the limits it would have
#: used. Kept for reference; not tuned.
DISABLED_SURFACTANT_PARAMETERS = {
    'adcsu': [0.0, 200.0], 'adusft': [5e-5, 5e-3],
    'vscsu': [0.0, 200.0], 'vsusft': [1e-3, 1e-2],
    'stcsu': [0.0, 200.0], 'stusft': [5e-3, 5e-1],
}


def trainSurfactantFlood(trainSetup, mismatchFn, observed,
                         NonLinearSolver=None, ParameterType='rock',
                         lumping=False):
    """Return ``(history, trainParms, wellSols, states)``."""
    if ParameterType not in _TYPES:
        raise ValueError('ParameterType must be one of %s, got %r'
                         % (', '.join(_TYPES), ParameterType))
    nls = NonLinearSolver

    nc = int(trainSetup['model'].G['cells']['num'])
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
                       ('krw', [0.3, 0.8]), ('kro', [0.3, 0.8]),
                       # The same endpoints at full surfactant saturation.
                       ('sswl', [0.0, 0.4]), ('sswcr', [0.0, 0.4]),
                       ('sswu', [0.7, 1.0]), ('ssowcr', [0.0, 0.4]),
                       ('skrw', [0.8, 1.5]), ('skro', [0.8, 1.5])):
        trainFluid = add_parameter(trainFluid, trainSetup, name=name,
                                   box_lims=lims, scaling='linear',
                                   lumping=lump)

    trainSurf = []
    for name, lims in (('dsncl', [-20.0, 0.0]), ('dsncu', [0.0, 20.0])):
        trainSurf = add_parameter(trainSurf, trainSetup, name=name,
                                  box_lims=lims, scaling='linear',
                                  lumping=lump)

    groups = {'rock': trainRock, 'fluid': trainFluid, 'surfactant': trainSurf}
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
