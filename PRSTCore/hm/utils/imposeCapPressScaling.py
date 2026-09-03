"""Port of MRST ``imposeCapPressScaling.m`` (mrst-2026a/hm/utils).

Attaches capillary-pressure endpoint scaling to a model, and makes sure the
model's deck advertises ENDSCALE so the scaling is actually consulted.

Two places where the MATLAB as written cannot run, both on branches the
normal path does not reach; the port follows the evident intent and says
so rather than reproducing an error:

* the ``rock.pcscale`` update indexes ``map.kw.(fn{i})``, but
  ``getPcScalerMap`` returns ``struct('ph', ..., 'pc', pc)`` -- there is no
  ``kw`` field. The port uses ``pc``, the only mapping defined.
* ``warnProblem(prob)`` is called with an undefined ``prob``. The port
  warns with the offending keyword names.
"""

import warnings as _warnings

import numpy as _np

from .initCapPressScaling import initCapPressScaling

VALID_KEYWORDS = ('SWLPC', 'PCW', 'SGLPC', 'PCG')

# getPcScalerMap: phase index and column within pcscale.<branch>.<phase>.
_PHASES = ('w', 'g')
_PC_MAP = {'SWLPC': (0, 0), 'PCW': (0, 1), 'SGLPC': (1, 0), 'PCG': (1, 1)}
_IMB_MAP = {'ISWLPC': (0, 0), 'IPCW': (0, 1), 'ISGLPC': (1, 0), 'IPCG': (1, 1)}


def imposeCapPressScaling(model, **scale):
    """Impose Pc scaling given as ``KEYWORD=values`` pairs.

    Each value is a scalar (broadcast over the grid) or one value per cell.
    """
    if not scale:
        return model

    scale = {str(k).upper(): v for k, v in scale.items()}
    known = set(VALID_KEYWORDS) | set(_IMB_MAP)
    invalid = [k for k in scale if k not in known]
    if invalid:
        _warnProblem(invalid)
        scale = {k: v for k, v in scale.items() if k in known}

    nc = int(model.G['cells']['num'])
    for key, value in list(scale.items()):
        values = _np.atleast_1d(_np.asarray(value, dtype=float)).ravel()
        assert values.size in (1, nc), \
            'Scaling values does not match number of grid cells'
        if values.size == 1:
            values = _np.full(nc, values[0], dtype=float)
        scale[key] = values

    if not isinstance(model.rock, dict):
        raise TypeError('imposeCapPressScaling requires a dict rock')

    if 'pcscale' not in model.rock:
        model.rock['pcscale'] = initCapPressScaling({'PROPS': scale}, nc)
    else:
        pcscale = model.rock['pcscale']
        for key, values in scale.items():
            if key in _PC_MAP:
                branch, (ph, col) = 'drainage', _PC_MAP[key]
            elif key in _IMB_MAP:
                branch, (ph, col) = 'imbibition', _IMB_MAP[key]
            else:
                print('Unsupported capillary pressure scaling keyword: %s' % key)
                continue
            pcscale[branch][_PHASES[ph]][:, col] = values

    _ensure_endscale(model)
    return model


def _ensure_endscale(model, scalecrs='NO'):
    """The deck must advertise ENDSCALE for the scaling to be consulted."""
    endscale = ['NODIR', 'REVERS', 1, 20, 0]
    deck = getattr(model, 'inputdata', None)
    if not deck:
        model.inputdata = {
            'RUNSPEC': {'ENDSCALE': endscale},
            'PROPS': {'SCALECRS': [scalecrs]},
            'GRID': None, 'SOLUTION': None,
        }
        return model
    runspec = deck.setdefault('RUNSPEC', {})
    runspec.setdefault('ENDSCALE', endscale)
    props = deck.setdefault('PROPS', {})
    props.setdefault('SCALECRS', [scalecrs])
    return model


def _warnProblem(problems):
    for name in problems:
        _warnings.warn('Ignoring unrecognized/unsupported scaling keyword: %s'
                       % name, RuntimeWarning)
