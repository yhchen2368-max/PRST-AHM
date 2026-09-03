"""Port of MRST ``imposeMultipointVerticalRelpermScaling.m``
(mrst-2026a/hm/utils).

Attaches a *multipoint* relative-permeability scaler: instead of the two or
three endpoints ``initRelpermScaling`` provides, each curve gets a whole
(saturation, kr) table to be honoured, taken from the mobile part of the
supplied curve.

Accepted keywords, each an ``(n, 2)`` array of ``[saturation, kr]``:

    SW_KRW   water curve, from the last immobile point onward
    SW_KROW  oil-in-water curve, up to the first immobile point,
             re-expressed as an oil saturation (``1 - Sw``)
    SG_KRG   gas curve, from the last immobile point onward
    SG_KROG  oil-in-gas curve, re-expressed as ``1 - Sg - Swcon``
"""

import warnings as _warnings

import numpy as _np

VALID_MULTIPOINT = ('SW_KRW', 'SW_KROW', 'SG_KRG', 'SG_KROG')


def imposeMultipointVerticalRelpermScaling(model, nPoints=2, **scale):
    """Impose multipoint scaling given as ``KEYWORD=table`` pairs."""
    if not scale:
        return model
    if nPoints not in (2, 3):
        raise AssertionError('Only 2- or 3-point scaling is supported')

    fluid = model.fluid if isinstance(model.fluid, dict) else {}
    krPts = fluid.get('krPts')
    assert krPts, ("To impose rel-perm scaling, the fluid must contain field "
                   "'krPts'")

    # MATLAB reads model.rock.region.saturation here (singular), while the
    # rest of the module uses model.rock.regions -- accept either.
    rock = model.rock if isinstance(model.rock, dict) else {}
    for key in ('region', 'regions'):
        reg = (rock.get(key) or {}).get('saturation') if isinstance(
            rock.get(key), dict) else None
        if reg is not None:
            assert _np.unique(_np.asarray(reg)).size == 1, (
                'Multiple saturation tables do not support defining '
                'multipoint scalers yet.')

    w_table = krPts.get('w')
    swcon = float(_np.atleast_2d(_np.asarray(w_table, dtype=float))[0, 0]) \
        if w_table is not None else 0.0

    scale = {str(k).upper(): _np.atleast_2d(_np.asarray(v, dtype=float))
             for k, v in scale.items()}
    invalid = [k for k in scale if k not in VALID_MULTIPOINT]
    if invalid:
        _warnProblem(invalid)
        scale = {k: v for k, v in scale.items() if k in VALID_MULTIPOINT}

    nc = int(model.G['cells']['num'])
    if 'krscale' not in rock:
        # MRST keeps initRelpermScaling in model-io/deckformat/params/rock,
        # not in hm; PRSTCore mirrors that path.
        from PRSTCore.deckformat.params.rock import initRelpermScaling as _irs
        initRelpermScaling = _irs.initRelpermScaling
        rock['krscale'] = initRelpermScaling({'PROPS': scale}, nc)
    rock.setdefault('krscale', {})
    if 'multipoint' not in rock['krscale']:
        rock['krscale']['multipoint'] = {
            'w': _np.full(2, _np.nan), 'ow': _np.full(2, _np.nan),
            'og': _np.full(2, _np.nan), 'g': _np.full(2, _np.nan)}
    model.rock = rock

    for key, table in scale.items():
        lowered = key.lower()
        if lowered == 'sw_krw':
            values, loc = _from_last_immobile(table), 'w'
        elif lowered == 'sw_krow':
            values = _to_first_immobile(table)
            values[:, 0] = 1.0 - values[:, 0]
            loc = 'ow'
        elif lowered == 'sg_krg':
            values, loc = _from_last_immobile(table), 'g'
        elif lowered == 'sg_krog':
            values = _to_first_immobile(table)
            values[:, 0] = 1.0 - values[:, 0] - swcon
            loc = 'og'
        else:
            continue
        model.rock['krscale']['multipoint'][loc] = values

    _ensure_endscale(model, 'YES' if nPoints == 3 else 'NO')
    return model


def _from_last_immobile(table):
    """``ii = find(val(:,2) == 0, 1, 'last'); val = val(ii+1:end,:)``."""
    zeros = _np.flatnonzero(table[:, 1] == 0.0)
    start = int(zeros[-1]) + 1 if zeros.size else 0
    return table[start:, :].copy()


def _to_first_immobile(table):
    """``ii = find(val(:,2) == 0, 1, 'first'); val = val(1:ii-1,:)``."""
    zeros = _np.flatnonzero(table[:, 1] == 0.0)
    stop = int(zeros[0]) if zeros.size else table.shape[0]
    return table[:stop, :].copy()


def _ensure_endscale(model, scalecrs):
    deck = getattr(model, 'inputdata', None)
    if not deck:
        model.inputdata = {
            'RUNSPEC': {'ENDSCALE': ['NODIR', 'REVERS', 1, 20, 0]},
            'PROPS': {'SCALECRS': [scalecrs], 'MULTSCALECRS': ['YES']},
            'GRID': None, 'SOLUTION': None,
        }
        return model
    runspec = deck.setdefault('RUNSPEC', {})
    runspec.setdefault('ENDSCALE', ['NODIR', 'REVERS', 1, 20, 0])
    props = deck.setdefault('PROPS', {})
    props.setdefault('SCALECRS', [scalecrs])
    if 'MULTSCALECRS' not in props:
        # The MATLAB writes SCALECRS here, not MULTSCALECRS:
        #     if ~isfield(model.inputdata.PROPS, 'MULTSCALECRS')
        #         model.inputdata.PROPS.SCALECRS = {'YES'};
        # so an existing SCALECRS is overwritten with 'YES' and
        # MULTSCALECRS is never set at all. This is on a reachable path
        # and is almost certainly a typo, but it is what the module does,
        # so it is reproduced rather than corrected.
        props['SCALECRS'] = ['YES']
    return model


def _warnProblem(problems):
    for name in problems:
        _warnings.warn('Ignoring unrecognized/unsupported scaling keyword: %s'
                       % name, RuntimeWarning)
