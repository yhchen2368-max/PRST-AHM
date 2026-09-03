"""Port of MRST ``evaluateMatchFromEclipseRun.m`` and
``evaluateObjectiveFromEclipseRun.m`` (mrst-2026a/hm/utils/evaluate).

The same objective/gradient contract as :mod:`evaluateObjective`, but the
forward simulation is run by an external ECLIPSE-family simulator instead
of PRSTCore: the tuned parameters are written back into the deck, the deck
is written to disk, the simulator is invoked, and its restart output is
read back for the adjoint.

.. warning::
   These cannot be exercised here. They shell out to ``eclrun eclipse``,
   ``eclrun e300`` or tNavigator, none of which is present, so what is
   ported is the deck-preparation, command-construction and
   result-alignment logic; the invocation itself is faithful but untested.
   Everything that does not depend on an external binary -- the parameter
   round-trip, the well-index recompute trigger, the command strings -- is
   factored out so it can be, and is, tested directly.

One behaviour worth naming: ``evaluateMatchFromEclipseRun`` **negates**
the misfit (``misfitVal = -sum(...)/objScaling``) where
``evaluateObjective`` does not, because the optimisers it feeds maximise.
"""

import os as _os
import subprocess as _subprocess

import numpy as _np

from .updateDeckSchedule import updateDeckSchedule

_TNAV_ARGS = ('--no-dump-res --ecl-root -e -i -r -u --no-gui --ignore-lock '
              '--use-gpu')

# Parameters whose change invalidates the well indices.
_PERM_NAMES = ('permx', 'permy', 'permz')


def build_simulator_command(simulator, datafile):
    """The command line MRST builds for each supported simulator."""
    name = str(simulator).lower()
    if 'e300' in name:
        return 'eclrun e300 %s' % datafile
    if 'eclipse' in name:
        return 'eclrun %s %s' % (simulator, datafile)
    if 'tnavigator' in name:
        return '%s %s %s' % (simulator, _TNAV_ARGS, datafile)
    raise ValueError('Unsupported simulator')


def needs_well_index_recompute(parameters):
    """True when any tuned parameter is a permeability.

    The well index depends on permeability, so it must be recomputed
    before the deck is written -- otherwise the exported case keeps the
    indices of the untuned model.
    """
    names = [str(_get(p, 'name')).lower() for p in parameters]
    return any(n in _PERM_NAMES for n in names)


def apply_parameters(setup, parameters, pvec, enforceBounds=True):
    """Unscale the parameter vector into a fresh setup.

    Shared by both entry points and by :mod:`evaluateObjective`; returns
    ``(setupNew, pval)``.
    """
    nparam = [int(_get(p, 'nParam')) for p in parameters]
    p = _np.asarray(pvec, dtype=float).ravel()
    if enforceBounds:
        p = _np.clip(p, 0.0, 1.0)
    bounds = _np.concatenate([[0], _np.cumsum(nparam)]).astype(int)

    setupNew = dict(setup)
    model = setupNew['model']
    for field in ('FlowDiscretization', 'FlowPropertyFunctions',
                  'PVTPropertyFunctions'):
        if hasattr(model, field):
            setattr(model, field, None)

    pval = []
    for k, param in enumerate(parameters):
        value = _get(param, 'unscale')(p[bounds[k]:bounds[k + 1]])
        pval.append(value)
        setupNew = _get(param, 'setParameter')(setupNew, value)
    return setupNew, pval


def evaluateMatchFromEclipseRun(pvec, obj, setup, parameters, states_ref, deck,
                                path, name, simulator='eclipse',
                                objScaling=1.0, enforceBounds=True,
                                AdjointLinearSolver=None,
                                return_gradient=False, return_states=False,
                                return_setup=False):
    """Run the case through ECLIPSE and score it.

    Returns the *negated* misfit, matching the MATLAB.
    """
    from PRSTCore.deckformat.deckoutput.write_deck import write_deck
    from PRSTCore.hm.utils.recomputeWellIndex import recomputeWellIndex
    from PRSTCore.hm.utils.updateDeckFromModelParameter import \
        updateDeckFromModelParameter

    setupNew, pval = apply_parameters(setup, parameters, pvec, enforceBounds)

    if needs_well_index_recompute(parameters):
        setupNew['schedule'] = recomputeWellIndex(setupNew['model'],
                                                  setupNew['schedule'])

    deck = updateDeckFromModelParameter(deck, setupNew, parameters)
    datafile = _os.path.join(str(path), '%s.DATA' % name)
    command = build_simulator_command(simulator, datafile)

    write_deck(deck, path, filename=name)
    _subprocess.run(command, shell=True, check=False)

    try:
        states, wellSols, setupNew = _read_results(path, name, setupNew)
    except Exception:
        # MATLAB prints and returns; a failed external run leaves nothing
        # to score, so say so rather than returning a misleading number.
        raise RuntimeError(
            'Unable to read Eclipse results. There may be some errors '
            'during simulation.')

    misfitVals = obj(setupNew['model'], states, setupNew['schedule'],
                     states_ref, False, None, None)
    misfitVal = -float(_np.sum(_np.concatenate(
        [_np.atleast_1d(_np.asarray(v, dtype=float)).ravel()
         for v in misfitVals]))) / objScaling

    out = [misfitVal]
    if return_gradient:
        out.append(_adjoint_gradient(obj, setupNew, parameters, pval, states,
                                     states_ref, objScaling))
    if return_states:
        out.extend([wellSols, states])
    if return_setup:
        out.append(setupNew)
    return out[0] if len(out) == 1 else tuple(out)


def evaluateObjectiveFromEclipseRun(pvec, obj, setup, parameters, deck, path,
                                    name, simulator='eclipse',
                                    objScaling=1.0, enforceBounds=True,
                                    writeScheduleOnly=False,
                                    scheduleName=None, unit='metric',
                                    return_gradient=False,
                                    return_states=False, return_setup=False):
    """Write the tuned schedule (or the whole deck) and run it."""
    from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
    from PRSTCore.deckformat.deckoutput.write_deck import write_deck
    from PRSTCore.hm.utils.reduceEclipseDeckSchedule import \
        reduceEclipseDeckSchedule

    setupNew, pval = apply_parameters(setup, parameters, pvec, enforceBounds)
    deck['SCHEDULE'] = updateDeckSchedule(deck, setupNew['model'].G,
                                          setupNew['schedule'])

    if writeScheduleOnly:
        deck = convert_deck_units(deck, outputUnit=unit)
        deck = reduceEclipseDeckSchedule(deck)
        from PRSTCore.deckformat.deckoutput.write_schedule import write_schedule
        write_schedule(scheduleName or 'SCHEDULE_NEW.INC', path,
                       deck['reducedSCHEDULE'],
                       start=deck['RUNSPEC'].get('START'))
    else:
        write_deck(deck, path, filename=name)

    datafile = _os.path.join(str(path), '%s.DATA' % name)
    _subprocess.run(build_simulator_command(simulator, datafile),
                    shell=True, check=False)

    states, wellSols, setupNew = _read_results(path, name, setupNew)
    objVals = obj(setupNew['model'], states, setupNew['schedule'],
                  False, None, None, True)
    objVal = float(_np.sum(_np.concatenate(
        [_np.atleast_1d(_np.asarray(v, dtype=float)).ravel()
         for v in objVals]))) / objScaling

    out = [objVal]
    if return_states:
        out.extend([wellSols, states])
    if return_setup:
        out.append(setupNew)
    return out[0] if len(out) == 1 else tuple(out)


def _read_results(path, name, setupNew):
    from .getEclipseSimResults import getEclipseSimResults
    return getEclipseSimResults(path, name, setupNew)


def _adjoint_gradient(obj, setupNew, parameters, pval, states, states_ref,
                      objScaling):
    from PRSTCore.ad_core.simulators import compute_sensitivities_adjoint_ad

    def objh(tstep, model, state):
        return obj(setupNew['model'], states, setupNew['schedule'],
                   states_ref, True, tstep, state)

    gradient = compute_sensitivities_adjoint_ad(setupNew, states, parameters,
                                                objh)
    scaled = []
    for k, param in enumerate(parameters):
        scaled.append(_np.atleast_1d(_np.asarray(
            _get(param, 'scaleGradient')(gradient[_get(param, 'name')],
                                         pval[k]), dtype=float)).ravel())
    return _np.concatenate(scaled) / objScaling


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)
