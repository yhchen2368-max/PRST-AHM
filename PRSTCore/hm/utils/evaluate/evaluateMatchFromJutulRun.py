"""Port of MRST ``evaluateMatchFromJutulRun.m``
(mrst-2026a/hm/utils/evaluate).

The Jutul (Julia) counterpart of :mod:`evaluateMatchFromEclipseRun`: the
tuned setup is written out in Jutul's input format, a small ``.jl`` driver
is generated next to it, Julia runs that driver, and the resulting states
are read back for the adjoint.

.. warning::
   This cannot be exercised here -- it requires a Julia installation with
   Jutul and JutulDarcy, and MRST's ``writeJutulInput``, neither of which
   is present. The driver-script generation is faithful and is tested
   directly (it is pure text); the invocation and read-back are ported for
   completeness and are untested.
"""

import os as _os
import subprocess as _subprocess

import numpy as _np

from .evaluateMatchFromEclipseRun import apply_parameters, _get

# The solver settings MRST writes into the generated driver.
JUTUL_SETTINGS = (
    ('tol_cnv', '1e-2'),
    ('tol_mb', '1e-4'),
    ('max_nonlinear_iterations', '50'),
    ('max_timestep_cuts', '8'),
    ('precond', ':ilu0'),
    ('wells', ':simple'),
    ('linear_solver', ':gmres'),
)


def build_jutul_driver(jpth):
    """The ``.jl`` driver MRST writes beside the Jutul input."""
    lines = ['using Jutul, JutulDarcy',
             'jpth = "%s"' % jpth,
             'arg = (max_iterations = 200, rtol = 1e-2, atol = 1e-6);',
             'simulate_mrst_case(jpth,']
    lines += ['                    %s = %s,' % (k, v) for k, v in JUTUL_SETTINGS]
    lines.append('                    )')
    return '\n'.join(lines) + '\n'


def evaluateMatchFromJutulRun(pvec, obj, setup, parameters, states_ref,
                              name='jutul_case', objScaling=1.0,
                              enforceBounds=True, return_gradient=False,
                              return_states=False, return_setup=False):
    """Run the case through Jutul and score it."""
    from PRSTCore.hm.utils.processJutulStates import processJutulStates

    setupNew, pval = apply_parameters(setup, parameters, pvec, enforceBounds)

    jpth = _write_jutul_input(setupNew['state0'], setupNew['model'],
                              setupNew['schedule'], name)
    directory, base = _os.path.split(jpth)
    driver = _os.path.join(directory, '%s.jl' % _os.path.splitext(base)[0])
    with open(driver, 'w', encoding='utf-8') as handle:
        handle.write(build_jutul_driver(jpth))

    _subprocess.run('julia %s' % driver, shell=True, check=False)

    wellSols, states = _read_jutul_results(jpth)
    wellSols, states = processJutulStates(setupNew, wellSols, states)

    misfitVals = obj(setupNew['model'], states, setupNew['schedule'],
                     states_ref, False, None, None)
    misfitVal = -float(_np.sum(_np.concatenate(
        [_np.atleast_1d(_np.asarray(v, dtype=float)).ravel()
         for v in misfitVals]))) / objScaling

    out = [misfitVal]
    if return_gradient:
        from .evaluateMatchFromEclipseRun import _adjoint_gradient
        out.append(_adjoint_gradient(obj, setupNew, parameters, pval, states,
                                     states_ref, objScaling))
    if return_states:
        out.extend([wellSols, states])
    if return_setup:
        out.append(setupNew)
    return out[0] if len(out) == 1 else tuple(out)


def _write_jutul_input(state0, model, schedule, name):
    raise NotImplementedError(
        "evaluateMatchFromJutulRun needs MRST's writeJutulInput, which "
        'PRSTCore has not ported; build_jutul_driver and processJutulStates '
        'are available independently.')


def _read_jutul_results(jpth):
    raise NotImplementedError(
        'Reading a Jutul run needs its output format reader, which PRSTCore '
        'has not ported.')
