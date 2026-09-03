"""Modify the well setup of the ``norne_simple_wo`` case.

Port of MRST ``modules/network-models/examples/utils/modifyNorneTest.m``.

This is the case generator behind ``norneGPSNetGeneralityTest``: it takes
one test case and produces five well configurations from it, so a network
model trained on one of them can be scored against the others. The point
of the exercise is that the modifications are *outside* what the training
saw -- a shut-in producer, injectors turned into producers -- which is
where a purely data-driven network usually falls over.

``caseNo``:

0
    No modification.
1
    25% random perturbation of the well rates, 10% of the bhp controls.
2
    Shut in the dominant producer (P1) after 2/3 of the simulation time.
3
    Convert injectors I1/I2 to producers and shut in P1.
4
    Shut in P1 during the middle third of the simulation horizon.
"""

import copy as _copy

import numpy as _np

__all__ = ['modify_norne_test']

#: ``W(7)`` in the MATLAB -- the dominant producer P1 in the
#: ``norne_simple_wo`` well ordering, 1-based there and 0-based here.
_P1 = 6


def modify_norne_test(test, case_no):
    """Return ``test`` with one of five well configurations applied.

    ``test`` is the dict form of MRST's ``TestCase``: ``name`` and
    ``schedule``. The input is not modified.
    """
    case_no = int(case_no)
    if case_no not in (0, 1, 2, 3, 4):
        raise ValueError('caseNo must be 0 to 4, not %r' % case_no)

    test = _copy.deepcopy(test)
    if case_no == 0:
        return test

    schedule = test['schedule']
    controls = schedule['control']

    if case_no == 1:
        # ``rng(499)`` -- the perturbation has to be the same on every
        # run, or two scorings of the same network are not comparable.
        rng = _np.random.default_rng(499)
        test['name'] = 'test1'
        for w in controls[0]['W']:
            kind = str(w.get('type', '')).lower()
            if kind == 'rate':
                w['val'] = float((0.5 + rng.random()) * w['val'])
            elif kind == 'bhp':
                w['val'] = float((0.9 + 0.2 * rng.random()) * w['val'])
        return test

    if case_no == 3:
        test['name'] = 'test3'
        W = controls[0]['W']
        producer = W[_P1]
        for w in W[:2]:
            # The two injectors take the producer's control *and* its
            # composition and sign: converting a well means converting
            # what it produces, not only its target.
            w['type'] = producer['type']
            w['val'] = producer['val']
            w['name'] = 'P' + str(w.get('name', ''))
            w['compi'] = _copy.deepcopy(producer.get('compi'))
            w['sign'] = producer.get('sign')
        producer['status'] = False
        return test

    # Cases 2 and 4 both add a second control with P1 shut and point a
    # stretch of report steps at it.
    test['name'] = 'test2' if case_no == 2 else 'test4'
    step_control = _np.asarray(schedule['step']['control']).ravel().copy()
    n = step_control.size

    if case_no == 2:
        lo, hi = _thirds(n, 2, 3)         # ``2*end/3 : end``
    else:
        lo, hi = _thirds(n, 1, 2)         # ``end/3 : 2*end/3``

    shut = _copy.deepcopy(controls[0])
    shut['W'][_P1]['status'] = False
    controls.append(shut)
    step_control[lo:hi] = len(controls) - 1
    schedule['step']['control'] = step_control
    return test


def _thirds(n, first, last):
    """The 0-based half-open range MATLAB writes as ``a*end/3 : b*end/3``.

    MATLAB's indices are 1-based and inclusive, and it rejects a
    fractional one outright -- so a step count that is not a multiple of
    three is an error there and is reported as one here rather than
    silently rounded to a different stretch of the simulation.
    """
    if n % 3:
        raise ValueError('Number of report steps (%d) must be a multiple of '
                         'three to be split into thirds' % n)
    lo = first * n // 3 - 1                       # 1-based start -> 0-based
    hi = n if last == 3 else last * n // 3        # inclusive end -> exclusive
    return lo, hi
