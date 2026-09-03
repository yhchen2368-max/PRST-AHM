"""Port of MRST ``matchConstantPressureCore.m`` (mrst-2026a/hm/utils/evaluate).

Mismatch objective for a constant-pressure core flood: instead of well
rates it compares the water and oil fluxes across the *boundary faces*
named by the driving forces' BC.

``matchCases = [1; -1]`` -- the inlet and outlet faces enter with opposite
sign, so the term measures the flux imbalance across the core rather than
each face independently.

The face sign convention follows ``getBoundaryFlux``: a face whose second
neighbour is absent points out of the domain, so its flux is taken as
positive outward.
"""

import numpy as _np
import scipy.sparse as _sp

from .matchObservedLW import _concat, _sum


def matchConstantPressureCore(model, states, schedule, observed,
                              WaterRateWeight=None, OilRateWeight=None,
                              BHPWeight=None, ComputePartials=False,
                              tStep=None, state=None, from_states=True,
                              matchOnlyProducers=False, mismatchSum=True,
                              accumulateWells=None, accumulateTypes=None):
    """Return one mismatch entry per requested report step."""
    dts = _np.asarray(schedule['step']['val'], dtype=float).ravel()
    totTime = float(dts.sum())

    if tStep is None:
        tSteps = _np.arange(dts.size)
        numSteps = dts.size
    else:
        tSteps = _np.atleast_1d(_np.asarray(tStep, dtype=int)).ravel()
        numSteps = 1
        dts = dts[tSteps]

    matchCases = _np.array([1.0, -1.0])
    obj = []

    for step in range(numSteps):
        forces = model.getDrivingForces(schedule['control'][int(tSteps[step])])
        faces = _np.atleast_1d(_np.asarray(
            _get(_get(forces, 'bc'), 'face'), dtype=int)).ravel()

        state_obs = observed[int(tSteps[step])]
        flux_obs = _np.atleast_2d(_np.asarray(
            model.getProps(state_obs, 'flux'), dtype=float))
        fw_obs = flux_obs[faces, 0]
        fo_obs = flux_obs[faces, 1]

        ww, wo = _getWeights(fw_obs, fo_obs, WaterRateWeight, OilRateWeight)

        st = (model.getStateAD(states[int(tSteps[step])], True)
              if (ComputePartials and from_states)
              else (state if ComputePartials else states[int(tSteps[step])]))
        fw, fo = getBoundaryFlux(model, st, forces)

        dt = float(dts[step])
        nmatch = int(_np.count_nonzero(matchCases))
        fac = dt / (totTime * max(nmatch, 1))

        terms = [(ww * matchCases * (fw - fw_obs)) ** 2,
                 (wo * matchCases * (fo - fo_obs)) ** 2]

        if mismatchSum:
            obj.append(fac * sum(_sum(t) for t in terms))
            continue

        mm = [fac * t for t in terms]
        if accumulateTypes is None:
            tmp = mm
        else:
            pt = _np.atleast_1d(_np.asarray(accumulateTypes, dtype=int)).ravel()
            tmp = [0] * int(pt.max())
            for k in range(len(mm)):
                if k < pt.size and pt[k] > 0:
                    tmp[pt[k] - 1] = tmp[pt[k] - 1] + mm[k]
        if accumulateWells is not None:
            pw = _np.atleast_1d(_np.asarray(accumulateWells, dtype=int)).ravel()
            keep = _np.flatnonzero(pw)
            M = _sp.csr_matrix((_np.ones(keep.size), (pw[keep] - 1, keep)),
                               shape=(int(pw.max()), pw.size))
            tmp = [M @ x for x in tmp]
        obj.append(_concat(tmp))

    return obj


def getBoundaryFlux(model, state, drivingForces):
    """Port of the local ``getBoundaryFlux``.

    Returns the water and oil boundary fluxes, signed positive outward: a
    face whose second neighbour is absent points out of the domain.
    """
    try:
        from PRSTCore.ad_core.utils.getBoundaryConditionFluxesAD import \
            getBoundaryConditionFluxesAD as get_boundary_condition_fluxes_ad
    except ImportError as exc:
        raise NotImplementedError(
            'matchConstantPressureCore needs getBoundaryConditionFluxesAD, '
            'which MRST keeps in ad-core/utils and PRSTCore has not ported '
            'yet. Everything else in this module is available.') from exc

    p, s, mob, r, b = model.getProps(
        state, 'PhasePressures', 's', 'Mobility', 'Density', 'ShrinkageFactors')
    sat = _expand_to_list(s)
    rho = _expand_to_list(r)

    bc = _get(drivingForces, 'bc')
    _, _, _, fRes = get_boundary_condition_fluxes_ad(model, p, sat, mob, rho,
                                                     b, bc)

    active = model.getActivePhases()
    fWOG = [None, None, None]
    for slot, value in zip(_np.flatnonzero(_np.asarray(active, dtype=bool)),
                           fRes):
        fWOG[int(slot)] = value

    faces = _np.atleast_1d(_np.asarray(_get(bc, 'face'), dtype=int)).ravel()
    neighbors = _np.asarray(model.G['faces']['neighbors'], dtype=int)
    # -1 marks "no cell" in PRSTCore where MRST uses 0.
    sgn = 1 - 2 * (neighbors[faces, 1] < 0)

    out = []
    for value in fWOG[:2]:
        out.append(None if value is None else value * sgn)
    return out[0], out[1]


def _getWeights(fw, fo, ww, wo):
    """Reciprocal magnitude, or zero when the flux is flat."""
    if ww is None:
        total = float(_np.sum(_np.abs(fw)))
        ww = 0.0 if total == 0 else 1.0 / _np.abs(fw)
    if wo is None:
        total = float(_np.sum(_np.abs(fo)))
        wo = 0.0 if total == 0 else 1.0 / _np.abs(fo)
    return ww, wo


def _expand_to_list(values):
    """Port of ``expandMatrixToCell``: columns of a matrix -> a list."""
    if isinstance(values, (list, tuple)):
        return list(values)
    arr = _np.atleast_2d(_np.asarray(values))
    return [arr[:, i] for i in range(arr.shape[1])]


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)
