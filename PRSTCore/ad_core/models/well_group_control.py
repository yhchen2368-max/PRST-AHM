"""Group control: distributing a group rate target across its wells.

Port of ``GenericFacilityModel.updateWellGroupControl`` /
``getWellLimits`` and ``SimpleWell.getWellPotential`` from MRST-0
(autodiff/ad-core/models/facilities). MRST 2026a has none of these; the
group-control machinery is one of the things that makes MRST-0 the
version a real history match runs on.

A GCONPROD/GCONINJE target says what a *group* must produce, not what
each well in it must. The allocation:

1. share the target out pro rata by each well's **potential** -- the
   rate it would deliver flowing against its own bhp limit;
2. any well whose share exceeds one of its own limits is **held** at
   that limit and its control type switched to the limiting one;
3. the remainder of the group target is redistributed among the wells
   not yet held;
4. repeat until nothing more is held.

Which is to say the group target is met by the wells that can take it,
and a well is never asked to exceed its own constraints to make the
group's number.

**Index mismatch in MRST-0, corrected here.** Its version builds
``lims`` from ``W(act)`` and ``type`` as ``repmat(..., numel(act), 1)``
-- both indexed by position within the group -- but then reads them as
``lims(w,:)`` and writes ``type(w)`` with ``w`` a *global* well number,
while the final loop reads ``type{j}`` by position again. The two agree
only when the group's wells happen to be wells 1..N, which is true of
the FIELD group and generally false of a named one: there it reads the
wrong entries, or raises when the global index exceeds the group size.
This port indexes by position throughout, which is what the surrounding
code plainly intends.
"""

import numpy as _np

#: Control types, in the column order ``get_well_potential`` returns and
#: ``get_well_limits`` fills.
CONTROL_TYPES = ('wrat', 'orat', 'grat', 'lrat', 'rate', 'resv')


def get_well_limits(wells):
    """Port of ``getWellLimits``: one row per well, six columns.

    A limit the well does not declare is infinite -- it never binds.
    """
    lims = _np.full((len(wells), len(CONTROL_TYPES)), _np.inf)
    for i, well in enumerate(wells):
        limits = _get(well, 'lims') or {}
        for j, name in enumerate(CONTROL_TYPES):
            value = _get(limits, name)
            if value is not None:
                lims[i, j] = float(value)
    return lims


def get_well_potential(W, wellSol, bw, mob, pw, rs=None, rv=None,
                       phase_index=(0, 1, 2)):
    """Port of ``SimpleWell.getWellPotential``.

    The rate the well would deliver flowing against its own bhp limit,
    summed over perforations, in the six-column layout::

        [qw, qo, qg, qw+qo, qw+qo+qg, sum of reservoir rates]

    Dissolved gas and vaporised oil are folded into the surface rates
    they belong to, so ``qo`` includes oil that arrives as vapour.
    """
    w, o, g = phase_index
    wi = _np.asarray(_get(W, 'WI'), dtype=float).ravel()
    cstatus = _get(W, 'cstatus')
    if cstatus is not None:
        wi = wi * _np.asarray(cstatus, dtype=float).ravel()

    bhp_lim = float(_get(_get(W, 'lims') or {}, 'bhp') or 0.0)
    cdp = _np.asarray(_get(wellSol, 'cdp', 0.0), dtype=float).ravel()
    dp = _np.asarray(pw, dtype=float).ravel() - (bhp_lim + cdp)

    q = _np.asarray(mob, dtype=float) * (-wi * dp).reshape(-1, 1)
    bw = _np.asarray(bw, dtype=float)

    qw = q[:, w] * bw[:, w]
    qo = q[:, o] * bw[:, o]
    if rv is not None and _np.any(rv):
        qo = qo + _np.asarray(rv, dtype=float).ravel() * q[:, g] * bw[:, g]
    qg = q[:, g] * bw[:, g]
    if rs is not None and _np.any(rs):
        qg = qg + _np.asarray(rs, dtype=float).ravel() * q[:, o] * bw[:, o]

    columns = _np.column_stack([qw, qo, qg, qw + qo, qw + qo + qg,
                                q.sum(axis=1)])
    return columns.sum(axis=0)


def update_well_group_control(wellSol, groups, wells, q_p,
                              active=None, max_rounds=100):
    """Port of ``updateWellGroupControl``. Returns the updated wellSol.

    ``groups`` is the driving forces' ``G``: entries with ``name``,
    ``type`` and ``val``. A group named ``FIELD`` covers every active
    well; any other covers the wells whose ``group`` matches its name.
    """
    wellSol = list(wellSol)
    if not groups:
        return wellSol

    if active is None:
        active = [i for i, s in enumerate(wellSol)
                  if bool(_get(s, 'status', True))]
    active = list(active)
    q_p = _np.atleast_2d(_np.asarray(q_p, dtype=float))

    for group in groups:
        name = str(_get(group, 'name', ''))
        gtype = str(_get(group, 'type', ''))
        if gtype not in CONTROL_TYPES:
            continue
        ref = CONTROL_TYPES.index(gtype)

        act = _group_wells(name, wells, active)
        if not act:
            continue

        lims = get_well_limits([wells[i] for i in act])
        _allocate(wellSol, act, lims, q_p, ref, float(_get(group, 'val', 0.0)),
                  gtype, max_rounds)
    return wellSol


def _group_wells(name, wells, active):
    """The active wells this group covers, in their global numbering."""
    if name.upper() == 'FIELD':
        return list(active)
    return [i for i in active
            if str(_get(wells[i], 'group', '')).upper() == name.upper()]


def _allocate(wellSol, act, lims, q_p, ref, val, gtype, max_rounds):
    """The hold-and-redistribute loop, indexed by position within act."""
    n = len(act)
    # Rows of q_p for this group's wells, in group order.
    potential = q_p[act, :]
    types = [gtype] * n

    share = _pro_rata(potential, ref, val, list(range(n)))
    free = list(range(n))

    for _ in range(max_rounds):
        held = []
        for j in list(free):
            with _np.errstate(divide='ignore', invalid='ignore'):
                ratio = lims[j, :] / share[j, :]
            binding = _np.isfinite(ratio) & (ratio < 1)
            if not _np.any(binding):
                continue
            # Scale the whole row by the tightest binding ratio, so the
            # well sits exactly on the limit it hit first.
            worst = float(_np.max(ratio[binding]))
            share[j, :] = share[j, :] * worst
            val = val - share[j, ref]
            types[j] = CONTROL_TYPES[int(_np.argmax(_np.where(binding, ratio,
                                                              -_np.inf)))]
            held.append(j)
        if not held:
            break
        free = [j for j in free if j not in held]
        if not free:
            break
        share = _pro_rata(potential, ref, val, free, share)

    for j, well_index in enumerate(act):
        sol = wellSol[well_index]
        column = CONTROL_TYPES.index(types[j])
        _set(sol, 'type', types[j])
        _set(sol, 'val', float(share[j, column]))


def _pro_rata(potential, ref, val, rows, share=None):
    """Share ``val`` among ``rows`` in proportion to their potential."""
    share = _np.zeros_like(potential) if share is None else share.copy()
    total = float(_np.sum(potential[rows, ref]))
    if total == 0:
        # Nothing to share out on; leave these wells where they are
        # rather than dividing by zero.
        return share
    scale = val / total
    for j in rows:
        share[j, :] = scale * potential[j, :]
    return share


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    return default if value is None else value


def _set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)
