"""Port of MRST ``getEclipseSimResults.m`` (mrst-2026a/hm/utils/evaluate).

Reads an ECLIPSE run's restart output back into MRST states and well
solutions, and aligns them with the MRST setup.

The alignment is the substance of this function: ECLIPSE may report wells
and their perforations in a different order than the schedule holds them,
so both are permuted back to the schedule's order before anything else
reads them. It also copies the reference depth and dZ ECLIPSE computed
into the schedule, and fills in the control type/value for wells whose
restart record omits them.

.. note::
   This reads output that only a licensed ECLIPSE run produces. The
   restart parsing itself is PRSTCore's
   :mod:`PRSTCore.deckformat.resultinput`; there is no ECLIPSE here to
   generate the input, so this module is ported for completeness and its
   file-reading path cannot be exercised without such a run. The
   reordering logic below is independent of that and is tested directly.

The compositional branch (``ThreePhaseCompositionalModel``: recovering
x/y/K and the liquid/vapour compressibility factors through the EOS) is
not ported -- PRSTCore has no compositional model for it to act on.
"""

import numpy as _np

from PRSTCore.hm.utils.controlIndex import control_index


def getEclipseSimResults(directory, filename, setup, useMinisteps=False):
    """Return ``(states, wellSols, setup)``."""
    import os

    from PRSTCore.deckformat.resultinput.convert_restart_to_states import \
        convert_restart_to_states
    from PRSTCore.deckformat.resultinput.process_eclipse_restart_spec import \
        process_eclipse_restart_spec

    model = setup['model']
    names = [w['name'] for w in setup['schedule']['control'][0]['W']]
    cells = [_np.atleast_1d(_np.asarray(w['cells'], dtype=int)).ravel()
             for w in setup['schedule']['control'][0]['W']]

    prefix = os.path.join(str(directory), str(filename))
    rsspec = process_eclipse_restart_spec(prefix, 'all')
    # ``convertRestartToStates``'s option names, in PRSTCore's spelling.
    # ``includeMobilities`` has no counterpart: the port never computes
    # them, so there is nothing to switch off.
    # ``convertRestartToStates`` returns [states, restartInfo]; MATLAB
    # takes the first output only.
    states, _ = convert_restart_to_states(
        prefix, model.G, neighbors=model.operators['N'], restart_info=rsspec,
        split_wells_on_sign_change=False, remove_closed_wells=False,
        remove_crossflow=False, include_well_sols=True,
        include_aquifers=True, include_fluxes=False,
        set_to_closed_tol=1e-8 / 86400.0)

    dt = _np.asarray(setup['schedule']['step']['val'], dtype=float).ravel()
    T_rep = _np.concatenate([[0.0], _np.cumsum(dt)])
    T_sim = _np.asarray([s['time'] for s in states], dtype=float)

    if useMinisteps:
        ctrl = _np.asarray(
            [int(_np.flatnonzero(t <= T_rep)[0]) for t in T_sim])
        setup['schedule']['step']['control'] = ctrl[1:] - 1
        setup['schedule']['step']['val'] = _np.diff(T_sim)
    else:
        states = [states[i] for i in _sim2rep(T_sim, T_rep)]

    state0, states = states[0], states[1:]

    well_order, cell_order = _alignment(states, names, cells)
    if well_order is not None:
        state0['wellSol'] = [state0['wellSol'][i] for i in well_order]
    if cell_order is not None:
        state0['wellSol'] = _sortWellSol(state0['wellSol'], cell_order)
    setup['state0'] = state0

    wellSols = [None] * len(states)
    for i, st in enumerate(states):
        if not st.get('wellSol'):
            continue
        if well_order is not None:
            st['wellSol'] = [st['wellSol'][j] for j in well_order]
        if cell_order is not None:
            st['wellSol'] = _sortWellSol(st['wellSol'], cell_order)

        cno = control_index(setup['schedule']['step'], i,
                            len(setup['schedule']['control']))
        W = setup['schedule']['control'][cno]['W']
        for j, sol in enumerate(st['wellSol']):
            if j < len(W):
                # ECLIPSE's own reference depth and dZ win.
                W[j]['refDepth'] = sol.get('refDepth', W[j].get('refDepth'))
                W[j]['dZ'] = sol.get('dZ', W[j].get('dZ'))
                # A restart record may omit the control; take the
                # schedule's.
                if not sol.get('type'):
                    sol['type'] = W[j].get('type')
                    sol['val'] = W[j].get('val')
        wellSols[i] = st['wellSol']

    return states, wellSols, setup


def _alignment(states, names, cells):
    """The permutations that put ECLIPSE's wells back in schedule order.

    Returns ``(well_order, cell_order)``, either of which is ``None`` when
    that level already agrees.
    """
    for st in states:
        if not st.get('wellSol'):
            continue
        names_sol = [w['name'] for w in st['wellSol']]
        well_order = None
        if names_sol != list(names):
            well_order = _stable_order(names, names_sol)
        ordered = ([st['wellSol'][i] for i in well_order] if well_order
                   else st['wellSol'])
        cells_sol = [_np.atleast_1d(_np.asarray(w['cells'], dtype=int)).ravel()
                     for w in ordered]
        cell_order = None
        if not all(_np.array_equal(a, b) for a, b in zip(cells, cells_sol)):
            cell_order = [_stable_order(a, list(b))
                          for a, b in zip(cells, cells_sol)]
        return well_order, cell_order
    return None, None


def _stable_order(wanted, available):
    """Indices into ``available`` that put it in ``wanted``'s order."""
    lookup = {v: i for i, v in enumerate(available)}
    return [lookup[v] for v in wanted if v in lookup]


def _sortWellSol(wellSol, cell_order):
    """Reorder each well's per-perforation arrays."""
    out = []
    for w, order in zip(wellSol, cell_order):
        w = dict(w)
        order = _np.asarray(order, dtype=int)
        for key, value in w.items():
            arr = _np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == order.size and key != 'name':
                w[key] = arr[order]
        out.append(w)
    return out


def _sim2rep(T_sim, T_rep):
    """The simulated step nearest each report time."""
    return [int(_np.argmin(_np.abs(T_sim - t))) for t in T_rep]
