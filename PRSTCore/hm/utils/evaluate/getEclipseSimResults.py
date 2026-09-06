"""FAHM external-result contract. Local MRST dev/utils/evaluate is the oracle.

Returned setup, states and wellSols own their mutable data. PRST indices are
zero based. Source defects/strict rejection policy: Stage 11 registry.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from PRSTCore.deckformat.resultinput.restart_contract import (
    RestartContractError, exact_order, report_indices, time_vector)

_PERF_FIELDS = ('WI', 'dZ', 'dir', 'r', 'rR', 'cells', 'cstatus', 'cqs',
                'cdp', 'flux', 'press', 'cell_origin')


def getEclipseSimResults(directory, filename, setup, useMinisteps=False):
    """Read actual binary results, then enforce the FAHM report contract."""
    from PRSTCore.deckformat.resultinput.convert_restart_to_states import convert_restart_to_states
    from PRSTCore.deckformat.resultinput.process_eclipse_restart_spec import process_eclipse_restart_spec

    owned = deepcopy(setup)
    model = owned['model']
    if hasattr(model, 'validateModel'):
        model = model.validateModel()
        owned['model'] = model
    prefix = str(Path(directory) / filename)
    rsspec, _ = process_eclipse_restart_spec(prefix, 'all')
    states, _ = convert_restart_to_states(
        prefix, model.G, neighbors=model.operators['N'], restart_info=rsspec,
        split_wells_on_sign_change=False, remove_closed_wells=False,
        remove_crossflow=False, include_well_sols=True, include_aquifers=True,
        include_fluxes=False, include_mobilities=False,
        include_components=_is_compositional(model),
        set_to_closed_tol=1e-8 / 86400.0)
    return align_eclipse_states(states, owned, useMinisteps=useMinisteps)


def _is_compositional(model):
    declared = 'COMPS' in (getattr(model, 'inputdata', None) or {}).get('RUNSPEC', {})
    eos = getattr(model, 'EOSModel', None)
    if declared and eos is None:
        raise RestartContractError('Compositional results require an explicit EOSModel; black-oil fallback is forbidden')
    return eos is not None


def align_eclipse_states(raw_states, setup, useMinisteps=False):
    """Alignment stage, exposed for deterministic replay of reader fixtures."""
    setup, raw_states = deepcopy(setup), deepcopy(raw_states)
    model, schedule = setup['model'], setup['schedule']
    dt = np.asarray(schedule['step']['val'], dtype=float).ravel(order='F')
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise RestartContractError('Schedule dt must be finite and positive')
    control = np.asarray(schedule['step']['control']).ravel(order='F')
    controls = schedule['control']
    if control.size != dt.size or np.any(control != control.astype(int)) or np.any(control < 0) or np.any(control >= len(controls)):
        raise RestartContractError('Schedule step/control count or index mismatch')
    control = control.astype(int)
    T_rep = np.r_[0.0, np.cumsum(dt)]
    T_sim = time_vector([state['time'] for state in raw_states], 'T_sim')
    nc = int(model.G['cells']['num'])
    phase_shape = None
    for state in raw_states:
        if np.shape(state.get('pressure')) != (nc,):
            raise RestartContractError('Restart pressure must have one row per active cell')
        shape = np.shape(state.get('s'))
        if len(shape) != 2 or shape[0] != nc or shape[1] not in (1,2,3):
            raise RestartContractError('Restart saturation shape mismatch')
        if phase_shape is not None and shape != phase_shape:
            raise RestartContractError('Phase count changed across restart states')
        phase_shape = shape
        if not np.all(np.isfinite(state['pressure'])) or not np.all(np.isfinite(state['s'])):
            raise RestartContractError('Nonfinite restart pressure/saturation')
    selected = report_indices(T_sim, T_rep)
    if selected[0] != 0 or T_sim[0] != 0:
        raise RestartContractError('The first restart state must be state0 at time zero')
    if useMinisteps:
        # Report ordinal -> schedule.control, not ordinal -> control array.
        ends = selected[1:]
        report = np.searchsorted(ends, np.arange(1, len(raw_states)), side='left')
        schedule['step']['control'] = control[report].copy()
        schedule['step']['val'] = np.diff(T_sim)
    else:
        raw_states = [raw_states[i] for i in selected]
    if _is_compositional(model):
        from .restart_state_reconstruction import reconstruct_compositional
        for state in raw_states:
            reconstruct_compositional(state, model.EOSModel)
    state0, states = raw_states[0], raw_states[1:]
    if len(states) != len(schedule['step']['val']):
        raise RestartContractError('State count must equal schedule step count')
    if controls:
        initial_control = control[0] if control.size else 0
        state0['wellSol'] = _align_wells(state0.get('wellSol', []), controls[initial_control]['W'])
    setup['state0'] = state0
    well_sols = []
    for i, state in enumerate(states):
        W = controls[int(schedule['step']['control'][i])]['W']
        state['wellSol'] = _align_wells(state.get('wellSol', []), W)
        for well, sol in zip(W, state['wellSol']):
            for field in ('refDepth', 'dZ'):
                if field not in sol:
                    raise RestartContractError(f'{sol["name"]}: missing {field}')
                well[field] = deepcopy(sol[field])
            if np.size(sol.get('type', [])) == 0 or sol.get('type') == '':
                sol['type'], sol['val'] = deepcopy(well['type']), deepcopy(well['val'])
            if _is_compositional(model) and well['sign'] > 0:
                compi = np.asarray(well['compi']).ravel(order='F')
                if model.water:
                    compi = compi[1:]
                components = np.asarray(sol['components'])
                if not components.size and np.any(compi > 0):
                    raise RestartContractError('Missing injector component fractions')
                if components.size:
                    if components.ndim != 2 or components.shape[0] != compi.size:
                        raise RestartContractError('Injector components/compi shape mismatch')
                    components = components[compi > 0, :]
                    if components.size:
                        well['components'] = components.copy()
        well_sols.append(deepcopy(state['wellSol']))
    aquifer = getattr(model, 'AquiferModel', None)
    present = ['aquiferSol' in state for state in states]
    if present and any(present) and not all(present):
        raise RestartContractError('Partial aquiferSol history is not permitted')
    if aquifer is not None and states and not any(present):
        from .restart_state_reconstruction import reconstruct_aquifers
        reconstruct_aquifers(setup, states)
    return states, well_sols, setup


def _align_wells(wells, W):
    names, available = [w['name'] for w in W], [w['name'] for w in wells]
    if len(set(names)) != len(names) or len(set(available)) != len(available):
        raise RestartContractError('Duplicate well names are ambiguous')
    order = exact_order(names, available, 'well names')
    ordered = [wells[i] for i in order]
    perfs = [exact_order(np.asarray(w['cells']).ravel(order='F'),
                         np.asarray(s['cells']).ravel(order='F'), w['name'])
             for w, s in zip(W, ordered)]
    return _sortWellSol(ordered, perfs)


def _alignment(states, names, cells):
    for state in states:
        if state.get('wellSol'):
            order = exact_order(names, [w['name'] for w in state['wellSol']])
            perfs = [exact_order(c, state['wellSol'][i]['cells']) for c, i in zip(cells, order)]
            return order, perfs
    return None, None


def _stable_order(wanted, available):
    return exact_order(wanted, available)


def _sortWellSol(wellSol, cell_order):
    if len(wellSol) != len(cell_order):
        raise RestartContractError('Well/perforation permutation count mismatch')
    out = deepcopy(wellSol)
    for well, order in zip(out, cell_order):
        order = np.asarray(order, dtype=int)
        nc = np.size(well['cells'])
        if sorted(order.tolist()) != list(range(nc)):
            raise RestartContractError('Perforation permutation must be a bijection')
        for field in _PERF_FIELDS:
            if field not in well or np.size(well[field]) == 0:
                continue
            array = np.asarray(well[field])
            if array.ndim == 0 or array.shape[0] != nc:
                raise RestartContractError(f'{well["name"]}.{field}: expected {nc} perforation rows')
            well[field] = array[order].copy()
    return out


def _sim2rep(T_sim, T_rep):
    return report_indices(T_sim, T_rep).tolist()
