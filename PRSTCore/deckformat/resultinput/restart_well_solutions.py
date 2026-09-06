"""FAHM createWellSol/makeWellSolsConsistent, using simulator record strides."""

from copy import deepcopy

import numpy as np

from .restart_contract import RestartContractError, subset_order


def create_well_solutions(block, G, units):
    from .convert_restart_to_states import _ijk_to_active, _connection_cells, _active_restart_phases
    from .get_restart_well_info import getRestartWellInfo

    ih = np.asarray(block['INTEHEAD']['values']).ravel()
    nw = int(ih[16])
    if nw == 0:
        return []
    records = {name: item['values'] for name, item in block.items()}
    required = {'ZWEL': (27, 1), 'IWEL': (24, 11), 'SWEL': (25, 10),
                'XWEL': (26, 7), 'ICON': (32, 14), 'SCON': (33, 4), 'XCON': (34, 50)}
    for name, (offset, minimum) in required.items():
        stride = int(ih[offset])
        count = nw * stride * (int(ih[17]) if name.endswith('CON') else 1)
        if stride < minimum or name not in records or np.size(records[name]) != count:
            raise RestartContractError(f'{name}: missing/invalid well record dimensions')
    wells, _ = getRestartWellInfo(records)
    dims = np.asarray(G['cartDims'], dtype=int)
    lookup = _ijk_to_active(G, dims)
    phase_names = _active_restart_phases(block)
    phases = np.array([p in phase_names for p in ('WAT', 'OIL', 'GAS')])
    ql, qg, qr = (units[key] / units['time'] for key in ('liqvol_s', 'gasvol_s', 'resvolume'))
    trans = units['viscosity'] * units['resvolume'] / (units['time'] * units['press'])
    output = []
    for well in wells:
        if not well['name'] or well['ncon'] < 0 or well['ncon'] > int(ih[17]):
            raise RestartContractError('Invalid well name or connection count')
        if well['ncon'] == 0:
            for field in ('cdiam', 'cwi', 'cdepth', 'cstat', 'cqr', 'press'):
                well[field] = np.array([])
            well['cdir'] = np.array([], dtype='U1')
        cells = _connection_cells(well['cijk'], dims, lookup)
        cqs = well['cqs'][:, [1, 0, 2]] * np.array([ql, ql, qg])
        sol = dict(name=well['name'], cells=cells, type='', val=np.array([]),
                   r=well['cdiam'] / 2 * units['length'], WI=well['cwi'] * trans,
                   compi=np.array([]), refDepth=well['depth'] * units['length'],
                   dZ=(well['cdepth'] - well['depth']) * units['length'], dir=well['cdir'],
                   sign=0, status=well['stat'], cstatus=well['cstat'] > 0, lims=np.array([]),
                   qWs=well['qWs'] * ql * phases[0], qOs=well['qOs'] * ql * phases[1],
                   qGs=well['qGs'] * qg * phases[2], bhp=well['bhp'] * units['press'],
                   resv=well['qr'] * qr, flux=well['cqr'] * qr,
                   press=well['press'] * units['press'], cqs=cqs[:, phases],
                   components=np.array([]), cdp=np.zeros(len(cells)))
        if well['stat']:
            types = ('orat', 'wrat', 'grat', 'lrat', 'resv', 'unknown', 'bhp')
            vals = (sol['qOs'], sol['qWs'], sol['qGs'], sol['qOs'] + sol['qWs'], sol['resv'], np.nan, sol['bhp'])
            producer = well['type'] == 1
            sol['sign'] = -1 if producer else 1
            if producer:
                sol['compi'] = np.array([0., 1., 0.])[phases]
            elif 2 <= well['type'] <= 4:
                sol['compi'] = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])[well['type'] - 2, phases]
            else:
                sol['compi'] = np.asarray(vals[:3]) / sum(vals[:3])
            cntr = well['cntr']
            if 1 <= cntr <= 7:
                sol['type'] = 'rate' if not producer and cntr <= 3 else types[cntr - 1]
                sol['val'] = vals[cntr - 1]
            else:
                sol['val'] = np.nan
        output.append(sol)
    return output


def make_well_sols_consistent(states):
    """Use the final restart template exactly; reject lossy intersections.

    MRST intentionally initializes not-yet-existing wells/connections as shut.
    These are source-defined inactive records, not substitutes for bad data.
    Final-frame geometric metadata (including press/dZ) is source behavior.
    """
    states = deepcopy(states)
    if not states:
        return states
    template = deepcopy(states[-1]['wellSol'])
    zfields = ('qWs', 'qOs', 'qGs', 'bhp', 'resv')
    efields = ('type', 'val', 'sign', 'compi')
    for well in template:
        nc = len(well['cells'])
        for field in zfields + efields:
            well[field] = 0
        well.update(status=False, cstatus=np.zeros(nc, bool), flux=np.zeros(nc),
                    cqs=np.zeros_like(well['cqs']), r=np.zeros(nc), WI=np.zeros(nc))
    names = [w['name'] for w in template]
    if len(names) != len(set(names)):
        raise RestartContractError('Duplicate final-frame well names')
    for state in states:
        source = state['wellSol']
        state['wellSol'] = deepcopy(template)
        if len({w['name'] for w in source}) != len(source):
            raise RestartContractError('Duplicate restart well names')
        for well in source:
            if well['name'] not in names:
                raise RestartContractError('Restart well absent from final template: ' + well['name'])
            target = state['wellSol'][names.index(well['name'])]
            ia = subset_order(target['cells'], well['cells'], well['name'])
            for field in zfields + efields:
                target[field] = deepcopy(well[field])
            # Source bug: components was not among the copied fields and
            # became the final-frame composition for the complete history.
            target['components'] = deepcopy(well['components'])
            if well['status']:
                target['status'] = True
                for field in ('cstatus', 'flux', 'cqs', 'r', 'WI'):
                    value = np.asarray(well[field])
                    if value.shape[0] != len(ia):
                        raise RestartContractError(f'{well["name"]}.{field}: perforation count mismatch')
                    target[field][ia] = value * well['cstatus'] if field == 'WI' else value
    signs = [w['sign'] for w in states[0]['wellSol']]
    for state in states:
        for i, well in enumerate(state['wellSol']):
            if well['sign'] != 0:
                signs[i] = well['sign']
    for state in reversed(states):
        for i, well in enumerate(state['wellSol']):
            if well['sign'] == 0:
                well['sign'] = signs[i]
            else:
                signs[i] = well['sign']
    return states


def merge_summary(states, summary, *, is_eclipse, program, include_components,
                  connection_quantities=True, well_quantities=True):
    """FAHM summary control/connection/composition additions, matched by name."""
    from .restart_contract import exact_order
    if len(states) != len(summary):
        raise RestartContractError('Summary/restart history count mismatch')
    for state, step in zip(states, summary):
        if not step:
            continue
        wells = state['wellSol']
        order = exact_order([w['name'] for w in wells], [w['name'] for w in step], 'summary wells')
        matched = [step[i] for i in order]
        if well_quantities and (not is_eclipse or program == 300):
            for field in ('type', 'val', 'qWs', 'qOs', 'qGs', 'bhp'):
                values = [w[field] for w in wells]
                empty = all(np.size(v) == 0 or (isinstance(v, str) and v == '') or
                            (not isinstance(v, str) and np.all(np.isnan(v))) for v in values)
                if empty:
                    for well, smry in zip(wells, matched):
                        well[field] = deepcopy(smry[field])
        for well, smry in zip(wells, matched):
            nc = len(well['cells'])
            for field in ('cp', 'cpd') if connection_quantities else ():
                if np.size(smry[field]) not in (0, nc):
                    raise RestartContractError(f'{well["name"]}.{field}: summary connection count mismatch')
            # Summary connection rows are in final restart completion order.
            if connection_quantities and np.size(smry['cp']):
                well['cdp'] = smry['cp'] - well['bhp']
            elif connection_quantities and np.size(smry['cpd']):
                well['cdp'] = state['pressure'][well['cells']] - smry['cpd'] - well['bhp']
            if connection_quantities and np.size(smry['cqs']) and not np.size(well['cqs']):
                if smry['cqs'].shape[0] != nc:
                    raise RestartContractError('Summary cqs connection count mismatch')
                well['cqs'] = smry['cqs'].copy()
            if include_components:
                fields = ('xi', 'yi') if well['sign'] > 0 else ('x', 'y')
                ncpt = state['components'].shape[1]
                if well['sign'] == 0:
                    well['components'] = np.zeros((2, ncpt))
                else:
                    values = [np.asarray(smry[f]) for f in fields]
                    if any(v.size != ncpt for v in values):
                        raise RestartContractError('Missing well component fractions in summary')
                    well['components'] = np.vstack(values)
    return states
