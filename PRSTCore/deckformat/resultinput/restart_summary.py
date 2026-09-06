"""FAHM summary supplements: controls, connection pressure and compositions."""

from pathlib import Path

import numpy as np

from .read_eclipse_summary import read_eclipse_summary, convert_summary_to_well_sols
from .restart_contract import RestartContractError
from ..unit_conversion_factors import unit_conversion_factors


def read_restart_summary(prefix, unit):
    spec, data = Path(prefix + '.SMSPEC'), Path(prefix + '.UNSMRY')
    if not spec.exists() and not data.exists():
        return [], np.array([])
    if not spec.exists() or not data.exists():
        raise RestartContractError('Incomplete SMSPEC/UNSMRY file pair')
    smry = read_eclipse_summary(prefix)
    wells, time = convert_summary_to_well_sols(smry, unit=unit)
    units = unit_conversion_factors(unit)
    return supplement_summary(smry, wells, units), time


def supplement_summary(smry, wells, units):
    names = np.asarray(smry['WGNAMES'])
    keywords = np.asarray(smry['KEYWORDS'])
    data = np.asarray(smry['data'])
    if data.shape[1] != len(wells):
        raise RestartContractError('Summary time/data count mismatch')
    ql = units['liqvol_s'] / units['time']
    qg = units['gasvol_s'] / units['time']

    def rows(name, keyword):
        return data[(names == name) & (keywords == keyword)]

    for k, step in enumerate(wells):
        for well in step:
            name = well['name']
            well.update(type=np.array([]), val=0.0,
                        status=abs(well['qWs'] + well['qOs'] + well['qGs']) > 0)
            fields = {'CPR': ('cp', units['press']), 'CDRD': ('cpd', units['press']),
                      'CTFAC': ('ctfac', units['viscosity'] * units['resvolume'] / (units['time'] * units['press']))}
            for keyword, (field, factor) in fields.items():
                well[field] = rows(name, keyword)[:, k].copy() * factor
            cqs = [rows(name, key)[:, k] * -factor for key, factor in
                   (('CWFR', ql), ('COFR', ql), ('CGFR', qg))]
            well['cqs'] = np.column_stack(cqs) if all(v.size for v in cqs) else np.array([])
            control = rows(name, 'WMCTL')
            if control.size:
                if control.shape[0] != 1:
                    raise RestartContractError('WMCTL must have one row per well')
                cno = int(control[0, k])
                if 1 <= cno <= 7:
                    qr = -sum(np.sum(rows(name, key)[:, k]) for key in ('WVPR', 'WVIR')) * ql
                    vals = (well['qOs'], well['qWs'], well['qGs'], well['qOs'] + well['qWs'], qr, np.nan, well['bhp'])
                    well['type'] = ('orat', 'wrat', 'grat', 'lrat', 'resv', 'thp', 'bhp')[cno - 1]
                    well['val'] = vals[cno - 1]
            for prefix, field in (('WXMF', 'x'), ('WYMF', 'y'), ('WZMF', 'z'),
                                   ('WXMFI', 'xi'), ('WYMFI', 'yi'), ('WZMFI', 'zi')):
                fractions = []
                for component in range(1, 1001):
                    value = rows(name, f'{prefix}_{component}')
                    if not value.size:
                        break
                    if value.shape[0] != 1:
                        raise RestartContractError('Well component summary must have one row')
                    fractions.append(value[0, k])
                well[field] = np.asarray(fractions)
    return wells
