"""Unit conversion factors between ECLIPSE unit systems and SI.

Port of MRST ``unitConversionFactors.m`` (model-io/deckformat).

Every factor is *derived* from :mod:`PRSTCore.core.utils.units` rather
than written as a decimal, so it cannot drift from the definition it
stands for. That matters more here than anywhere else in the port: a
wrong factor does not raise, it silently rescales a whole deck, and the
result still looks like a reservoir. Two have already been got wrong in
this codebase -- transmissibility written as permeability times length
(twelve orders of magnitude out), and non-SI output refused outright.

A factor converts *from* the named system *to* SI by multiplication.
Converting between two non-SI systems is the ratio of their factors,
which is what ``generalConversionFactors`` does.

MRST's field names are authoritative; the longer names PRSTCore used
before (``compressibility``, ``gas_volume``, ``transmissibility``) are
kept as aliases so existing callers are unaffected.
"""

import numpy as np

from PRSTCore.core.utils.units import (Kelvin, Newton, Pascal, Rankine, atm,
                                       barsa, btu, centi, darcy, day, dyne, ft,
                                       giga, gram, hour, inch, joule, kilo,
                                       kilogram, lbf, meter, milli, poise,
                                       pound, stb)

#: The older PRSTCore names, and the MRST name each stands for. Kept so
#: that changing to MRST's vocabulary does not break existing callers.
_ALIASES = {
    'compressibility': 'compr',
    'gas_volume': 'gasvol_s',
    'transmissibility': 'trans',
    # Not an MRST field, but PRSTCore has used it since before this port
    # for reservoir volume -- pore volume, aquifer volume, reservoir
    # rates. It is liqvol_r under another name.
    'resvolume': 'liqvol_r',
}


def _system(name):
    """Port of ``unit_system``: the unit each quantity is written in."""
    if name == 'METRIC':
        return {
            'length': meter, 'area': meter ** 2, 'invarea': meter ** -2,
            'time': day, 'density': kilogram / meter ** 3, 'press': barsa,
            'temp': Kelvin, 'tempoffset': 273.15, 'mol': kilo,
            'mass': kilogram, 'concentr': kilogram / meter ** 3,
            'compr': 1 / barsa, 'viscosity': centi * poise,
            'surf_tension': Newton / meter,
            'jsurftens': dyne / (centi * meter), 'perm': milli * darcy,
            'liqvol_s': meter ** 3, 'liqvol_r': meter ** 3,
            'gasvol_s': meter ** 3, 'gasvol_r': meter ** 3,
            'volume': meter ** 3,
            'trans': centi * poise * meter ** 3 / (day * barsa),
            'rockcond': kilo * joule / (meter * day * Kelvin),
            'volumheatcapacity': kilo * joule / (meter ** 3 * Kelvin),
            'massheatcapacity': kilo * joule / (kilogram * Kelvin),
            'ymodule': giga,
        }
    if name == 'FIELD':
        return {
            'length': ft, 'area': ft ** 2, 'invarea': ft ** -2,
            'time': day, 'density': pound / ft ** 3, 'press': psia_(),
            'temp': Rankine, 'tempoffset': 459.67, 'mol': 453.59237,
            'mass': pound, 'concentr': pound / stb, 'compr': 1 / psia_(),
            'viscosity': centi * poise, 'surf_tension': lbf / inch,
            'jsurftens': dyne / (centi * meter), 'perm': milli * darcy,
            'liqvol_s': stb, 'liqvol_r': stb,
            # Mscf, not cubic feet.
            'gasvol_s': 1000 * ft ** 3, 'gasvol_r': stb,
            'volume': ft ** 3,
            'trans': centi * poise * stb / (day * psia_()),
            'rockcond': btu / (ft * day * Rankine),
            'volumheatcapacity': btu / (ft ** 3 * Rankine),
            'massheatcapacity': btu / (pound * Rankine),
        }
    if name == 'LAB':
        cm = centi * meter
        return {
            'length': cm, 'area': cm ** 2, 'invarea': cm ** -2,
            'time': hour, 'density': gram / cm ** 3, 'press': atm,
            'temp': Kelvin, 'tempoffset': 273.15, 'mol': 1.0,
            'mass': gram * kilo, 'concentr': gram / cm ** 3,
            'compr': 1 / atm, 'viscosity': centi * poise,
            'surf_tension': dyne / cm, 'jsurftens': dyne / cm,
            'perm': milli * darcy,
            'liqvol_s': cm ** 3, 'liqvol_r': cm ** 3,
            'gasvol_s': cm ** 3, 'gasvol_r': cm ** 3, 'volume': cm ** 3,
            'trans': centi * poise * cm ** 3 / (hour * atm),
            'rockcond': joule / (cm * hour * Kelvin),
            'volumheatcapacity': joule / (cm ** 3 * Kelvin),
            'massheatcapacity': joule / (gram * Kelvin),
        }
    if name == 'PVT_M':
        return {
            'length': meter, 'area': meter ** 2, 'invarea': meter ** -2,
            'time': day, 'density': kilogram / meter ** 3, 'press': atm,
            'temp': Kelvin, 'tempoffset': 273.15, 'mol': kilo,
            'mass': kilogram, 'concentr': kilogram / meter ** 3,
            'compr': 1 / atm, 'viscosity': centi * poise,
            'surf_tension': Newton / meter,
            'jsurftens': dyne / (centi * meter), 'perm': milli * darcy,
            'liqvol_s': meter ** 3, 'liqvol_r': meter ** 3,
            'gasvol_s': meter ** 3, 'gasvol_r': meter ** 3,
            'volume': meter ** 3,
            'trans': centi * poise * meter ** 3 / (day * atm),
            'rockcond': kilo * joule / (meter * day * Kelvin),
            'volumheatcapacity': kilo * joule / (meter ** 3 * Kelvin),
            'massheatcapacity': kilo * joule / (kilogram * Kelvin),
        }
    if name == 'SI':
        keys = ('length', 'area', 'invarea', 'time', 'density', 'press',
                'temp', 'mol', 'mass', 'concentr', 'compr', 'viscosity',
                'surf_tension', 'jsurftens', 'perm', 'liqvol_s', 'liqvol_r',
                'gasvol_s', 'gasvol_r', 'volume', 'trans', 'rockcond',
                'volumheatcapacity', 'massheatcapacity', 'ymodule')
        u = {k: 1.0 for k in keys}
        u['tempoffset'] = 0.0
        return u
    raise ValueError('Unknown unit system: %s' % name)


def psia_():
    """psia, kept as a call so the import list stays flat."""
    return lbf / inch ** 2


def unit_conversion_factors(input_unit="METRIC", output_unit="SI"):
    """Return conversion factors between two unit systems.

    Parameters
    ----------
    input_unit : str
        'METRIC', 'FIELD', 'LAB', 'PVT_M' or 'SI'.
    output_unit : str
        Where to convert to; 'SI' by default.

    Returns
    -------
    dict
        A factor per quantity. Multiplying a value in ``input_unit`` by
        its factor gives the value in ``output_unit``.
    """
    input_name = str(input_unit).upper()
    output_name = str(output_unit).upper()

    u = dict(_system(input_name))

    if output_name != 'SI':
        # Port of ``generalConversionFactors``: every factor is a plain
        # multiplier into SI, so going between two non-SI systems is the
        # ratio. This is what writeDeck needs to take an SI deck back out
        # in METRIC or FIELD.
        out = _system(output_name)
        ratio = dict(out)
        for key, value in u.items():
            if key == 'tempoffset':
                continue
            if out.get(key):
                ratio[key] = value / out[key]
        if 'temp' in ratio and ratio['temp']:
            ratio['tempoffset'] = u.get('tempoffset', 0.0) \
                - out.get('tempoffset', 0.0) / ratio['temp']
        u = ratio

    for alias, name in _ALIASES.items():
        u[alias] = u[name]
    u['unit_in'] = input_name
    u['unit_out'] = output_name
    return u


def convert_from(val, factor):
    """Convert value from input unit to SI using factor."""
    val = np.asarray(val, dtype=float)
    return val * float(factor)
