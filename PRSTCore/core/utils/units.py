"""Port of MRST's units module (core/utils/units).

MRST gives each unit its own file holding a function that returns the
unit's value in SI, so an expression reads as it would on paper::

    perm = 100 * milli * darcy
    p    = 200 * barsa
    q    = 300 * meter**3 / day

This is the same set as module-level constants, which reads identically
at the call site and costs nothing to evaluate. Everything is SI, so a
value in some unit becomes SI by *multiplying*, and SI becomes that unit
by *dividing* -- which is exactly what :func:`convert_from` and
:func:`convert_to` do.

The derived ones are computed here rather than written as decimals, so
they cannot drift from their definitions:

* ``darcy`` is not a round number -- it is the permeability at which a
  1 cP fluid flows 1 cm/s under a 1 atm/cm gradient, which works out to
  9.869233e-13 m^2. Since atm is 101325 Pa exactly, so is darcy.
* ``psia`` is pound-force per square inch, and pound-force is a pound
  under standard gravity.
* ``stb`` is 42 US gallons, a gallon is 231 cubic inches.
"""

import numpy as _np

# ------------------------------------------------------------ prefixes --

pico = 1e-12
nano = 1e-9
micro = 1e-6
milli = 1.0 / 1000
centi = 1.0 / 100
deci = 1.0 / 10
kilo = 1e3
mega = 1e6
giga = 1e9

# ---------------------------------------------------------- base units --

meter = 1.0
kilogram = 1.0
second = 1.0
ampere = 1.0
mol = 1.0
Kelvin = 1.0
volt = 1.0
farad = 1.0
Pascal = 1.0
Newton = 1.0

#: A Rankine degree is five ninths of a Kelvin.
Rankine = 5.0 / 9

# ------------------------------------------------------------- length --

inch = 2.54 * centi * meter
ft = 0.3048 * meter

# --------------------------------------------------------------- time --

minute = 60 * second
hour = 60 * minute
day = 24 * hour
#: The mean Gregorian year, not 365 days.
year = 365.2425 * day

# --------------------------------------------------------------- mass --

gram = 1e-3 * kilogram
tonne = 1000 * kilogram
pound = 0.45359237 * kilogram
dalton = 1.66053904020e-27

# ------------------------------------------------------------- volume --

litre = (deci * meter) ** 3
gallon = 231 * inch ** 3
#: Stock-tank barrel: 42 US gallons.
stb = 42 * gallon

# -------------------------------------------------- force and pressure --

#: Standard gravity, used to turn a mass into a force.
gravity_constant = 9.80665 * meter / second ** 2
dyne = 1e-5 * Newton
lbf = pound * gravity_constant

atm = 101325 * Pascal
barsa = 1e5
psia = lbf / inch ** 2

# ---------------------------------------------------- energy and power --

joule = 1 * Newton * meter
watt = joule / second
#: One of several definitions in circulation; this is MRST's.
btu = 1054.3503

# ------------------------------------------------------------ assorted --

poise = 0.1 * Pascal * second
gal = 0.01
ohm = 1 * volt / ampere
siemens = 1 * ampere / volt
#: One over Avogadro's number.
site = (6.00221413e23) ** -1


def _darcy():
    """The Darcy, from its definition rather than as a decimal.

    A 1 cP fluid flowing 1 cm/s through a 1 cm^2 face under a gradient of
    1 atm/cm. Because atm is exactly 101325 Pa, the darcy is exactly
    1e-5 / 1.01325e7 m^2.
    """
    cm = centi * meter
    mu = centi * poise                      # 1 cP
    p_grad = atm / cm                       # Pa/m
    area = cm ** 2                          # m^2
    flow = cm ** 3 / second                 # m^3/s
    vel = flow / area                       # m/s
    return vel * mu / p_grad


darcy = _darcy()

#: The unit permeabilities are given in almost everywhere.
milli_darcy = milli * darcy


# ------------------------------------------------------- conversions --

def _check(unit):
    """Port of the ``isnumeric`` assertion in convertFrom/convertTo.

    numpy calls a string a scalar, so testing for scalarity is not
    enough: a unit *name* passed where a unit *value* belongs has to be
    caught here rather than producing a confusing multiplication error
    further down.
    """
    if isinstance(unit, (str, bytes)):
        raise TypeError("Unsupported 'unit' representation %r -- pass the "
                        "unit's value, not its name" % type(unit).__name__)
    if not isinstance(unit, (int, float, _np.number, _np.ndarray)):
        raise TypeError("Unsupported 'unit' representation %r"
                        % type(unit).__name__)


def convert_from(q, unit):
    """Port of ``convertFrom``: a quantity *in* ``unit``, expressed in SI.

    ``convert_from(100, milli*darcy)`` is 100 mD as m^2.
    """
    _check(unit)
    return _np.asarray(q, dtype=float) * unit


def convert_to(q, unit):
    """Port of ``convertTo``: an SI quantity, expressed *in* ``unit``.

    ``convert_to(p, barsa)`` is a pressure in bar.
    """
    _check(unit)
    return _np.asarray(q, dtype=float) / unit


# ------------------------------------------------------ unit systems --

def get_unit_system(name):
    """Port of ``getUnitSystem`` (core/utils/units).

    The unit each quantity is written in, for one of ECLIPSE's three
    systems. Note ``trans``: transmissibility is not permeability times
    length but ``cP * rm^3 / (day * bar)``, and ``qg`` in FIELD is Mscf
    per day rather than cubic feet.
    """
    name = str(name).lower()
    if name == 'metric':
        return {
            'length': meter, 'time': day, 'press': barsa,
            'viscosity': centi * poise, 'perm': milli * darcy,
            'resvolume': meter ** 3,
            'ql': meter ** 3 / day, 'qg': meter ** 3 / day,
            'qr': meter ** 3 / day, 'bl': 1.0, 'bg': 1.0,
            'temp': Kelvin, 'tempoffset': 273.15,
            'density': kilo * gram / meter ** 3,
            'moledensity': kilo / meter ** 3,
            'trans': centi * poise * meter ** 3 / (day * barsa),
        }
    if name == 'field':
        return {
            'length': ft, 'time': day, 'press': psia,
            'viscosity': centi * poise, 'perm': milli * darcy,
            'resvolume': stb,
            'ql': stb / day, 'qg': 1000 * ft ** 3 / day,
            'qr': stb / day, 'bl': 1.0, 'bg': stb / (1000 * ft ** 3),
            'temp': Rankine, 'tempoffset': 459.67,
            'density': pound / ft ** 3,
            'moledensity': 453.59237 / stb,
            'trans': centi * poise * stb / (day * psia),
        }
    if name == 'lab':
        return {
            'length': centi * meter, 'time': hour, 'press': atm,
            'viscosity': centi * poise, 'perm': milli * darcy,
            'resvolume': (centi * meter) ** 3,
            'ql': (centi * meter) ** 3 / hour,
            'qg': (centi * meter) ** 3 / hour,
            'qr': (centi * meter) ** 3 / hour, 'bl': 1.0, 'bg': 1.0,
            'temp': Kelvin, 'tempoffset': 273.15,
            'density': gram / (centi * meter) ** 3,
            'moledensity': 1.0 / (centi * meter) ** 3,
            'trans': centi * poise * (centi * meter) ** 3 / (hour * atm),
        }
    raise ValueError('Unit %r not supported' % name)
