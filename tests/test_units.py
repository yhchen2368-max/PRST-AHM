"""Tests for the units module and the deck unit-conversion factors.

A wrong conversion factor never raises -- it rescales a whole deck and
the result still looks like a reservoir. So these check the values
against their physical definitions and against independently known
figures, not against the constants the code itself declares.
"""

import numpy as np
import pytest

from PRSTCore.core.utils import units as U
from PRSTCore.deckformat.unit_conversion_factors import \
    unit_conversion_factors


# ------------------------------------------------------ unit constants --

def test_si_base_units_are_one():
    assert (U.meter, U.kilogram, U.second, U.Kelvin) == (1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize('value, expected', [
    (U.pico, 1e-12), (U.nano, 1e-9), (U.micro, 1e-6), (U.milli, 1e-3),
    (U.centi, 1e-2), (U.deci, 1e-1), (U.kilo, 1e3), (U.mega, 1e6),
    (U.giga, 1e9),
])
def test_prefixes(value, expected):
    assert value == pytest.approx(expected)


def test_an_inch_is_exactly_2_54_cm():
    assert U.inch == pytest.approx(0.0254, rel=0, abs=1e-15)


def test_a_foot_is_twelve_inches():
    assert U.ft == pytest.approx(12 * U.inch)


def test_a_day_is_86400_seconds():
    assert U.day == 86400.0


def test_a_year_is_the_mean_gregorian_one():
    """365.2425 days, not 365 -- the difference is 3.5 hours a year."""
    assert U.year == pytest.approx(365.2425 * 86400.0)


def test_a_darcy_follows_from_its_definition():
    """1 cP flowing 1 cm/s under 1 atm/cm. Because atm is exactly
    101325 Pa, the darcy is exactly 1e-5/1.01325e7 m^2."""
    assert U.darcy == pytest.approx(1e-5 / (101325.0 / 0.01), rel=1e-15)


def test_a_millidarcy_is_the_familiar_figure():
    assert U.milli * U.darcy == pytest.approx(9.869233e-16, rel=1e-6)


def test_psi_is_pound_force_over_a_square_inch():
    assert U.psia == pytest.approx(6894.757293168361, rel=1e-12)


def test_an_atmosphere_is_exact():
    assert U.atm == 101325.0


def test_a_bar_is_1e5_pascal():
    assert U.barsa == 1e5


def test_a_poise_is_a_tenth_of_a_pascal_second():
    assert U.poise == pytest.approx(0.1)
    assert U.centi * U.poise == pytest.approx(1e-3)


def test_a_barrel_is_42_gallons():
    assert U.stb == pytest.approx(42 * U.gallon)
    assert U.stb == pytest.approx(0.158987294928, rel=1e-9)


def test_a_litre_is_a_cubic_decimetre():
    assert U.litre == pytest.approx(1e-3)


def test_a_rankine_degree_is_five_ninths_of_a_kelvin():
    assert U.Rankine == pytest.approx(5.0 / 9)


def test_a_pound_is_the_international_avoirdupois_one():
    assert U.pound == 0.45359237


# ------------------------------------------------------- convert_from/to --

def test_convert_from_scales_into_si():
    assert U.convert_from(100, U.milli * U.darcy) == \
        pytest.approx(9.869233e-14, rel=1e-6)


def test_convert_to_scales_out_of_si():
    assert U.convert_to(200e5, U.barsa) == pytest.approx(200.0)


def test_the_two_are_inverses():
    value = 137.0
    assert U.convert_to(U.convert_from(value, U.psia), U.psia) == \
        pytest.approx(value)


def test_conversion_works_on_arrays():
    out = U.convert_from(np.array([1.0, 2.0]), U.day)
    assert list(out) == [86400.0, 172800.0]


def test_a_non_numeric_unit_is_rejected():
    with pytest.raises(TypeError, match='Unsupported'):
        U.convert_from(1.0, 'barsa')


# ---------------------------------------------------------- get_unit_system --

def test_metric_transmissibility_is_not_permeability_times_length():
    """cP*m^3/(day*bar) is 1.16e-13; perm*length would be 1e-25 and every
    connection would be twelve orders of magnitude too tight."""
    u = U.get_unit_system('metric')
    assert u['trans'] == pytest.approx(1e-3 / (86400.0 * 1e5), rel=1e-12)


def test_field_gas_rate_is_mscf_per_day():
    """Thousand cubic feet, not cubic feet -- a factor of 1000."""
    u = U.get_unit_system('field')
    assert u['qg'] == pytest.approx(1000 * U.ft ** 3 / U.day, rel=1e-12)


def test_field_liquid_rate_is_stb_per_day():
    u = U.get_unit_system('field')
    assert u['ql'] == pytest.approx(0.158987294928 / 86400.0, rel=1e-9)


def test_an_unknown_system_is_rejected():
    with pytest.raises(ValueError, match='not supported'):
        U.get_unit_system('cgs')


# ------------------------------------------------ unit_conversion_factors --

_SYSTEMS = ('METRIC', 'FIELD', 'LAB', 'PVT_M', 'SI')


@pytest.mark.parametrize('system', _SYSTEMS)
def test_every_system_carries_the_full_mrst_field_set(system):
    u = unit_conversion_factors(system)
    for field in ('length', 'area', 'invarea', 'time', 'density', 'press',
                  'temp', 'tempoffset', 'mol', 'mass', 'concentr', 'compr',
                  'viscosity', 'surf_tension', 'jsurftens', 'perm',
                  'liqvol_s', 'liqvol_r', 'gasvol_s', 'gasvol_r', 'volume',
                  'trans', 'rockcond', 'volumheatcapacity',
                  'massheatcapacity'):
        assert field in u, '%s missing %s' % (system, field)


@pytest.mark.parametrize('system', _SYSTEMS)
def test_the_older_prstcore_names_still_resolve(system):
    """Kept as aliases so switching to MRST's vocabulary breaks nothing."""
    u = unit_conversion_factors(system)
    assert u['compressibility'] == u['compr']
    assert u['gas_volume'] == u['gasvol_s']
    assert u['transmissibility'] == u['trans']
    assert u['resvolume'] == u['liqvol_r']


def test_si_factors_are_all_one():
    u = unit_conversion_factors('SI')
    for key, value in u.items():
        if isinstance(value, float) and key != 'tempoffset':
            assert value == 1.0, key


def test_metric_pressure_is_bar():
    assert unit_conversion_factors('METRIC')['press'] == 1e5


def test_field_pressure_is_psi():
    assert unit_conversion_factors('FIELD')['press'] == \
        pytest.approx(6894.757293168361, rel=1e-12)


def test_lab_volume_is_a_cubic_centimetre():
    """Not a litre. PRSTCore had 1e-3 here, which is a thousand times
    too large."""
    assert unit_conversion_factors('LAB')['volume'] == pytest.approx(1e-6)


def test_lab_density_converts_grams_per_cc_to_si():
    """1 g/cm^3 is 1000 kg/m^3, so the factor is 1000 -- PRSTCore had
    1e-3, out by a factor of a million."""
    assert unit_conversion_factors('LAB')['density'] == pytest.approx(1e3)


def test_pvt_m_differs_from_metric_only_in_pressure():
    """PVT-M is METRIC with atmospheres instead of bar."""
    metric = unit_conversion_factors('METRIC')
    pvtm = unit_conversion_factors('PVT_M')
    assert pvtm['press'] == pytest.approx(101325.0)
    assert pvtm['length'] == metric['length']
    assert pvtm['compr'] == pytest.approx(1 / 101325.0)


def test_an_unknown_system_is_rejected():
    with pytest.raises(ValueError, match='Unknown unit system'):
        unit_conversion_factors('CGS')


# ------------------------------------------------ non-SI output (ratios) --

def test_converting_a_system_to_itself_gives_unity():
    u = unit_conversion_factors('FIELD', 'FIELD')
    for key in ('length', 'press', 'perm', 'trans', 'liqvol_s'):
        assert u[key] == pytest.approx(1.0), key


def test_metric_to_field_length_is_metres_per_foot():
    u = unit_conversion_factors('METRIC', 'FIELD')
    assert u['length'] == pytest.approx(1 / 0.3048, rel=1e-12)


def test_metric_to_field_pressure_is_bar_per_psi():
    u = unit_conversion_factors('METRIC', 'FIELD')
    assert u['press'] == pytest.approx(1e5 / 6894.757293168361, rel=1e-12)


def test_a_round_trip_between_systems_is_the_identity():
    """This is what writeDeck relies on: read METRIC into SI, write it
    back out in METRIC, and the numbers must be the ones that came in."""
    to_field = unit_conversion_factors('METRIC', 'FIELD')
    back = unit_conversion_factors('FIELD', 'METRIC')
    for key in ('length', 'press', 'perm', 'liqvol_s', 'trans', 'density'):
        assert to_field[key] * back[key] == pytest.approx(1.0, rel=1e-12), key


def test_the_output_system_is_recorded():
    u = unit_conversion_factors('FIELD', 'METRIC')
    assert u['unit_in'] == 'FIELD' and u['unit_out'] == 'METRIC'


def test_temperature_offset_is_carried_across_systems():
    """Rankine and Kelvin have different zero points as well as different
    sizes, so the offset cannot just be a ratio."""
    u = unit_conversion_factors('FIELD', 'METRIC')
    assert u['tempoffset'] != 0.0
    assert np.isfinite(u['tempoffset'])
