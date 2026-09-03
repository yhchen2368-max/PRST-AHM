"""The relperm scaling points a saturation table implies -- ``fluid.krPts``.

Port of the ``getPoints`` subfunctions MRST's ``assignSWOF.m``,
``assignSGOF.m``, ``assignSWFN.m``, ``assignSGFN.m``, ``assignSOF2.m``
and ``assignSOF3.m`` each carry (autodiff/ad-props/props). Every one of
those ``assign*`` functions has the same shape: it builds the relperm
callables and, alongside them, records four numbers per saturation
region that say where the curve begins, where the phase starts to move,
where the table ends, and how high the relperm gets.

The four columns are always ``[connate, critical, max-saturation,
max-relperm]``, one row per SATNUM region:

===========  ==================================================
``krPts.w``  from the water column: Swl, Swcr, Swu, KRW
``krPts.ow`` oil against water:     --, Sowcr, 1, KRO
``krPts.og`` oil against gas:       --, Sogcr, 1, KRO
``krPts.g``  from the gas column:   Sgl, Sgcr, Sgu, KRG
===========  ==================================================

which is exactly the set of quantities history matching tunes, and what
``getRelpermScalingPoints`` and ``imposeRelpermScaling`` both expect to
find on the fluid.

Two details are MRST's and easy to get wrong. The *critical* saturation
is read as the **last** table row where the relperm is still zero for a
phase whose curve rises with its own saturation (water, gas), but as the
**first** zero row for oil, whose tabulated curve falls as the other
phase's saturation rises. And ``og``'s critical value subtracts connate
water: MRST tabulates Krog against Sg, so the oil saturation at that row
is ``1 - Sg - Swco``, not ``1 - Sg``.

Nothing here evaluates a relperm. The tables themselves are read by
:func:`PRSTCore.ad_props.relperm_tables.build_swof_sgof_tables`; this
only summarises them.

**There is a second implementation.**
``GenericBlackOilModel._get_relperm_scaling`` computes the same four
sets inline, for the one SATNUM region it is evaluating, because the
flow path needs them before a fluid dict exists. The two are held level
by ``tests/test_kr_points_parity.py`` rather than merged: the inline one
carries a guard this one does not, and removing it would change the flow
path's answers. Change one and that test will tell you about the other.
"""

import numpy as _np

from PRSTCore.ad_props.relperm_tables import (resolve_table_defaults,
                                              split_table_regions)

#: Column meanings of every ``krPts`` row.
CONNATE, CRITICAL, MAXIMUM, MAX_KR = 0, 1, 2, 3


def get_kr_points(props):
    """The scaling points of a deck's PROPS section, per region.

    Returns a dict with whichever of ``w``, ``ow``, ``og``, ``g`` the
    deck supports, each an ``(nregions, 4)`` array. A deck with no
    saturation tables gives an empty dict rather than an error -- an
    analytical fluid has no tabulated points to report.
    """
    props = props or {}
    points = {}

    swof = _regions(props, 'SWOF', 4)
    if swof is not None:
        points['w'] = _np.array([_water_points(t) for t in swof])
        points['ow'] = _np.array([_oil_against_water_points(t) for t in swof])
    else:
        swfn = _regions(props, 'SWFN', 3)
        if swfn is not None:
            points['w'] = _np.array([_water_points(t) for t in swfn])

    # Connate water, needed for the oil-against-gas saturation. MRST reads
    # it from the water points it has just assigned; without a water curve
    # it is zero.
    swco = (points['w'][:, CONNATE] if 'w' in points
            else _np.zeros(1, dtype=float))

    sgof = _regions(props, 'SGOF', 4)
    if sgof is not None:
        points['g'] = _np.array([_water_points(t) for t in sgof])
        points['og'] = _np.array(
            [_oil_against_gas_points(t, _pick(swco, i))
             for i, t in enumerate(sgof)])
    else:
        sgfn = _regions(props, 'SGFN', 3)
        if sgfn is not None:
            points['g'] = _np.array([_water_points(t) for t in sgfn])

    # Family II states the oil curves separately, against So directly.
    sof3 = _regions(props, 'SOF3', 3)
    if sof3 is not None:
        points['ow'] = _np.array([_oil_table_points(t, 1) for t in sof3])
        points['og'] = _np.array([_oil_table_points(t, 2) for t in sof3])
    else:
        sof2 = _regions(props, 'SOF2', 2)
        if sof2 is not None:
            # assignSOF2 assigns a single ``krPts.o``, which
            # imposeRelpermScaling then copies into both ow and og.
            shared = _np.array([_oil_table_points(t, 1) for t in sof2])
            points['ow'] = shared
            points['og'] = shared.copy()

    return points


def _regions(props, name, ncol):
    """One table per saturation region, or None if the keyword is absent
    or malformed. Same acceptance test ``build_swof_sgof_tables`` uses,
    so the two never disagree about what a deck contains."""
    raw = props.get(name, [])
    size = _np.asarray(raw, dtype=object).size
    if size == 0 or size % ncol != 0:
        return None
    tables = split_table_regions(resolve_table_defaults(raw, ncol))
    return tables or None


def _pick(values, index):
    values = _np.atleast_1d(values)
    return float(values[min(index, values.size - 1)])


def _water_points(table):
    """``getPoints`` of assignSWOF/assignSWFN -- and of assignSGOF and
    assignSGFN, which read their own first two columns the same way."""
    table = _np.asarray(table, dtype=float)
    saturation, relperm = table[:, 0], table[:, 1]
    immobile = _np.flatnonzero(relperm == 0.0)
    return _np.array([
        saturation[0],
        # The *last* row still immobile: the curve rises with this phase's
        # own saturation.
        saturation[immobile[-1]] if immobile.size else saturation[0],
        saturation[-1],
        relperm[-1],
    ])


def _oil_against_water_points(swof):
    """``getPoints``'s second output in assignSWOF.

    Connate oil is left at zero, as MRST leaves it, and the maximum
    saturation at 1: the oil curve is stated against ``1 - Sw`` and so
    always reaches unit saturation in principle.
    """
    swof = _np.asarray(swof, dtype=float)
    sw, krow = swof[:, 0], swof[:, 2]
    zero = _np.flatnonzero(krow == 0.0)
    # The *first* zero row: krow falls as Sw rises, so this is where oil
    # stops moving.
    critical = 1.0 - sw[zero[0]] if zero.size else 0.0
    return _np.array([0.0, critical, 1.0, krow[0]])


def _oil_against_gas_points(sgof, swco):
    """``getPoints``'s second output in assignSGOF.

    ``swco`` is subtracted because Krog is tabulated against gas
    saturation in a system that also holds connate water.
    """
    sgof = _np.asarray(sgof, dtype=float)
    sg, krog = sgof[:, 0], sgof[:, 2]
    zero = _np.flatnonzero(krog == 0.0)
    critical = 1.0 - sg[zero[0]] - swco if zero.size else 0.0
    return _np.array([0.0, critical, 1.0, krog[0]])


def _oil_table_points(table, column):
    """``getPoints`` of assignSOF3/assignSOF2, where oil is tabulated
    against its own saturation and therefore rises with it."""
    table = _np.asarray(table, dtype=float)
    so, kro = table[:, 0], table[:, column]
    zero = _np.flatnonzero(kro == 0.0)
    return _np.array([0.0, so[zero[0]] if zero.size else so[0], 1.0, kro[-1]])
