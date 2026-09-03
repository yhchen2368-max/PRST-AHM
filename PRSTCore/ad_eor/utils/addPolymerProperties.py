"""Port of MRST ``addPolymerProperties.m``.

Copies the polymer-related fluid fields (``muWMult``, ``dps``, ``rrf``,
``rhoR``, ``adsInx``, ``adsMax``, ``ads``, ``mixPar``, ``cpmax``) from a
separately-parsed ``POLYMER.DATA`` deck's fluid onto an existing fluid dict,
then zeroes ``dps``/``rhoR`` (matching the ``.m`` source, which always
overrides those two after copying).
"""


def addPolymerProperties(fluid, poly_fluid):
    """``poly_fluid``: fluid dict built from a POLYMER.DATA deck (caller's
    responsibility to parse/convert units, mirroring
    ``readEclipseDeck``/``convertDeckUnits``/``initDeckADIFluid`` in the
    ``.m`` source -- deck-keyword parsing for PLYVISC/PLYADS/PLYROCK/PLYMAX/
    TLMIXPAR is not yet implemented in PRSTCore's deck reader)."""
    fields = ['muWMult', 'dps', 'rrf', 'rhoR', 'adsInx', 'adsMax', 'ads',
              'mixPar', 'cpmax']
    fluid = dict(fluid)
    for name in fields:
        fluid[name] = poly_fluid[name]
    fluid['dps'] = 0.0
    fluid['rhoR'] = 0.0
    return fluid
