"""Port of MRST ``getRestartWellInfo.m``
(model-io/deckformat/resultinput/private).

Decodes a restart step's well records. ECLIPSE stores wells as seven
parallel fixed-stride arrays rather than as structures, and the strides
live in INTEHEAD:

    ZWEL  CHAR  nzwel per well          names
    IWEL  INTE  niwel per well          i/j/k, connection count, type,
                                        control mode, open/shut
    SWEL  REAL  nswel per well          reference depth
    XWEL  DOUB  nxwel per well          rates and bhp
    ICON  INTE  nicon per connection    i/j/k, open/shut, direction
    SCON  REAL  nscon per connection    well index, depth, diameter, kh
    XCON  DOUB  nxcon per connection    phase rates, reservoir rate,
                                        pressure

Connections are stored ncwmax per well whether or not that many exist,
so a well's block starts at ``well * nicon * ncwmax`` and only its first
``ncon`` entries are meaningful.

Rates come out of XWEL/XCON negated: ECLIPSE writes production positive,
MRST's convention is negative for production and positive for injection.
"""

import numpy as _np

#: INTEHEAD positions, 0-based (MRST indexes these 1-based).
_IH = {'unit': 2, 'nx': 8, 'ny': 9, 'nz': 10, 'nactive': 11, 'iphs': 14,
       'nwell': 16, 'ncwma': 17, 'nwgmax': 19, 'ngmaxz': 20, 'niwel': 24,
       'nswel': 25, 'nxwel': 26, 'nzwel': 27, 'nicon': 32, 'nscon': 33,
       'nxcon': 34, 'nigrpz': 36, 'iday': 64, 'imon': 65, 'iyear': 66,
       'iprog': 94}


def getINTEHEAD(intehead):
    """Port of ``getINTEHEAD``: the record dimensions, as a dict."""
    values = _np.atleast_1d(_np.asarray(intehead)).ravel()
    return {name: int(values[pos]) if pos < values.size else 0
            for name, pos in _IH.items()}


def getRestartWellInfo(records, intehead=None):
    """Return ``(wells, ih)`` for one restart step.

    ``records`` maps a keyword to its values for this step. A keyword the
    file does not carry leaves its fields empty rather than raising, as
    the MATLAB does -- restart files vary a lot by simulator and version.
    """
    if intehead is None:
        intehead = records.get('INTEHEAD')
    ih = getINTEHEAD(intehead)
    if ih['nwell'] <= 0:
        return [], ih

    wells = [{} for _ in range(ih['nwell'])]
    _zwel(wells, records.get('ZWEL'), ih)
    _iwel(wells, records.get('IWEL'), ih)
    _swel(wells, records.get('SWEL'), ih)
    _xwel(wells, records.get('XWEL'), ih)
    _icon(wells, records.get('ICON'), ih)
    _scon(wells, records.get('SCON'), ih)
    _xcon(wells, records.get('XCON'), ih)
    return wells, ih


# ------------------------------------------------------- per-well data --

def _zwel(wells, zwel, ih):
    """Port of ``getZWEL``: the name is the first of nzwel CHAR slots."""
    for k, well in enumerate(wells):
        if zwel is None:
            well['name'] = None
            continue
        values = _np.atleast_1d(_np.asarray(zwel)).ravel()
        index = k * ih['nzwel']
        well['name'] = str(values[index]).strip() if index < values.size \
            else None


def _iwel(wells, iwel, ih):
    """Port of ``getIWEL``."""
    empty = iwel is None
    values = None if empty else _np.atleast_1d(_np.asarray(iwel)).ravel()
    for k, well in enumerate(wells):
        if empty:
            well.update(ijk=None, ncon=0, ginx=None, type=None, cntr=None,
                        stat=None)
            continue
        d = k * ih['niwel']
        well['ijk'] = values[d:d + 3].astype(int)
        well['ncon'] = int(values[d + 4])
        well['ginx'] = int(values[d + 5])
        well['type'] = int(values[d + 6])
        well['cntr'] = int(values[d + 8])
        well['stat'] = bool(values[d + 10] > 0)


def _swel(wells, swel, ih):
    """Port of ``getSWEL``: only the reference depth is read."""
    values = None if swel is None else \
        _np.atleast_1d(_np.asarray(swel, dtype=float)).ravel()
    for k, well in enumerate(wells):
        well['depth'] = None if values is None \
            else float(values[k * ih['nswel'] + 9])


def _xwel(wells, xwel, ih):
    """Port of ``getXWEL``.

    Rates are negated into MRST's sign convention: negative produced,
    positive injected. The bhp is not.
    """
    values = None if xwel is None else \
        _np.atleast_1d(_np.asarray(xwel, dtype=float)).ravel()
    for k, well in enumerate(wells):
        if values is None:
            well.update(qOs=None, qWs=None, qGs=None, lrat=None, qr=None,
                        bhp=None)
            continue
        d = k * ih['nxwel']
        well['qOs'] = -float(values[d + 0])
        well['qWs'] = -float(values[d + 1])
        well['qGs'] = -float(values[d + 2])
        well['lrat'] = -float(values[d + 3])
        well['qr'] = -float(values[d + 4])
        well['bhp'] = float(values[d + 6])


# ------------------------------------------------- per-connection data --

def _icon(wells, icon, ih):
    """Port of ``getICON``: connection i/j/k, open/shut, and direction."""
    values = None if icon is None else \
        _np.atleast_1d(_np.asarray(icon)).ravel()
    for k, well in enumerate(wells):
        ncon = int(well.get('ncon') or 0)
        if values is None or ncon == 0:
            well.update(cijk=None, cstat=None, cdir=None)
            continue
        base = k * ih['nicon'] * ih['ncwma']
        dc = _np.arange(ncon) * ih['nicon'] + base
        well['cijk'] = _np.column_stack([values[dc + 1], values[dc + 2],
                                         values[dc + 3]]).astype(int)
        well['cstat'] = values[dc + 5].astype(int)
        index = values[dc + 13].astype(int)
        cdir = _np.full(ncon, 'z', dtype='<U1')
        cdir[index == 1] = 'x'
        cdir[index == 2] = 'y'
        well['cdir'] = cdir


def _scon(wells, scon, ih):
    """Port of ``getSCON``: well index, depth, diameter, kh."""
    values = None if scon is None else \
        _np.atleast_1d(_np.asarray(scon, dtype=float)).ravel()
    for k, well in enumerate(wells):
        ncon = int(well.get('ncon') or 0)
        if values is None or ncon == 0:
            well.update(cwi=None, cdepth=None, cdiam=None, ckh=None)
            continue
        base = k * ih['nscon'] * ih['ncwma']
        dc = _np.arange(ncon) * ih['nscon'] + base
        well['cwi'] = values[dc + 0]
        well['cdepth'] = values[dc + 1]
        well['cdiam'] = values[dc + 2]
        well['ckh'] = values[dc + 3]


def _xcon(wells, xcon, ih):
    """Port of ``getXCON``: per-connection phase rates and pressure."""
    values = None if xcon is None else \
        _np.atleast_1d(_np.asarray(xcon, dtype=float)).ravel()
    for k, well in enumerate(wells):
        ncon = int(well.get('ncon') or 0)
        if values is None or ncon == 0:
            well.update(cqs=_np.zeros((0, 3)), cqr=_np.zeros(0), press=None)
            continue
        base = k * ih['nxcon'] * ih['ncwma']
        dc = _np.arange(ncon) * ih['nxcon'] + base
        well['cqs'] = -_np.column_stack([values[dc + 0], values[dc + 1],
                                         values[dc + 2]])
        well['cqr'] = -values[dc + 49]
        well['press'] = values[dc + 34]
