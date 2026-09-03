import numpy as _np


def _ensure_2d_array(tbl):
    """Convert common parsed table formats into a 2D numpy array when possible.

    MRST decks may represent tables as lists, nested lists, or numpy arrays.
    This helper attempts to coerce to a numeric 2D array; if rows have
    differing lengths, returns a list of 1D numpy arrays.
    """
    if tbl is None:
        return None
    # If already numpy array
    if isinstance(tbl, _np.ndarray):
        if tbl.ndim == 1:
            # try to reshape into (-1, ncols) if possible
            return tbl.astype(float)
        return tbl.astype(float)
    # If list-like
    try:
        rows = list(tbl)
    except Exception:
        return None
    # If every row is scalar, return 1D array
    if all(not hasattr(r, '__len__') or isinstance(r, (str, bytes)) for r in rows):
        try:
            return _np.asarray(rows, dtype=float)
        except Exception:
            return _np.asarray(rows, dtype=object)
    # If rows are list-like, try to coerce to 2D array
    row_lens = [len(r) if hasattr(r, '__len__') else 1 for r in rows]
    if min(row_lens) == max(row_lens):
        try:
            return _np.asarray([_np.asarray(r, dtype=float) for r in rows], dtype=float)
        except Exception:
            return _np.asarray([_np.asarray(r) for r in rows], dtype=object)
    # Ragged rows -> return list of arrays
    out = []
    for r in rows:
        try:
            out.append(_np.asarray(r, dtype=float))
        except Exception:
            out.append(_np.asarray(r, dtype=object))
    return out


from PRSTCore.ad_core.initialization.pvt_tables import parse_pvt_table


def init_deck_adi_fluid(deck):
    """Parse PVT-related tables from `deck.PROPS` into a fluid dict.

    This function aims to mirror MRST `initDeckADIFluid` in structure by
    extracting common black-oil tables such as `PVTO`, `PVTG`, `PVTW`, and
    `PVDG`. Tables are coerced to numeric arrays when possible; ragged
    tables are returned as lists of arrays. The result is a plain dict
    used by downstream model constructors.
    """
    fluid = {}
    # deck may be an object with attributes or a dict
    props = getattr(deck, 'PROPS', None) if hasattr(deck, 'PROPS') else deck.get('PROPS', {})
    if props is None:
        props = {}

    # Helper to read and normalize tables
    def read_table(name):
        val = props.get(name)
        return _ensure_2d_array(val)

    # Black-oil relevant tables
    raw_pvto = read_table('PVTO')
    raw_pvtg = read_table('PVTG')
    raw_pvtw = read_table('PVTW')
    raw_pvdg = read_table('PVDG')
    raw_pvtog = read_table('PVTOG')
    raw_pvt = read_table('PVT')

    # Parse into PVTable wrappers when possible
    fluid['PVTO'] = parse_pvt_table(raw_pvto)
    fluid['PVTG'] = parse_pvt_table(raw_pvtg)
    fluid['PVTW'] = parse_pvt_table(raw_pvtw)
    fluid['PVDG'] = parse_pvt_table(raw_pvdg)
    fluid['PVTOG'] = parse_pvt_table(raw_pvtog)
    fluid['PVTEXTRA'] = parse_pvt_table(raw_pvt)

    # Additional useful tables
    fluid['ROCKTAB'] = read_table('ROCKTAB')
    fluid['SWATINIT'] = read_table('SWATINIT')

    # Flags for convenience
    fluid['has_pvto'] = fluid['PVTO'] is not None
    fluid['has_pvtg'] = fluid['PVTG'] is not None
    fluid['has_pvtw'] = fluid['PVTW'] is not None

    # Carry raw props for downstream code that expects original format
    fluid['raw_props'] = props

    # The scaling points the saturation tables imply, which MRST's
    # assignSWOF/assignSGOF/... record alongside the relperm callables they
    # build. Nothing in the flow path reads them; imposeRelpermScaling and
    # getRelpermScalingPoints -- and so history matching's endpoint
    # parameters -- have nothing to work from without them.
    try:
        from PRSTCore.ad_props.kr_points import get_kr_points
        fluid['krPts'] = get_kr_points(props)
    except Exception:
        fluid['krPts'] = {}

    # Keep the deck-driven black-oil evaluator independent of the legacy
    # PVTO convenience wrapper.  A wrapper parse failure must not silently
    # disable all PVT tables for the actual model.
    try:
        from PRSTCore.ad_core.initialization.pvt_tables import PVTOHandler
        fluid['pvto_obj'] = PVTOHandler(fluid['PVTO']) if fluid['PVTO'] is not None else None
    except Exception:
        fluid['pvto_obj'] = None
    try:
        from PRSTCore.ad_core.initialization.pvt_tables import DeckBlackOilPVT
        fluid['blackoil_pvt'] = DeckBlackOilPVT(props)
    except Exception:
        fluid['blackoil_pvt'] = None

    return fluid

