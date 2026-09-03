"""Impose relative permeability end-point scaling on a model.

Port of MRST ``imposeRelpermScaling.m`` (autodiff/ad-props).

Follows MRST-0's version rather than 2026a's. The two differ in which
keywords they accept: 2026a takes eleven drainage endpoints, MRST-0
takes thirty-eight -- the drainage set plus the residual-endpoint
relative permeabilities (KRWR, KRORW, KRORG, KRGR), the imbibition set
(``I*``), and the miscible/surfactant set (``SS*``, ``SKR*``).

That difference is not cosmetic here: ``trainSurfactantFlood`` tunes
``sswl``, ``sswcr``, ``sswu``, ``ssowcr``, ``skrw`` and ``skro``, none of
which 2026a's list contains -- with that version those six parameters
would be dropped without a word and the surfactant training would
silently tune nothing but the rock.

MRST-0 also merges into an existing ``rock.krscale`` instead of replacing
it, so scaling can be imposed in more than one call, and returns
immediately when given nothing to do.
"""

import warnings as _warnings

import numpy as np

#: Drainage endpoints, and where each lands in the krscale table.
_DRAINAGE = {"SWL", "SWCR", "SWU", "SGL", "SGCR", "SGU", "SOWCR", "SOGCR",
             "KRW", "KRO", "KRG", "KRWR", "KRORW", "KRORG", "KRGR"}

#: The same set measured on the imbibition curve.
_IMBIBITION = {"I" + name for name in _DRAINAGE}

#: The saturation functions that apply at full surfactant concentration.
_MISCIBLE = {"SSWL", "SSWCR", "SSWU", "SSGL", "SSGCR", "SSOWCR", "SSOGCR",
             "SKRW", "SKRO"}

VALID_SCALERS = _DRAINAGE | _IMBIBITION | _MISCIBLE

#: Which table a keyword belongs to.
_TABLES = (("drainage", _DRAINAGE), ("imbibition", _IMBIBITION),
           ("miscible", _MISCIBLE))


def _get(model, name, default=None):
    """Read a model field.

    MRST's model is a class instance and the fields are properties; the
    ones ``selectModelFromDeck`` builds here are too, while much of
    PRSTCore's own test and CGNet code passes a plain dict. Both shapes
    reach this function, so both are read the same way.
    """
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)


def _set(model, name, value):
    if isinstance(model, dict):
        model[name] = value
    else:
        setattr(model, name, value)


def impose_relperm_scaling(model, n_points=2, **scaling_kwargs):
    """Impose rel-perm endpoint scaling for a model not associated with a deck.

    Parameters
    ----------
    model : dict or model object
        Model with ``fluid.krPts``.
    n_points : int
        2 or 3-point scaling.
    **scaling_kwargs : dict
        Scaling values named from :data:`VALID_SCALERS`. Each value may be
        a scalar or an array of length nc.

    Returns
    -------
    dict or model object
        The same model, with ``rock.krscale`` and ``inputdata`` set.
    """
    if not scaling_kwargs:
        return model

    fluid = _get(model, "fluid") or {}
    if "krPts" not in fluid:
        raise ValueError("To impose rel-perm scaling, fluid must contain 'krPts'")

    assert n_points in (2, 3), "Only 2- or 3-point scaling is supported"

    nc = _get(model, "G")["cells"]["num"]

    scale, rejected = {}, []
    for kw, val in scaling_kwargs.items():
        kw_upper = kw.upper()
        if kw_upper not in VALID_SCALERS:
            rejected.append(kw_upper)
            continue
        val_arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
        if len(val_arr) == 1:
            val_arr = np.full(nc, val_arr[0])
        elif len(val_arr) != nc:
            raise ValueError(f"Scaling values for {kw} do not match grid cells")
        scale[kw_upper] = val_arr

    if rejected:
        # MRST calls warnProblem(prob) here with `prob` never assigned,
        # so this branch raises there rather than warning. Naming the
        # keywords is the evident intent, and dropping them silently --
        # which is what the current PRSTCore port did -- is worse than
        # either.
        _warnings.warn('Unsupported relative permeability scaling '
                       'keyword(s) ignored: %s' % ', '.join(sorted(rejected)),
                       RuntimeWarning)

    # Setup krscale in rock, merging into whatever is already there.
    rock = _get(model, "rock")
    if rock is None:
        rock = {}
        _set(model, "rock", rock)
    existing = rock.get("krscale")
    if existing is None:
        rock["krscale"] = _init_relperm_scaling(scale, nc)
    else:
        rock["krscale"] = _merge_relperm_scaling(existing, scale, nc)

    # The fake deck FlowPropertyFunctions needs to switch endpoint scaling
    # on.  MRST-0 *merges* it into whatever deck the model already carries,
    # adding ENDSCALE and SCALECRS only where they are absent; 2026a
    # replaces the deck outright.  The difference is not cosmetic: a model
    # built by selectModelFromDeck keeps its deck in ``inputdata``, and
    # PRSTCore's residual assembly reads SWOF/SGOF back out of it.
    # Replacing it leaves the model with no saturation tables at all --
    # "Deck AD assembly requires SWOF/SGOF tables" the moment an adjoint
    # tries to assemble.
    endscale_str = "YES" if n_points == 3 else "NO"
    endscale = ["NODIR", "REVERS", 1, 20, 0]
    inputdata = _get(model, "inputdata")
    if not inputdata:
        _set(model, "inputdata", {
            "RUNSPEC": {"ENDSCALE": endscale},
            "PROPS": {"SCALECRS": endscale_str},
            "GRID": {},
            "SOLUTION": {},
        })
    else:
        runspec = inputdata.setdefault("RUNSPEC", {})
        runspec.setdefault("ENDSCALE", endscale)
        props = inputdata.setdefault("PROPS", {})
        props.setdefault("SCALECRS", endscale_str)

    # Handle krO renaming for oil-water / oil-gas systems
    fluid = dict(fluid)
    has_oil = _get(model, "oil", True)
    has_water = _get(model, "water", True)
    has_gas = _get(model, "gas", False)

    if has_oil and "krO" in fluid:
        if not has_gas and has_water and "krOW" not in fluid:
            fluid["krOW"] = fluid["krO"]
            if "o" in fluid.get("krPts", {}):
                fluid["krPts"]["ow"] = fluid["krPts"]["o"]
        if not has_water and has_gas and "krOG" not in fluid:
            # Deliberate divergence: both MRST trees write
            # ``fluid.krOG = fluid.krG`` here, which puts the *gas* curve
            # where the oil-gas curve belongs -- a copy-paste slip in the
            # branch above it. Reproducing it would give a two-phase
            # gas-oil model with endpoint scaling the wrong oil relperm,
            # silently. Flagged rather than matched.
            fluid["krOG"] = fluid["krO"]
            if "o" in fluid.get("krPts", {}):
                fluid["krPts"]["og"] = fluid["krPts"]["o"]
        if ("krOW" in fluid or not has_water) and ("krOG" in fluid or not has_gas):
            del fluid["krO"]
    _set(model, "fluid", fluid)

    return model


#: Port of ``getScalerMap``: which phase table and column each endpoint
#: occupies. Columns are [L, CR, U, KM, KR-at-residual]. KRO lands in
#: two tables at once, since one oil curve serves both oil-water and
#: oil-gas.
_PHASES = ("w", "ow", "g", "og")

_COLUMNS = {
    "SWL": ((0, 0),), "SWCR": ((0, 1),), "SWU": ((0, 2),),
    "SGL": ((2, 0),), "SGCR": ((2, 1),), "SGU": ((2, 2),),
    "SOWCR": ((1, 1),), "SOGCR": ((3, 1),),
    "KRW": ((0, 3),), "KRG": ((2, 3),), "KRO": ((1, 3), (3, 3)),
    "KRWR": ((0, 4),), "KRORW": ((1, 4),),
    "KRORG": ((3, 4),), "KRGR": ((3, 4),),
}


def _table_of(keyword):
    """The krscale table a keyword belongs to, and its base name."""
    for table, names in _TABLES:
        if keyword in names:
            if table == "imbibition":
                return table, keyword[1:]
            if table == "miscible":
                # SSWL -> SWL, SKRW -> KRW.
                return table, keyword[1:]
            return table, keyword
    return None, None


def _assign(tables, table, base, values, nc):
    """Place one endpoint into every column it occupies."""
    for phase_index, column in _COLUMNS.get(base, ()):
        phase = _PHASES[phase_index]
        target = tables.setdefault(table, {})
        if phase not in target:
            target[phase] = np.full((nc, 5), np.nan)
        target[phase][:, column] = values


def _init_relperm_scaling(scale, nc):
    """Build a fresh krscale structure from the given endpoints."""
    # initRelpermScaling always creates all three branches and all four
    # phase tables, each as nc-by-5 NaNs, even if only drainage endpoints
    # were supplied.  The NaNs mean "fall back to fluid.krPts".
    tables = {
        branch: {phase: np.full((nc, 5), np.nan)
                 for phase in _PHASES}
        for branch in ('drainage', 'imbibition', 'miscible')
    }
    for keyword, values in scale.items():
        table, base = _table_of(keyword)
        if table is not None:
            _assign(tables, table, base, values, nc)
    return tables


def _merge_relperm_scaling(existing, scale, nc):
    """Port of MRST-0's merge branch: overwrite only the columns given.

    Imposing scaling twice must not discard what the first call set, so
    each endpoint is written into the existing table rather than a new
    one being built from scratch.
    """
    tables = _init_relperm_scaling({}, nc)
    for name, table in (existing or {}).items():
        target = tables.setdefault(name, {})
        for phase, values in (table or {}).items():
            target[phase] = np.array(values, dtype=float, copy=True)
    for keyword, values in scale.items():
        table, base = _table_of(keyword)
        if table is not None:
            _assign(tables, table, base, values, nc)
    return tables
