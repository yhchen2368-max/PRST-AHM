"""Tunable model parameters -- port of MRST ``ModelParameter.m``
(autodiff/optimization/utils).

A parameter is a named handle on somewhere in the simulation setup:
which of ``model`` / ``schedule`` / ``state0`` it lives in
(``belongs_to``) and the path within it (``location``). Everything else
-- reading the value, writing it back, scaling it into the optimiser's
unit box, scaling the gradient back out -- follows from those two.

That indirection is what lets the adjoint seed a parameter as an AD
variable and read ``dR/dtheta`` straight out of the assembled Jacobian:
``set_parameter`` puts the AD object where the model expects a number,
and the assembly carries the derivative through whatever the parameter
feeds. It is also why permeability works at all -- see
:func:`set_permeability_fun`, which rebuilds the transmissibilities from
the permeability rather than assuming how one scales the other.
"""

import copy
import warnings

import numpy as np

#: Where each named parameter lives, ported from ``setupByName``.
#: ``(belongs_to, location)``; entries needing a custom writer are
#: completed in :meth:`ModelParameter._setup_by_name`.
_LOCATIONS = {
    'transmissibility': ('model', ('operators', 'T')),
    'porevolume':       ('model', ('operators', 'pv')),
    'conntrans':        ('schedule', ('control', 'W', 'WI')),
    'pressure':         ('state0', ('pressure',)),
}

#: ``getScalerMap``: which relperm curve and column each endpoint is.
#: KRO names two, because the oil maximum is shared between the
#: water-oil and gas-oil curves.
_SCALER_PHASES = ('w', 'ow', 'g', 'og')
_SCALER_MAP = {
    'swl': ((0, 0),), 'swcr': ((0, 1),), 'swu': ((0, 2),),
    'sgl': ((2, 0),), 'sgcr': ((2, 1),), 'sgu': ((2, 2),),
    'sowcr': ((1, 1),), 'sogcr': ((3, 1),),
    'krw': ((0, 3),), 'krg': ((2, 3),), 'kro': ((1, 3), (3, 3)),
}

#: The well-control parameters, which the adjoint treats specially
#: because the control equations carry non-differentiable logic.
WELL_CONTROL_TYPES = ('bhp', 'rate', 'wrat', 'orat', 'grat')


def get_scaler_map():
    """Port of ``getScalerMap``: ``(phases, {keyword: ((phase, col), ...)})``."""
    return _SCALER_PHASES, {k.upper(): v for k, v in _SCALER_MAP.items()}


class ModelParameter:
    def __init__(self, name, n_param, scaling="linear", box_lims=None,
                 lumping=None, subset=None, relative_limits=None,
                 setup=None, belongs_to=None, location=None,
                 type="value", reference_value=None, control_steps=None):
        self.name = name
        self.n_param = int(n_param)
        self.scaling = scaling
        self.lumping = lumping
        self.subset = subset
        self.relative_limits = relative_limits

        # ``setupByName``'s outputs, and the custom accessors some
        # parameters need.
        self.type = type
        self.reference_value = reference_value
        self.control_type = None
        self.control_steps = control_steps
        self.extra_locations = []
        self.getfun = None
        self.setfun = None
        self.belongs_to = belongs_to
        self.location = tuple(location) if location else None
        if self.belongs_to is None or self.location is None:
            self._setup_by_name(setup)

        # ``if isempty(p.getfun), p.getfun = @getfield; end``.  Callers --
        # updateDeckFromModelParameter among them -- invoke it directly
        # rather than testing for emptiness first, so it has to be there.
        # ``setfun`` keeps ``None`` for "the default writer": PRSTCore's
        # custom writers take ``(target, param, value)`` rather than
        # MATLAB's ``(obj, location..., value)``, and only
        # :meth:`set_parameter_value` ever reads it.
        if self.getfun is None:
            self.getfun = _default_getfun

        if box_lims is None:
            if scaling == "log":
                # Sensible default for log-scaled parameters (positive)
                self.box_lims = np.tile([1e-6, 1e6], (self.n_param, 1))
            else:
                self.box_lims = np.tile([0.0, 1.0], (self.n_param, 1))
        else:
            box_lims = np.asarray(box_lims, dtype=float)
            if box_lims.ndim == 1:
                if box_lims.size == 2:
                    box_lims = np.tile(box_lims, (self.n_param, 1))
                else:
                    raise ValueError("box_lims must have length 2 or shape (n_param,2)")
            if box_lims.shape == (1, 2):
                box_lims = np.tile(box_lims, (self.n_param, 1))
            if box_lims.shape != (self.n_param, 2):
                raise ValueError(f"box_lims must have shape ({self.n_param}, 2), got {box_lims.shape}")
            self.box_lims = box_lims

    # ------------------------------------------------ setupByName --

    def _setup_by_name(self, setup):
        """Port of ``setupByName``: derive where this parameter lives.

        A name with no entry here is left unlocated rather than guessed
        at -- MRST errors, and a wrong location would write the tuned
        value somewhere the model never reads.
        """
        key = str(self.name).lower()

        if key in _LOCATIONS:
            self.belongs_to, self.location = _LOCATIONS[key]
            return

        if key in ('permx', 'permy', 'permz'):
            column = 'xyz'.index(key[-1])
            self.belongs_to = 'model'
            self.location = ('rock', 'perm', (slice(None), column))
            self.setfun = set_permeability_fun
            return

        if key in ('sw', 'sg'):
            self.belongs_to = 'state0'
            phase = 0 if key.endswith('w') else 2
            self.location = ('s', (slice(None), phase))
            return

        if key in _SCALER_MAP:
            self.belongs_to = 'model'
            spots = _SCALER_MAP[key]
            phase, column = spots[0]
            self.location = ('rock', 'krscale', 'drainage',
                             _SCALER_PHASES[phase], (slice(None), column))
            self.extra_locations = [
                ('rock', 'krscale', 'drainage', _SCALER_PHASES[p],
                 (slice(None), c)) for p, c in spots[1:]]
            self.setfun = set_relperm_scalers_fun
            return

        if key in WELL_CONTROL_TYPES:
            self.belongs_to = 'schedule'
            self.location = ('control', 'W', 'val')
            self.control_type = key
            return

        raise ValueError('No default setup for parameter: %s' % self.name)

    # ------------------------------------------------- get and set --

    def get_parameter(self, setup):
        """Port of ``getParameter``: the value the optimiser tunes.

        A ``multiplier`` parameter reports its factor rather than the
        underlying field, so 1.0 means untouched.
        """
        if self.type != 'multiplier':
            return self.get_parameter_value(setup)
        value = self.get_parameter_value(setup, collapse=False)
        return collapse_lumps(np.asarray(value, dtype=float)
                              / self.reference_value, self.lumping)

    def set_parameter(self, setup, value):
        """Port of ``setParameter``: write a tuned value back.

        ``value`` may carry derivatives -- that is the whole point. The
        writers below are chosen so an AD object survives the trip.
        """
        if self.type != 'multiplier':
            return self.set_parameter_value(setup, value)
        value = expand_lumps(value, self.lumping) * self.reference_value
        return self.set_parameter_value(setup, value, expand=False)

    def get_parameter_value(self, setup, collapse=True):
        """Port of ``getParameterValue``."""
        if self._is_control_parameter():
            control = _controls(setup)[self._first_control_step()]
            return _get_control_value(control, self.control_type, collapse,
                                      self.lumping)
        # MRST always reads through ``p.getfun``; its default is a plain
        # getfield, so the two agree except where a parameter installs a
        # reader of its own.
        target = setup[self.belongs_to]
        value = self.getfun(target, *self.location)
        if self.subset is not None:
            value = np.asarray(value)[self.subset]
        return collapse_lumps(value, self.lumping) if collapse else value

    def set_parameter_value(self, setup, value, expand=True):
        """Port of ``setParameterValue``."""
        if expand:
            value = expand_lumps(value, self.lumping)

        if self._is_control_parameter():
            for step in (self.control_steps or [0]):
                _set_control_value(_controls(setup)[step],
                                   self.control_type, value)
            return setup

        if self.subset is not None:
            whole = _getfield(setup[self.belongs_to], self.location)
            value = _set_subset(whole, value, self.subset)

        target = setup[self.belongs_to]
        if self.setfun is not None:
            setup[self.belongs_to] = self.setfun(target, self, value)
        else:
            _setfield(target, self.location, value)
        return setup

    def collapse_gradient(self, g):
        """Port of ``collapseGradient``: sum each lump."""
        if self.lumping is None:
            return g
        lumping = np.asarray(self.lumping, dtype=int).ravel()
        g = np.asarray(g, dtype=float).ravel()
        if self.subset is not None:
            g = g[self.subset]
        return np.bincount(lumping, weights=g)

    # --------------------------------------------- MRST's spellings --
    #
    # The hm evaluate and optimizer layers are ports of MATLAB that call
    # ``p.nParam``, ``p.setParameter``, ``p.scaleGradient`` and so on --
    # MRST's own names for these, kept deliberately so those modules read
    # like the files they came from.  Exposing the aliases here is one
    # change; renaming the call sites would be five, and would make each
    # of them diverge from its MATLAB.

    @property
    def nParam(self):
        return self.n_param

    @property
    def boxLims(self):
        return self.box_lims

    @property
    def relativeLimits(self):
        return self.relative_limits

    @property
    def belongsTo(self):
        return self.belongs_to

    @property
    def referenceValue(self):
        return self.reference_value

    @property
    def controlSteps(self):
        return self.control_steps

    @property
    def controlType(self):
        return self.control_type

    def getParameter(self, setup):
        return self.get_parameter(setup)

    def setParameter(self, setup, value):
        return self.set_parameter(setup, value)

    def getParameterValue(self, setup, collapse=True):
        return self.get_parameter_value(setup, collapse)

    def setParameterValue(self, setup, value, expand=True):
        return self.set_parameter_value(setup, value, expand)

    def scaleGradient(self, grad, pval):
        return self.scale_gradient(grad, pval)

    def collapseGradient(self, g):
        return self.collapse_gradient(g)

    def _is_control_parameter(self):
        return (self.belongs_to == 'schedule'
                and self.location and self.location[0] == 'control'
                and self.control_type is not None)

    def _first_control_step(self):
        steps = self.control_steps
        return int(steps[0]) if steps else 0

    def unscale(self, u):
        u = np.asarray(u, dtype=float)
        lower, upper = self.box_lims[:, 0], self.box_lims[:, 1]
        if self.scaling == "linear":
            return lower + u * (upper - lower)
        if self.scaling == "log":
            if np.any(lower <= 0):
                raise ValueError("Log scaling requires positive lower bounds")
            factor = upper / lower
            return lower * np.power(factor, u)
        raise ValueError(f"Unsupported scaling: {self.scaling}")

    def scale(self, pval):
        pval = np.asarray(pval, dtype=float)
        lower, upper = self.box_lims[:, 0], self.box_lims[:, 1]
        if self.scaling == "linear":
            return (pval - lower) / (upper - lower)
        if self.scaling == "log":
            if np.any(pval <= 0) or np.any(lower <= 0):
                raise ValueError("Log scaling requires positive values")
            factor = upper / lower
            return np.log(pval / lower) / np.log(factor)
        raise ValueError(f"Unsupported scaling: {self.scaling}")

    def scale_gradient(self, grad, pval):
        grad = np.asarray(grad, dtype=float)
        lower, upper = self.box_lims[:, 0], self.box_lims[:, 1]
        if self.scaling == "linear":
            return grad * (upper - lower)
        if self.scaling == "log":
            if np.any(pval <= 0):
                raise ValueError("Log scaling requires positive values")
            factor = upper / lower
            return grad * pval * np.log(factor)
        raise ValueError(f"Unsupported scaling: {self.scaling}")


# ------------------------------------------------------ field access --

def _step(obj, key):
    if isinstance(key, tuple):
        return np.asarray(obj)[key]
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _getfield(obj, location):
    """``getfield(obj, location{:})`` over dicts, objects and slices."""
    for key in location:
        obj = _step(obj, key)
    return obj


def _default_getfun(obj, *location):
    """``@getfield`` -- the reader ``ModelParameter`` installs by default.

    Takes the location unpacked, as MATLAB's ``getfield(obj, loc{:})``
    does, so a parameter carrying a custom reader and one using the
    default are called identically.
    """
    return _getfield(obj, location)


def _setfield(obj, location, value):
    """``setfield`` over the same. The final step may be a slice, in
    which case the value is written into the existing array -- unless it
    carries derivatives, which no float array can hold, so then the
    whole array is replaced by a list of columns."""
    for key in location[:-1]:
        obj = _step(obj, key)
    last = location[-1]
    if isinstance(last, tuple):
        np.asarray(obj)[last] = value
    elif isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def _set_subset(whole, value, subset):
    out = np.asarray(whole, dtype=float).copy()
    out[subset] = value
    return out


def collapse_lumps(value, lumping):
    """One value per lump: the mean over each group, as MRST takes it."""
    if lumping is None:
        return value
    lumping = np.asarray(lumping, dtype=int).ravel()
    value = np.asarray(value, dtype=float).ravel()
    counts = np.bincount(lumping)
    return np.bincount(lumping, weights=value) / np.maximum(counts, 1)


def expand_lumps(value, lumping):
    """The inverse: scatter one value per lump back over its members."""
    if lumping is None:
        return value
    lumping = np.asarray(lumping, dtype=int).ravel()
    if hasattr(value, 'val'):          # keep an AD object AD
        return value[lumping]
    return np.asarray(value, dtype=float).ravel()[lumping]


def _controls(setup):
    schedule = setup['schedule']
    return schedule['control'] if isinstance(schedule, dict) \
        else schedule.control


def _get_control_value(control, control_type, collapse, lumping):
    wells = (control.get('W') if isinstance(control, dict)
             else control.W) or []
    value = np.asarray([float(w.get('val', 0.0)) for w in wells],
                       dtype=float)
    return collapse_lumps(value, lumping) if collapse else value


def _set_control_value(control, control_type, value):
    wells = (control.get('W') if isinstance(control, dict)
             else control.W) or []
    value = np.atleast_1d(value)
    for i, well in enumerate(wells):
        well['val'] = value[min(i, value.size - 1)]


# --------------------------------------------------- custom writers --

def perm2directional_trans(model, perm_column, direction):
    """Port of ``perm2directionalTrans``.

    The half-transmissibilities a permeability contributes along one
    coordinate direction: build a diagonal tensor that is zero in the
    other two directions and run ``computeTrans`` on it.

    ``ti`` is *linear* in the permeability, which is what makes the AD
    cheap -- compute it once from the numbers, then multiply by
    ``p/value(p)`` per cell to attach the derivative, instead of pushing
    an AD object through the geometry.
    """
    from PRSTCore.solvers.incomp.compute_trans import compute_trans

    values = perm_column.val if hasattr(perm_column, 'val') \
        else np.asarray(perm_column, dtype=float)
    values = np.asarray(values, dtype=float).ravel()

    perm = np.zeros((values.size, 3), dtype=float)
    perm[:, direction] = values
    ti = compute_trans(model.G, {'perm': perm})

    if not hasattr(perm_column, 'val'):
        return ti

    # One cell index per half-face, so the per-cell factor lines up.
    cellno = _half_face_cells(model.G)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = perm_column / values
    return ratio[cellno] * ti


def _half_face_cells(G):
    """``rldecode(1:num, diff(facePos))`` -- the owning cell of each
    half face."""
    cells = G['cells']
    face_pos = np.asarray(cells['facePos'], dtype=int).ravel()
    return np.repeat(np.arange(int(cells['num'])), np.diff(face_pos))


def set_permeability_fun(model, param, value):
    """Port of ``setPermeabilityFun``.

    Writing a permeability is not writing one array: the
    transmissibilities are a function of all three directions, so they
    are rebuilt here from every column. Doing it this way is what lets an
    AD permeability produce ``dT/dperm`` exactly -- including the
    harmonic averaging of the two half faces -- on any grid, rather than
    relying on a rule like "scaling PERMX scales the I-faces by the same
    factor", which only holds for an axis-aligned Cartesian grid.
    """
    import scipy.sparse as sp

    rock = model['rock'] if isinstance(model, dict) else model.rock
    perm = rock['perm']
    columns = list(perm) if isinstance(perm, list) else \
        [np.asarray(perm, dtype=float)[:, k]
         for k in range(np.asarray(perm).shape[1])]

    column = param.location[-1][-1]
    if column >= len(columns):
        raise ValueError("Can't set column %d: perm has %d column(s)"
                         % (column, len(columns)))
    columns[column] = value

    half = 0
    for k, col in enumerate(columns):
        half = half + perm2directional_trans(model, col, k)

    G = model['G'] if isinstance(model, dict) else model.G
    cf = np.asarray(G['cells']['faces'], dtype=int)
    cf = cf[:, 0] if cf.ndim == 2 else cf
    nf = int(np.asarray(G['faces']['neighbors']).shape[0])
    M = sp.csr_matrix((np.ones(cf.size), (cf, np.arange(cf.size))),
                      shape=(nf, cf.size))
    neighbors = np.asarray(G['faces']['neighbors'], dtype=int)
    internal = (neighbors[:, 0] >= 0) & (neighbors[:, 1] >= 0)

    reciprocal = 1.0 / half
    stacked = (reciprocal.linear_map(M[internal])
               if hasattr(reciprocal, 'val') else M[internal] @ reciprocal)
    operators = model['operators'] if isinstance(model, dict) \
        else model.operators
    operators['T'] = 1.0 / stacked

    if all(not hasattr(c, 'val') for c in columns):
        rock['perm'] = np.column_stack(columns)
    else:
        rock['perm'] = columns
    return model


def set_relperm_scalers_fun(model, param, value):
    """Port of ``setRelPermScalersFun``: write an endpoint into every
    curve that shares it. KRO is the reason this exists -- ECLIPSE's one
    keyword sets the oil maximum on both the water-oil and the gas-oil
    curve, and writing only the first would leave half the derivative
    behind."""
    for location in (param.location,) + tuple(param.extra_locations):
        _setfield(model, location, value)
    return model


def add_parameter(param_list, setup, *, name, scaling="linear",
                  box_lims=None, lumping=None, subset=None,
                  relative_limits=None, uniform_limits=True):
    """Create a ModelParameter and append it to the parameter list.

    If box_lims is not provided, it is computed from the current parameter
    values and relative_limits (default [0.5, 2]).

    uniform_limits follows MRST's ModelParameter: True (the default
    there and here) gives every entry the same box, taken from the
    parameter's global min and max scaled by relative_limits; False gives
    each entry its own box, v .* relative_limits. The two differ a lot on
    a heterogeneous field -- a uniform box lets a low-permeability cell
    reach a high-permeability cell's value, a per-entry box does not.
    """
    # ``setupDefaults``: ``v = getParameterValue(p, setup, false)`` and
    # ``nParam = numel(v)``. The value comes through the parameter's
    # *location*, not a switch on its name -- which is the whole reason
    # ModelParameter carries one. Reading it by name only works for the
    # handful the switch knows, and every other parameter silently comes
    # back as a single zero: one tunable number where the field has
    # 54080, and an optimiser that reports convergence after moving
    # nothing.
    probe = ModelParameter(name, n_param=1, scaling=scaling,
                           box_lims=[0.0, 1.0], setup=setup)
    if subset is not None:
        probe.subset = (np.flatnonzero(np.asarray(subset))
                        if np.asarray(subset).dtype == bool else subset)
    try:
        pval = np.asarray(probe.get_parameter_value(setup, collapse=False),
                          dtype=float).ravel()
    except Exception:
        pval = np.asarray(_get_model_parameter_value(setup["model"], name),
                          dtype=float).ravel()

    if box_lims is None:
        if relative_limits is None:
            relative_limits = [0.5, 2.0]
        rlo, rhi = float(relative_limits[0]), float(relative_limits[1])

        if uniform_limits:
            # One box for all entries. A negative endpoint takes the
            # *other* relative limit, since scaling a negative number by
            # the larger factor produces the smaller value.
            lo_v, hi_v = float(np.min(pval)), float(np.max(pval))
            lo = lo_v * (rhi if lo_v < 0 else rlo)
            hi = hi_v * (rlo if hi_v < 0 else rhi)
            lower = np.full(pval.size, lo)
            upper = np.full(pval.size, hi)
        else:
            lower = pval * rlo
            upper = pval * rhi

        # MRST-0 replaces MRST's swap with an error. Stock MRST writes
        # ``boxLims(isNeg,:) = boxLims(isNeg,[2,1])`` and MRST-0 comments
        # that line out in favour of refusing outright, because a
        # negative limit on a permeability or a pore volume means the
        # input was wrong -- quietly swapping lets the optimiser search a
        # box with no physical meaning and report a converged answer
        # from inside it. Only a schedule's ``val`` may legitimately be
        # negative: a producer's rate is.
        negative = (lower < 0) | (upper < 0)
        is_schedule_val = (probe.belongs_to == 'schedule'
                           and probe.location
                           and probe.location[-1] == 'val')
        if np.any(negative) and not is_schedule_val:
            raise ValueError('Negative limits found for parameter %s'
                             % name)

        # Saturations are bounded by their own physical range.
        if name in ("sw", "sg"):
            lower = np.zeros(pval.size)
            upper = np.ones(pval.size)
        elif np.any(pval == 0) and not uniform_limits:
            # MRST-0: ``[0, 1]``, and only for per-entry boxes. Stock
            # MRST uses ``[-1, 1]`` unconditionally, which lets a cell
            # whose pore volume is zero be tuned negative.
            zero = pval == 0
            lower = lower.copy()
            upper = upper.copy()
            lower[zero] = 0.0
            upper[zero] = 1.0
            warnings.warn(
                'Parameter %s contains zero-values. Defaulting lower/upper '
                "limits ('boxLims') to [0 1]" % name, RuntimeWarning,
                stacklevel=2)

        # Not in MRST: log scaling cannot represent a non-positive
        # bound, so clamp rather than produce a box the scaler will
        # return NaN from.
        if scaling == "log":
            lower = np.maximum(lower, 1e-12)
            upper = np.maximum(upper, lower * 2)
        box_lims = np.column_stack([lower, upper])

    param = ModelParameter(name, n_param=pval.size,
                           scaling=scaling, box_lims=box_lims,
                           lumping=lumping, subset=probe.subset,
                           relative_limits=relative_limits, setup=setup)
    if param_list is None:
        return [param]
    if isinstance(param_list, list):
        return param_list + [param]
    raise ValueError("Unknown format of input 'param_list'")


def update_setup_from_scaled_parameters(setup, parameters, pvec,
                                        recompute_wi=False):
    """Port of ``updateSetupFromScaledParameters``.

    Each parameter unscales its own slice and writes it back through
    :meth:`ModelParameter.set_parameter`, so the ``belongs_to``/
    ``location``/``setfun`` indirection decides where the value lands --
    a permeability column, a relperm scaler table, a well control.  The
    three property-function caches are cleared afterwards because they
    hold values derived from what has just changed.
    """
    pvec = np.asarray(pvec, dtype=float)
    pvals = []
    idx = 0
    for param in parameters:
        vec = pvec[idx:idx + param.n_param]
        pvals.append(param.unscale(vec))
        idx += param.n_param
    if idx != pvec.size:
        raise ValueError("Parameter vector length does not match parameter definitions")

    setup_new = dict(setup)
    setup_new["model"] = _copy_model(setup["model"])
    if "schedule" in setup:
        setup_new["schedule"] = copy.deepcopy(setup["schedule"])
    if "state0" in setup:
        setup_new["state0"] = copy.deepcopy(setup["state0"])

    for param, pval in zip(parameters, pvals):
        setup_new = param.set_parameter(setup_new,
                                        np.asarray(pval, dtype=float))

    if recompute_wi:
        from PRSTCore.hm.utils.recomputeWellIndex import recomputeWellIndex
        setup_new["schedule"] = recomputeWellIndex(setup_new["model"],
                                                   setup_new["schedule"])

    for field in ("FlowDiscretization", "FlowPropertyFunctions",
                  "PVTPropertyFunctions"):
        if hasattr(setup_new["model"], field):
            setattr(setup_new["model"], field, None)
    return setup_new


def get_scaled_parameter_vector(setup, params):
    """Port of ``getScaledParameterVector``.

    ``p.getParameter(setup)`` then ``p.scale(...)``.  Reading the field
    through the parameter -- rather than by name off the model -- is what
    applies its ``subset``: tuning permeability on the cells where it is
    positive must hand ``scale`` only those cells, since log scaling has
    nothing to say about a zero.
    """
    u = []
    for p in params:
        val = np.asarray(p.get_parameter(setup), dtype=float).ravel()
        u.append(p.scale(val))
    return np.concatenate(u)


def _copy_model(model):
    if isinstance(model, dict):
        out = dict(model)
        if "operators" in out and isinstance(out["operators"], dict):
            out["operators"] = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v)
                                for k, v in out["operators"].items()}
        return out
    out = copy.copy(model)
    ops = getattr(model, "operators", None)
    if isinstance(ops, dict):
        setattr(out, "operators", {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v)
                                   for k, v in ops.items()})
    return out


def _model_get(model, name, default=None):
    if isinstance(model, dict):
        return model.get(name, default)
    return getattr(model, name, default)


def _model_set(model, name, value):
    if isinstance(model, dict):
        model[name] = value
    else:
        setattr(model, name, value)


def _get_model_parameter_value(model, name, default_size=1):
    ops = _model_get(model, "operators", {}) or {}
    if name == "porevolume":
        val = _model_get(model, "porevolume", None)
        if val is not None:
            return val
        if isinstance(ops, dict) and "pv" in ops:
            return ops["pv"]
    if name == "transmissibility":
        if isinstance(ops, dict) and "T" in ops:
            return ops["T"]
    if name == "conntrans":
        val = _model_get(model, "conntrans", None)
        if val is not None:
            return val
    val = _model_get(model, name, None)
    if val is None:
        return np.zeros(default_size)
    return val


def _apply_model_parameter(setup, name, value):
    model = setup["model"]
    ops = _model_get(model, "operators", None)
    if name == "porevolume":
        _model_set(model, "porevolume", value)
        if isinstance(ops, dict):
            ops["pv"] = np.asarray(value, dtype=float).ravel()
        _model_set(model, name, value)
        return

    if name == "transmissibility":
        val = np.asarray(value, dtype=float).ravel()
        if isinstance(ops, dict):
            if "T" in ops and np.asarray(ops["T"]).size == val.size:
                ops["T"] = val
                if "T_all" in ops:
                    t_all = np.asarray(ops["T_all"], dtype=float).copy()
                    N_all = _coarse_all_neighbors(model)
                    if N_all is not None and t_all.size == N_all.shape[0]:
                        internal = np.all(N_all != 0, axis=1)
                        if internal.sum() == val.size:
                            t_all[internal] = val
                            ops["T_all"] = t_all
            elif "T_all" in ops and np.asarray(ops["T_all"]).size == val.size:
                ops["T_all"] = val
                N_all = _coarse_all_neighbors(model)
                if N_all is not None and val.size == N_all.shape[0]:
                    ops["T"] = val[np.all(N_all != 0, axis=1)]
            else:
                ops["T"] = val
        _model_set(model, name, val)
        return

    if name == "conntrans":
        val = np.asarray(value, dtype=float).ravel()
        _model_set(model, "conntrans", val)
        _apply_conntrans_to_schedule(setup, val)
        return

    _model_set(model, name, value)


def _coarse_all_neighbors(model):
    G = _model_get(model, "G", None)
    if isinstance(G, dict) and "faces" in G and "neighbors" in G["faces"]:
        return np.asarray(G["faces"]["neighbors"], dtype=int)
    return None


def _apply_conntrans_to_schedule(setup, conntrans):
    schedule = setup.get("schedule")
    if not isinstance(schedule, dict) or "control" not in schedule:
        return
    idx = 0
    for ctrl in schedule["control"]:
        wells = ctrl.get("W", []) if isinstance(ctrl, dict) else []
        for w in wells:
            cells = np.atleast_1d(w.get("cells", []))
            nperf = len(cells)
            base = np.asarray(w.get("_base_WI", w.get("WI", np.ones(nperf))), dtype=float).ravel()
            if base.size == 1 and nperf > 1:
                base = np.full(nperf, base[0])
            if base.size < nperf:
                base = np.pad(base, (0, nperf - base.size), constant_values=0.0)
            if "_base_WI" not in w:
                w["_base_WI"] = base.copy()
            if conntrans.size == 1:
                mult = np.full(nperf, conntrans[0])
            elif conntrans.size >= idx + nperf:
                mult = conntrans[idx:idx + nperf]
            elif conntrans.size == len(wells):
                mult = np.full(nperf, conntrans[min(idx, conntrans.size - 1)])
            else:
                mult = np.ones(nperf)
            w["WI"] = (base[:nperf] * mult).tolist()
            idx += nperf
