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
                 type="value", reference_value=None, control_steps=None,
                 scaling_base=None):
        self.name = name
        self.n_param = int(n_param)
        self.scaling = scaling
        self.lumping = copy.deepcopy(lumping)
        self.subset = copy.deepcopy(subset)
        self.scaling_base = scaling_base
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
            self.box_lims = box_lims.copy()

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
            value = value[self.subset] if hasattr(value, 'val') else np.asarray(value)[self.subset]
        return copy.deepcopy(collapse_lumps(value, self.lumping) if collapse else value)

    def set_parameter_value(self, setup, value, expand=True):
        """Port of ``setParameterValue``."""
        return self._set_parameter_value_owned(copy.deepcopy(setup), value, expand)

    def _set_parameter_value_owned(self, setup, value, expand=True):
        """Internal writer: caller owns every mutable object in setup."""
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
            _setfield(target, self.location, copy.deepcopy(value))
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

    @boxLims.setter
    def boxLims(self, value):
        limits = np.atleast_2d(np.asarray(value, dtype=float))
        if limits.shape not in ((1, 2), (self.n_param, 2)):
            raise ValueError('boxLims must have one row or nParam rows')
        self.box_lims = limits.copy()

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
        u = np.asarray(u, dtype=float).ravel(order='F')
        lower, upper = self.box_lims[:, 0], self.box_lims[:, 1]
        if self.scaling == "log":
            base = self._scaling_base()
            u = (base ** u - 1) / (base - 1)
        elif self.scaling == "exp":
            base = self._scaling_base()
            u = np.log((base - 1) * u + 1) / np.log(base)
        elif self.scaling != "linear":
            raise ValueError(f"Unsupported scaling: {self.scaling}")
        return u * (upper - lower) + lower

    def scale(self, pval):
        pval = np.asarray(pval, dtype=float).ravel(order='F')
        lower, upper = self.box_lims[:, 0], self.box_lims[:, 1]
        vs = (pval - lower) / (upper - lower)
        if self.scaling == "log":
            base = self._scaling_base()
            vs = np.log((base - 1) * vs + 1) / np.log(base)
        elif self.scaling == "exp":
            base = self._scaling_base()
            vs = (base ** vs - 1) / (base - 1)
        elif self.scaling != "linear":
            raise ValueError(f"Unsupported scaling: {self.scaling}")
        return vs

    def scale_gradient(self, grad, pval):
        grad = np.asarray(grad, dtype=float).ravel(order='F')
        lower, upper = self.box_lims[:, 0], self.box_lims[:, 1]
        gs = grad * (upper - lower)
        v = (np.asarray(pval).ravel(order='F') - lower) / (upper - lower)
        if self.scaling == "log":
            base = self._scaling_base()
            gs = gs / ((base - 1) / (((base - 1) * v + 1) * np.log(base)))
        elif self.scaling == "exp":
            base = self._scaling_base()
            gs = gs / (base ** v * (np.log(base) / (base - 1)))
        elif self.scaling != "linear":
            raise ValueError(f"Unsupported scaling: {self.scaling}")
        return gs

    def _scaling_base(self):
        if self.scaling_base is not None:
            return self.scaling_base
        lower, upper = self.box_lims[:, 0], self.box_lims[:, 1]
        if np.any(lower <= 0) or np.any(upper <= lower):
            raise ValueError('Log/exp scaling requires positive, non-degenerate bounds')
        return upper / lower


# ------------------------------------------------------ field access --

def _step(obj, key):
    if isinstance(key, tuple):
        if isinstance(obj, list):
            return obj[key[1]][key[0]]
        if hasattr(obj, 'val'):
            return obj[key]
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
    if hasattr(whole, 'val') or hasattr(value, 'val'):
        from PRSTCore.ad_core.adi import SparseADI
        nvar = whole.nvar if hasattr(whole, 'val') else value.nvar
        out = whole if hasattr(whole, 'val') else SparseADI.constant(np.asarray(whole), nvar)
        val = value if hasattr(value, 'val') else SparseADI.constant(np.asarray(value), nvar)
        return out + SparseADI.scatter(subset, val - out[subset], out.val.size)
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
    """MRST local-coordinate half-transmissibility, retaining AD."""
    from .parameter_transmissibility import directional_trans
    return directional_trans(model, perm_column, direction)

def _half_face_cells(G):
    """``rldecode(1:num, diff(facePos))`` -- the owning cell of each
    half face."""
    cells = G['cells']
    face_pos = np.asarray(cells['facePos'], dtype=int).ravel()
    return np.repeat(np.arange(int(cells['num'])), np.diff(face_pos))


def set_permeability_fun(model, param, value):
    """Set one diagonal PERM column and rebuild MRST T, including NNC."""
    from .parameter_transmissibility import assemble_trans
    rock = _model_get(model, 'rock')
    perm = rock['perm']
    columns = list(perm) if isinstance(perm, list) else [
        np.asarray(perm)[:, k].copy() for k in range(np.asarray(perm).shape[1])]
    column = param.location[-1][-1]
    if column >= len(columns):
        raise ValueError("Requested permeability direction is absent")
    columns[column] = copy.deepcopy(value)
    _model_get(model, 'operators')['T'] = assemble_trans(model, columns)
    rock['perm'] = columns if any(hasattr(c, 'val') for c in columns) else np.column_stack(columns)
    return model

def set_relperm_scalers_fun(model, param, value):
    """Port of ``setRelPermScalersFun``: write an endpoint into every
    curve that shares it. KRO is the reason this exists -- ECLIPSE's one
    keyword sets the oil maximum on both the water-oil and the gas-oil
    curve, and writing only the first would leave half the derivative
    behind."""
    for location in (param.location,) + tuple(param.extra_locations):
        table = _getfield(model, location[:-1])
        if isinstance(table, list) or hasattr(value, 'val'):
            # MATLAB stores AD columns in drainage.tmp.<phase> and exposes
            # a column-indexing callback. Python uses an explicit column
            # list; _step implements the same cells/column access.
            columns = copy.deepcopy(table) if isinstance(table, list) else [
                np.asarray(table)[:, k].copy() for k in range(np.asarray(table).shape[1])]
            columns[location[-1][1]] = copy.deepcopy(value)
            _setfield(model, location[:-1], columns)
        else:
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
    pval = np.asarray(probe.get_parameter_value(setup, collapse=False),
                      dtype=float).ravel(order='F')
    if pval.size == 0:
        raise ValueError('Parameter %s has an empty active-cell subset' % name)
    if probe.subset is not None:
        sub = np.asarray(probe.subset)
        if sub.ndim != 1 or sub.dtype.kind not in 'iu' or np.any(sub < 0) or np.unique(sub).size != sub.size:
            raise ValueError('subset must contain distinct zero-based indices')

    if box_lims is None:
        if relative_limits is None:
            relative_limits = [0.5, 2.0]
        relative_limits = np.atleast_2d(np.asarray(relative_limits, dtype=float))
        if relative_limits.shape not in ((1, 2), (pval.size, 2)):
            raise ValueError('relative_limits must have one row or one row per selected cell')
        rlo, rhi = relative_limits[:, 0], relative_limits[:, 1]

        if uniform_limits:
            # One box for all entries. A negative endpoint takes the
            # *other* relative limit, since scaling a negative number by
            # the larger factor produces the smaller value.
            lo_v, hi_v = float(np.min(pval)), float(np.max(pval))
            # MATLAB linear indexing of rlim uses its first two elements.
            flat = relative_limits.ravel(order='F')
            lo = lo_v * flat[int(lo_v < 0)]
            hi = hi_v * flat[1 - int(hi_v < 0)]
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

        box_lims = np.column_stack([lower, upper])

    if lumping is not None:
        lumping = np.asarray(lumping, dtype=int).ravel()
        # Python lump IDs are zero-based; MATLAB's scalar lump 1 is 0 here.
        if lumping.size == 1 and lumping[0] == 0:
            lumping = np.zeros(pval.size, dtype=int)
        if lumping.size != pval.size or np.any(lumping < 0):
            raise ValueError('Lumping vector has incorrect size or indices')
        n_param = int(lumping.max()) + 1
    else:
        n_param = pval.size
    param = ModelParameter(name, n_param=n_param,
                           scaling=scaling, box_lims=box_lims,
                           lumping=lumping, subset=probe.subset,
                           relative_limits=relative_limits, setup=setup)
    check = np.asarray(param.get_parameter(setup))
    if np.any((check < param.box_lims[:, 0]) | (check > param.box_lims[:, 1])):
        raise ValueError('Parameter values are not within given limits: %s' % name)
    if scaling in ('log', 'exp'):
        param._scaling_base()  # Never silently clamp SI permeabilities.
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
    pvec = np.asarray(pvec, dtype=float).ravel(order='F')
    if pvec.size != sum(p.n_param for p in parameters):
        raise ValueError('Parameter vector length does not match parameter definitions')
    pvals = []
    idx = 0
    for param in parameters:
        vec = pvec[idx:idx + param.n_param]
        pvals.append(param.unscale(vec))
        idx += param.n_param
    if idx != pvec.size:
        raise ValueError("Parameter vector length does not match parameter definitions")

    setup_new = copy.deepcopy(setup)

    for param, pval in zip(parameters, pvals):
        if param.type == 'multiplier':
            pval = expand_lumps(pval, param.lumping) * param.reference_value
            setup_new = param._set_parameter_value_owned(setup_new, pval, expand=False)
        else:
            setup_new = param._set_parameter_value_owned(setup_new, pval)

    if recompute_wi:
        from PRSTCore.hm.utils.recomputeWellIndex import recomputeWellIndex
        setup_new["schedule"] = recomputeWellIndex(setup_new["model"],
                                                   setup_new["schedule"])

    for field in ("FlowDiscretization", "FlowPropertyFunctions",
                  "PVTPropertyFunctions"):
        if isinstance(setup_new['model'], dict):
            setup_new['model'][field] = None
        elif hasattr(setup_new["model"], field):
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
    return np.concatenate(u) if u else np.empty(0, dtype=float)


def _copy_model(model):
    return copy.deepcopy(model)


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
