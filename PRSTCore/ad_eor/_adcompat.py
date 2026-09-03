"""Small AD/plain-array compatibility shims shared by the ``ad_eor`` port.

MRST's ``StateFunction.evaluateOnDomain`` methods run unchanged whether the
inputs are plain doubles or ``ADI``/``GenericAD`` objects (MATLAB operator
overloading). PRSTCore's :class:`~PRSTCore.ad_core.adi.SparseADI` overloads
``+ - * / **``, but elementwise ``max``/``min`` need the dedicated
``ad_maximum``/``ad_minimum`` helpers when either operand carries
derivatives. These wrappers pick the right one so the ``ad_eor`` property
functions can be called with plain ``numpy`` arrays (e.g. convergence checks)
or :class:`SparseADI` values (residual assembly) exactly like the ``.m``
source is called with ``double`` or ``ADI``.
"""

import numpy as _np

from PRSTCore.ad_core.adi import SparseADI as _SparseADI
from PRSTCore.ad_core.adi import ad_maximum as _ad_maximum
from PRSTCore.ad_core.adi import ad_minimum as _ad_minimum


def amax(a, b):
    if isinstance(a, _SparseADI) or isinstance(b, _SparseADI):
        return _ad_maximum(a, b)
    return _np.maximum(a, b)


def amin(a, b):
    if isinstance(a, _SparseADI) or isinstance(b, _SparseADI):
        return _ad_minimum(a, b)
    return _np.minimum(a, b)


def value(x):
    """MRST ``value(x)``: strip derivatives, if any."""
    return x.val if isinstance(x, _SparseADI) else _np.asarray(x)
