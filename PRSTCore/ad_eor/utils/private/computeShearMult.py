"""Port of MRST ``computeShearMult.m``.

Solves ECLIPSE Technical Description EQ 52.12 for the PLYSHEAR shear
multiplier: ``V*(1 + (P-1)*M(V))/P = Vw`` for ``V`` (the "sheared" velocity),
elementwise per cell, by Newton iteration.

MRST solves this with an ADI variable so the Jacobian of ``shFunc`` (a
diagonal matrix, since the equation is elementwise) comes for free; here the
same per-cell diagonal derivative is obtained by a central finite
difference, which is numerically equivalent for this scalar elementwise
root-find (the derivative is only ever used to pick a Newton direction, not
propagated further). The convergence tolerance is relaxed from the ``.m``
source's ``1e-15`` to ``1e-12`` to stay achievable under a finite-difference
Jacobian.
"""

import numpy as _np


def computeShearMult(fluid, Vw, muWMultf):
    Vw = _np.asarray(Vw, dtype=float).ravel()
    muWMultf = _np.asarray(muWMultf, dtype=float).ravel()
    plyshearMult = fluid['plyshearMult']

    Vsh = Vw.copy()

    def shFunc(x):
        return x * (1.0 + (muWMultf - 1.0) * plyshearMult(x)) - muWMultf * Vw

    eqs = shFunc(Vsh)
    resnorm = float(_np.max(_np.abs(eqs))) if eqs.size else 0.0
    iteration = 0
    maxit = 30
    abstol = 1.0e-12

    while resnorm > abstol and iteration <= maxit:
        eps = 1.0e-6 * _np.maximum(_np.abs(Vsh), 1.0)
        deriv = (shFunc(Vsh + eps) - eqs) / eps
        deriv = _np.where(_np.abs(deriv) < 1.0e-300, 1.0e-300, deriv)
        dVsh = -eqs / deriv
        Vsh = Vsh + dVsh

        eqs = shFunc(Vsh)
        resnorm = float(_np.max(_np.abs(eqs))) if eqs.size else 0.0
        iteration += 1

    if iteration >= maxit and resnorm > abstol:
        raise RuntimeError(
            f'Convergence failure within {maxit} iterations\n'
            f'Final residual = {resnorm:.8e}')

    M = plyshearMult(Vsh)
    return (1.0 + (muWMultf - 1.0) * M) / muWMultf
