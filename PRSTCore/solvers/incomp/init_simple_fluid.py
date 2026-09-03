"""Python port of MRST's ``initSimpleFluid.m``
(mrst-2026a/solvers/incomp/fluid/incompressible): an incompressible
two-phase fluid model with analytic (pure) Corey relative permeabilities
``kr = s**n``.

Built directly on :class:`PRSTCore.solvers.incomp.transport.TwoPhaseFluid` /
``corey_fluid`` (``swc=sor=0``, ``krw_max=kro_max=1`` reproduces
``kr = s**n`` exactly), rather than reimplementing the relperm formula.
"""

from __future__ import annotations

from .transport import corey_fluid


def init_simple_fluid(*, mu=(1.0e-3, 1.0e-3), rho=(1.0, 1.0), n=(1.0, 1.0)):
    """Port of ``initSimpleFluid.m``.

    Parameters
    ----------
    mu : (mu_w, mu_o)
        Phase viscosities, Pa.s (MRST default ``1*centi*poise`` each, i.e.
        ``1e-3``).
    rho : (rho_w, rho_o)
        Phase surface densities, kg/m^3 (informational; not used by the
        incompressible pressure/transport solvers in this port).
    n : (n_w, n_o)
        Corey exponents, one per phase (``krW(s) = s**n_w``,
        ``krO(s) = (1-s)**n_o``).
    """
    fluid = corey_fluid(mu[0], mu[1], nw=n[0], no=n[1])
    fluid.rhoWS, fluid.rhoOS = rho
    return fluid
