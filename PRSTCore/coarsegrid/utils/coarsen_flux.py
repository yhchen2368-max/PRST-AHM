"""Compute net flux on coarse faces.

1:1 Python translation of MRST multiscale/coarsegrid/utils/coarsenFlux.m
"""

import numpy as np


def coarsen_flux(cg, flux):
    """Accumulate fine-grid fluxes onto coarse faces.

    Parameters
    ----------
    cg : dict
        Coarse grid with parent, faces.connPos, faces.fconn, faces.num.
    flux : ndarray
        Fine-grid face fluxes (cg.parent.faces.num,).

    Returns
    -------
    ndarray
        Coarse face fluxes (cg.faces.num,).
    """
    flux = np.asarray(flux, dtype=float).ravel()
    faceno = np.repeat(np.arange(cg["faces"]["num"]),
                       np.diff(cg["faces"]["connPos"]))
    from .fine_to_coarse_sign import fine_to_coarse_sign
    sgn = fine_to_coarse_sign(cg)
    cflux = np.bincount(faceno, weights=flux[cg["faces"]["fconn"]] * sgn,
                         minlength=cg["faces"]["num"])
    return cflux
