"""Port of MRST ``makeConnListFromMat``: make the connectivity list from the
node distribution matrix of a structured grid."""

import numpy as np

from .._core import mergeOptions


def makeConnListFromMat(nd, **kwargs):
    """Make the connectivity list from the node distribution matrix ``nd``
    for a structured grid.

    Parameters
    ----------
    nd : ndarray
        Node distribution matrix (``nny x nnx``), the node of cell ``(i, j)``
        is ``{nd[i,j], nd[i+1,j], nd[i+1,j+1], nd[i,j+1]}``.
    order : str, optional
        ``'rows'`` (default) | ``'column'``: the picking order (the numbering
        of the connectivity list cycles along ``order`` fastest).

    Returns
    -------
    t : list of 1D arrays (each with 4 node ids, 0-based)
    """
    opt = mergeOptions({'order': 'rows'}, **kwargs)
    nd = np.asarray(nd, dtype=np.int64)
    if opt['order'] == 'column':
        nd = nd.T
    elif opt['order'] != 'rows':
        raise ValueError('Unknown order')

    ny = nd.shape[0] - 1
    nx = nd.shape[1] - 1
    # MATLAB: i = repmat((1:ny)', 1, nx), j = repmat((1:nx), ny, 1); the
    # column-major flatten of i/j makes the first index (rows / y) cycle
    # fastest, i.e. cell k = nd[:, y = k%ny, x = k//ny].
    i = np.tile(np.arange(ny), nx)
    j = np.repeat(np.arange(nx), ny)
    t = [np.array([nd[ii, jj], nd[ii + 1, jj], nd[ii + 1, jj + 1], nd[ii, jj + 1]],
                  dtype=np.int64)
         for ii, jj in zip(i, j)]
    return t
