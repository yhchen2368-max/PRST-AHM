"""Process partition to remove disconnected coarse blocks.

This mirrors the essential behavior of MRST's
``multiscale/coarsegrid/processPartition.m``: cells that initially belong
to the same coarse block are split into separate block numbers whenever
they are not connected through block-internal grid faces.
"""

from collections import defaultdict, deque

import numpy as np


def process_partition(G, partition, facelist=None):
    """Split disconnected components inside each coarse block.

    Parameters
    ----------
    G : dict
        Grid structure.
    partition : ndarray
        Raw partition vector.

    Returns
    -------
    ndarray
        Processed partition (1..N, no gaps).
    """
    p = np.asarray(partition, dtype=int).ravel().copy()
    if p.size != int(G["cells"]["num"]):
        raise ValueError("partition must have one entry per grid cell")
    if p.size == 0:
        return p
    if p.min() < 1:
        p = p - p.min() + 1

    nbrs = np.asarray(G["faces"]["neighbors"], dtype=int)
    nf = int(G["faces"]["num"])
    cut = np.zeros(nf, dtype=bool)
    if facelist is not None:
        fl = np.asarray(facelist)
        if fl.dtype == bool:
            if fl.size != nf:
                raise ValueError("logical facelist must have one entry per face")
            cut = fl.ravel().copy()
        else:
            # Accept both MRST-style 1-based and Python-style 0-based face ids.
            fl = fl.astype(int).ravel()
            if fl.size:
                if fl.min() >= 1 and fl.max() <= nf:
                    fl = fl - 1
                elif fl.min() < 0 or fl.max() >= nf:
                    raise ValueError("face number not within valid range")
                cut[fl] = True

    cells_by_block = defaultdict(list)
    for ci, bi in enumerate(p, start=1):
        cells_by_block[int(bi)].append(ci)

    out = np.full_like(p, -1)
    next_block = max(cells_by_block) + 1

    for block in sorted(cells_by_block):
        cells = cells_by_block[block]
        cell_set = set(cells)
        adj = {c: [] for c in cells}

        for f, (c1, c2) in enumerate(nbrs):
            if cut[f] or c1 <= 0 or c2 <= 0:
                continue
            if c1 in cell_set and c2 in cell_set and p[c1 - 1] == block and p[c2 - 1] == block:
                adj[int(c1)].append(int(c2))
                adj[int(c2)].append(int(c1))

        seen = set()
        first_component = True
        for seed in cells:
            if seed in seen:
                continue
            q = deque([seed])
            seen.add(seed)
            comp = []
            while q:
                c = q.popleft()
                comp.append(c)
                for nb in adj[c]:
                    if nb not in seen:
                        seen.add(nb)
                        q.append(nb)

            new_id = block if first_component else next_block
            if not first_component:
                next_block += 1
            first_component = False
            out[np.asarray(comp, dtype=int) - 1] = new_id

    if np.any(out < 1):
        raise RuntimeError("failed to assign processed partition for all cells")
    return out


def compress_partition(partition):
    """Renumber partition from 1..N without gaps.

    Parameters
    ----------
    partition : ndarray
        Partition vector.

    Returns
    -------
    ndarray
        Compressed partition (1..max).
    """
    p = np.asarray(partition, dtype=int)
    _, inv = np.unique(p, return_inverse=True)
    return inv + 1


def partition_ui(G, nxyz):
    """Create a uniform Cartesian partition.

    Parameters
    ----------
    G : dict
        Grid with cells.centroids.
    nxyz : list
        Number of blocks in each direction.

    Returns
    -------
    ndarray
        Partition vector (1..prod(nxyz)).
    """
    nxyz = np.atleast_1d(nxyz).ravel()
    nc = G["cells"]["num"]
    centroids = np.asarray(G["cells"]["centroids"])
    nd = centroids.shape[1]
    if len(nxyz) < nd:
        nxyz = np.pad(nxyz, (0, nd - len(nxyz)), constant_values=1)

    partition = np.zeros(nc, dtype=int)
    stride = int(np.prod(nxyz[1:]))
    for d in range(nd):
        bins = np.linspace(centroids[:, d].min() - 1e-6,
                           centroids[:, d].max() + 1e-6,
                           int(nxyz[d]) + 1)
        idx = np.digitize(centroids[:, d], bins) - 1
        idx = np.clip(idx, 0, int(nxyz[d]) - 1)
        if d == 0:
            partition += idx * stride
        else:
            partition += idx * int(np.prod(nxyz[d + 1:]))

    return partition + 1
