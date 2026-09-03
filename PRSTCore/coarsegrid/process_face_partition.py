"""Process face partition to ensure connected coarse faces."""

from collections import defaultdict, deque

import numpy as np


def process_face_partition(g, p, pf):
    """Ensure all coarse faces are connected collections of fine faces.

    Parameters
    ----------
    g : dict
        Grid with faces.neighbors, faces.nodePos, faces.nodes.
    p : ndarray
        Cell partition vector.
    pf : ndarray
        Face partition vector.

    Returns
    -------
    ndarray
        Processed face partition.
    """
    ndim = g.get("griddim", 2)
    if ndim != 2:
        raise ValueError("processFacePartition only supported in 2D")

    nbrs = np.asarray(g["faces"]["neighbors"], dtype=int)
    p_ext = np.concatenate([[0], p])
    B = np.sort(p_ext[nbrs], axis=1)
    f = np.where(np.any(B == 0, axis=1) | (B[:, 0] != B[:, 1]))[0]
    if f.size == 0:
        return np.asarray(pf, dtype=int).copy()

    node_pos = np.asarray(g["faces"]["nodePos"])
    nodes = np.asarray(g["faces"]["nodes"])
    pf = np.asarray(pf, dtype=int).ravel().copy()

    face_nodes = {}
    node_to_faces = defaultdict(list)
    selected = set(int(fi) for fi in f)
    for fi in f:
        fn = np.asarray(nodes[node_pos[fi]:node_pos[fi + 1]], dtype=int).ravel()
        face_nodes[int(fi)] = fn
        for node in fn:
            node_to_faces[int(node)].append(int(fi))

    adj = {int(fi): [] for fi in f}
    for incident in node_to_faces.values():
        for i in range(len(incident)):
            a = incident[i]
            if a not in selected:
                continue
            for j in range(i + 1, len(incident)):
                b = incident[j]
                if b not in selected:
                    continue
                if pf[a] == pf[b] and np.array_equal(B[a], B[b]):
                    adj[a].append(b)
                    adj[b].append(a)

    new_pf = pf.copy()
    next_id = int(max(pf.max(initial=0), 0)) + 1
    seen = set()
    for seed in f:
        seed = int(seed)
        if seed in seen:
            continue
        q = deque([seed])
        seen.add(seed)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)

        # Preserve the original id for the first component encountered for a
        # given face partition, split later components into fresh ids.
        same_before = [fi for fi in f if fi < seed and pf[fi] == pf[seed] and np.array_equal(B[fi], B[seed])]
        if same_before:
            new_id = next_id
            next_id += 1
            new_pf[comp] = new_id
        else:
            new_pf[comp] = pf[seed]

    return new_pf
