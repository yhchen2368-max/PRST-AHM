"""Strict FAHM external-result contracts (MRST getEclipseSimResults/sim2rep)."""

from collections import defaultdict, deque

import numpy as np


class RestartContractError(ValueError):
    """A simulator result cannot represent the requested schedule/state."""


def time_vector(value, label):
    result = np.asarray(value, dtype=float).ravel(order='F')
    if not result.size or not np.all(np.isfinite(result)):
        raise RestartContractError(f'{label} must be nonempty and finite')
    if np.any(np.diff(result) <= 0):
        raise RestartContractError(f'{label} must be strictly increasing')
    return result


def report_indices(simulated, reported, *, allow_trailing=False):
    """Source first crossing within min(diff(T_sim))/10, never nearest.

    FAHM's warning-only large mismatch is an error under Stage 11's Gate.
    No nearest-neighbour reuse, trimming or synthesized report state.
    """
    simulated = time_vector(simulated, 'T_sim')
    reported = time_vector(reported, 'T_rep')
    if simulated.size == 1:
        if np.array_equal(simulated, reported):
            return np.array([0], dtype=int)
        raise RestartContractError('Ministeps not compatible with report times')
    threshold = np.min(np.diff(simulated)) / 10.0
    indices = []
    for k, time in enumerate(simulated):
        if len(indices) == reported.size:
            if allow_trailing:
                break
            raise RestartContractError('Restart contains states after the final report')
        if reported[len(indices)] <= time + threshold:
            indices.append(k)
    if len(indices) != reported.size:
        raise RestartContractError('Ministeps not compatible with report times')
    indices = np.asarray(indices, dtype=int)
    if np.any(np.abs(reported - simulated[indices]) > threshold):
        raise RestartContractError('Mismatch observed between ministeps and reportsteps')
    return indices


def exact_order(wanted, available, label='well/perforation'):
    """Stable occurrence-wise bijection; missing/extra entries are errors."""
    wanted, available = list(wanted), list(available)
    pools = defaultdict(deque)
    for i, value in enumerate(available):
        pools[value].append(i)
    result = []
    for value in wanted:
        if not pools[value]:
            raise RestartContractError(f'{label}: missing entry {value!r}')
        result.append(pools[value].popleft())
    if any(pools.values()):
        raise RestartContractError(f'{label}: unexpected extra entries')
    return result


def subset_order(wanted, available, label='perforation'):
    """Map a source subset into the final MRST well template, without loss."""
    pools = defaultdict(deque)
    for i, value in enumerate(wanted):
        pools[value].append(i)
    result = []
    for value in available:
        if not pools[value]:
            raise RestartContractError(f'{label}: entry {value!r} absent from final template')
        result.append(pools[value].popleft())
    return np.asarray(result, dtype=int)
