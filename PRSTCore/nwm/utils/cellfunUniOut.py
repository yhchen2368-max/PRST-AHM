"""Port of MRST ``cellfunUniOut`` (``cellfun`` with ``UniformOutput=false``)."""


def cellfunUniOut(fun, x):
    """Apply ``fun`` to each element of the iterable ``x`` and return a list."""
    return [fun(xi) for xi in x]
