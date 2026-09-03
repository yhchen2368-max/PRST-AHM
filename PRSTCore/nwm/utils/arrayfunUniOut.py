"""Port of MRST ``arrayfunUniOut`` (``arrayfun`` with ``UniformOutput=false``)."""


def arrayfunUniOut(fun, x):
    """Apply ``fun`` to each element of ``x`` and return the list of results."""
    return [fun(xi) for xi in x]
