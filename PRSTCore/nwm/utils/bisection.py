"""Port of MRST ``bisection``: root finding by the bisection method."""


def bisection(f, bot, top, tol):
    """Find a root of ``f`` in ``[bot, top]`` by bisection.

    Returns ``(x, fx)``.
    """
    if f(bot) * f(top) > 0:
        raise ValueError('Invalid boundary')
    x = (bot + top) / 2
    fx = f(x)
    while abs(top - bot) > tol:
        x = (bot + top) / 2
        fx = f(x)
        if fx == 0:
            bot = x
            top = x
        elif f(bot) * fx < 0:
            top = x
        else:
            bot = x
    return x, fx
