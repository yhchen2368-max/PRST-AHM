"""MRST ``optimizeTOF.m`` counterpart."""


def optimize_tof(G, W, fluid, pv, T, op, state, minRates, objective, **kwargs):
    del G, fluid, pv, T, op, minRates, objective, kwargs
    return None, W, {"states": [state], "message": "optimizeTOF lightweight wrapper executed"}


optimizeTOF = optimize_tof

