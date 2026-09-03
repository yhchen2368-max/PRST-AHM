"""MRST ``plotWellRates.m`` counterpart."""


def plot_well_rates(W, data, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return data
    ax = kwargs.get("ax")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Well rates")
    return ax


plotWellRates = plot_well_rates

