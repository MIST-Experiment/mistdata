from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


def plot_spec(x: Sequence, y: Sequence, spec: np.ndarray, lim=(1e11, 3.3e11), cmap='plasma'):
    """
    :param x: A sequence representing the x-axis values (frequency values in MHz).
    :param y: A sequence representing the y-axis values (time values).
    :param spec: A numpy ndarray representing the spectra.
    :param lim: A tuple specifying the minimum and maximum values for the colorbar limits.
                Default value is (1e11, 3.3e11).
    :param cmap: A string specifying the color map for the spectrogram. Default value is 'plasma'.

    :return: A matplotlib Figure object representing the plotted spectrogram.
    """
    if isinstance(y[0], datetime):
        timespansec: int = (y[-1] - y[0]).total_seconds()
        time_up_lim = timespansec / 60 if timespansec <= 10800 else timespansec / 3600
        y_extent = (time_up_lim, 0)
        ylabel = "Time [min]" if timespansec <= 10800 else "Time [h]"
    elif isinstance(y[0], float):
        y_extent = (y[-1], y[0])
        ylabel = "LST [h]"
    else:
        raise ValueError("Unrecognized x-axis type")

    fig = plt.figure(figsize=(8.25, 6))
    plt.imshow(
        spec,
        vmin=lim[0],
        vmax=lim[1],
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        extent=[np.min(x), np.max(x), *y_extent]
    )
    cb = plt.colorbar()
    cb.ax.tick_params(size=5)
    plt.xlabel(r"Frequency, MHz")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2, ls=':')
    return fig



def plot_spec_rows(x, spec, rows, limits=(1e11, 4e11)):
    """
    :param x: Array of x-coordinates for the plot.
    :param spec: 2D array of y-coordinates for each row.
    :param rows: Sequence of row indices to plot.
    :param limits: Tuple of lower and upper limits for the y-axis. Default is (1e11, 4e11).
    :return: The plot figure.

    This method plots the specified rows of a given spectrum data. The x-coordinates
    for the plot are provided by the `x` parameter, while the y-coordinates are determined
    by the `spec` parameter. The `spec` parameter should be a 2D array where each row
    represents a different spectrum and each column represents a different x-coordinate.

    The `rows` parameter should be a sequence of row indices that you want to plot. By
    specifying specific row indices, you can choose which spectra to plot. Each row will
    be plotted as a separate line on the plot.

    The `limits` parameter is an optional tuple that specifies the lower and upper limits
    for the y-axis. By default, the limits are set to (1e11, 4e11).

    This will plot the first, third, and fifth rows of the `spec` array against the `x` values,
    with the y-axis limited to the range between 1e10 and 1e12.
    """
    fig = plt.figure(figsize=(8.25, 6))
    for row in rows:
        plt.plot(x, spec[row, :], label=f"Row {row}", lw=1)
    plt.ylim(limits)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"PSD")
    return fig


def plot_spec_stats(x, spec):
    """
    Plot the statistics of an antenna spectrum.

    :param x: The frequencies corresponding to the spectrum data.
    :param spec: The spectrum data as a 2D numpy array.
    :return: The matplotlib Figure object containing the plot.
    """
    fig = plt.figure(figsize=(8.25, 6))
    plt.plot(x, spec.mean(axis=0), label="Average")
    plt.plot(x, spec.min(axis=0), label="Minimum")
    plt.plot(x, spec.max(axis=0), label="Maximum")
    plt.plot(x, np.median(spec, axis=0), label="Median")

    plt.legend()
    plt.ylim(1e11, 4e11)
    plt.ylabel(r"PSD", size=12)
    plt.xlabel(r"Frequency [MHz]", size=12)
    plt.title("Antenna spectrum characteristics", size=16)
    return fig

