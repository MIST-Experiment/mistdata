import numpy as np
from matplotlib import pyplot as plt

from .plotting_raw import s2db


def plot_psd_ambient(self, limits=(1e11, 1.6e11)):
    timespansec: int = (self.spec.psd_ambient_time[-1] - self.spec.psd_ambient_time[0]).total_seconds()
    time_up_lim = timespansec / 60 if timespansec <= 10800 else timespansec / 3600
    ylabel = "Minutes passed" if timespansec <= 10800 else "Hours passed"
    fig = plt.figure(figsize=(8.25, 6))
    plt.imshow(
        self.spec.psd_ambient,
        vmin=limits[0],
        vmax=limits[1],
        aspect="auto",
        interpolation="none",
        cmap='viridis',
        extent=[
            np.min(self.spec.freq),
            np.max(self.spec.freq),
            time_up_lim,
            0,
        ]
    )
    cb = plt.colorbar()
    cb.ax.tick_params(size=5)
    plt.xlabel(r"Frequency, MHz")
    plt.ylabel(ylabel)
    plt.title("PSD of ambient load starting from " + str(self.spec.psd_ambient_time[0]), fontsize=14)
    plt.grid(alpha=0.2, ls=':')
    return fig


def plot_psd_ambient_ns(self, limits=(1e11, 7e11)):
    timespansec: int = (self.spec.psd_noise_source_time[-1] - self.spec.psd_noise_source_time[0]).total_seconds()
    time_up_lim = timespansec / 60 if timespansec <= 10800 else timespansec / 3600
    ylabel = "Minutes passed" if timespansec <= 10800 else "Hours passed"
    fig = plt.figure(figsize=(8.25, 6))
    plt.imshow(
        self.spec.psd_noise_source,
        vmin=limits[0],
        vmax=limits[1],
        aspect="auto",
        interpolation="none",
        cmap='viridis',
        extent=[
            np.min(self.spec.freq),
            np.max(self.spec.freq),
            time_up_lim,
            0,
        ]
    )
    cb = plt.colorbar()
    cb.ax.tick_params(size=5)
    plt.xlabel(r"Frequency, MHz")
    plt.ylabel(ylabel)
    plt.title("PSD of ambient load + NS starting from " + str(self.spec.psd_noise_source_time[0]), fontsize=14)
    plt.grid(alpha=0.2, ls=':')
    return fig


def plot_psd_antenna(self, limits=(1e11, 3.3e11)):
    timespansec: int = (self.spec.psd_ambient_time[-1] - self.spec.psd_antenna_time[0]).total_seconds()
    time_up_lim = timespansec / 60 if timespansec <= 10800 else timespansec / 3600
    ylabel = "Minutes passed" if timespansec <= 10800 else "Hours passed"
    fig = plt.figure(figsize=(8.25, 6))
    plt.imshow(
        self.spec.psd_antenna,
        vmin=limits[0],
        vmax=limits[1],
        aspect="auto",
        interpolation="none",
        cmap='viridis',
        extent=[
            np.min(self.spec.freq),
            np.max(self.spec.freq),
            time_up_lim,
            0,
        ]
    )
    cb = plt.colorbar()
    cb.ax.tick_params(size=5)
    plt.xlabel(r"Frequency, MHz")
    plt.ylabel(ylabel)
    plt.title("PSD of antenna starting from " + str(self.spec.psd_antenna_time[0]), fontsize=14)
    plt.grid(alpha=0.2, ls=':')
    return fig


def plot_psd_antenna_rows(self, rows, limits=(1e11, 4e11), title=None):
    fig = plt.figure(figsize=(8.25, 6))
    for row in rows:
        plt.plot(self.spec.freq, self.spec.psd_antenna[row, :], label=f"Row {row}", lw=1)
    plt.ylim(limits)
    plt.title(title or "Antenna PSD profile", size=16)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"PSD")
    return fig


def plot_spec_chars(self):
    fig = plt.figure(figsize=(8.25, 6))
    freq = self.spec.freq
    plt.plot(freq, self.spec.psd_antenna.mean(axis=0), label="Average")
    plt.plot(freq, self.spec.psd_antenna.min(axis=0), label="Minimum")
    plt.plot(freq, self.spec.psd_antenna.max(axis=0), label="Maximum")
    plt.plot(freq, np.median(self.spec.psd_antenna, axis=0), label="Median")

    plt.legend()
    plt.ylim(1e11, 4e11)
    plt.ylabel(r"PSD", size=12)
    plt.xlabel(r"Frequency [MHz]", size=12)
    plt.title("Antenna spectrum characteristics", size=16)
    return fig

