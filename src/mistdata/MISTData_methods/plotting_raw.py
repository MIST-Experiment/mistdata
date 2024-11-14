from __future__ import annotations

import matplotlib
import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('seaborn')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def s2db(arr):
    return 20. * np.log10(np.abs(arr))


def s2phi(arr):
    return np.angle(arr) / np.pi


def _plot_discont(ax: matplotlib.axes, x: np.ndarray, y: np.ndarray, color: str, label: str | None = None):
    cx, cy = x.copy(), y.copy()
    dpos = np.where(np.abs(np.diff(y)) >= 1.9)[0]
    # dpos = np.concatenate([dpos, dpos+1])
    cx[dpos] = np.nan
    cy[dpos] = np.nan
    ax.plot(cx, cy, color=color, label=label)
    # ax.scatter(x[dpos], y[dpos], color='w', edgecolor=color, lw=2, s=20)
    ax.set_ylim([0, 1])


def plot_raw_gamma_0dbm(self):
    fig, axs = plt.subplots(3, 1, figsize=(8.25, 12), sharex="all")
    freq = self.dut_recin.s11_freq

    fig.suptitle(r"Raw $\Gamma$ at $0 \; dBm$", size=20)
    fig.tight_layout()

    axs[0].plot(freq, s2db(self.dut_recin.s11_antenna), c=colors[0], label=r'$\Gamma_{antenna}$')
    axs[0].legend(fontsize=14)
    axs[0].set_ylabel(r"$\Gamma$ [dB]", size=12)

    axs[1].plot(freq, s2db(self.dut_recin.s11_open), c=colors[1], label=r'$\Gamma_{open}$')
    axs[1].plot(freq, s2db(self.dut_recin.s11_short), c=colors[2], label=r'$\Gamma_{short}$')
    axs[1].legend(fontsize=14)
    axs[1].set_ylabel(r"$\Gamma$ [dB]", size=12)

    axs[2].plot(freq, s2db(self.dut_recin.s11_ambient), c=colors[3], label=r'$\Gamma_{ambient}$')
    axs[2].plot(freq, s2db(self.dut_recin.s11_match), c=colors[4], label=r'$\Gamma_{match}$')
    axs[2].plot(freq, s2db(self.dut_recin.s11_noise_source), c=colors[5], label=r'$\Gamma_{NS}$')

    axs[2].legend(fontsize=14)
    axs[2].set_ylabel(r"$\Gamma$ [dB]", size=12)
    axs[2].set_xlabel(r"Frequency [MHz]", size=12)
    axs[2].set_xlim([0, np.max(freq) + 2])
    return fig


def plot_raw_phases_0dbm(self):
    fig = plt.figure(figsize=(8.25, 4))
    freq = self.dut_recin.s11_freq

    fig.suptitle(r"Raw $\varphi$ at $0 \; dBm$", size=20)
    fig.tight_layout()

    # _plot_discont(axs[0], freq, s2phi(self.dut_recin.s11_antenna), color=colors[0], label=r'$\varphi_{antenna}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_recin.s11_antenna))), color=colors[0],
             label=r'$\varphi_{antenna}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_recin.s11_open))), color=colors[1], label=r'$\varphi_{open}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_recin.s11_short))), color=colors[2],
             label=r'$\varphi_{short}$')
    plt.plot(freq, 180 / np.pi * (np.angle(self.dut_recin.s11_ambient)), color=colors[3], label=r'$\varphi_{ambient}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_recin.s11_match))), color=colors[4],
             label=r'$\varphi_{match}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_recin.s11_noise_source))), color=colors[5],
             label=r'$\varphi_{NS}$')
    plt.legend(fontsize=14)
    plt.ylabel(r"$\varphi [deg]$", size=12)
    plt.xlabel(r"Frequency [MHz]", size=12)
    plt.xlim([0, np.max(freq) + 2])
    return fig


def plot_raw_gamma_40dbm(self):
    fig, axs = plt.subplots(2, 1, figsize=(8.25, 6), sharex="all")
    freq = self.dut_lna.s11_freq

    fig.suptitle(r"Raw $\Gamma$ at $-40 \; dBm$", size=20)
    fig.tight_layout()

    axs[0].plot(freq, s2db(self.dut_lna.s11_lna), c=colors[0], label=r'$\Gamma_{LNA}$')
    axs[0].plot(freq, s2db(self.dut_lna.s11_match), c=colors[4], label=r'$\Gamma_{match}$')
    axs[0].legend(fontsize=14)
    axs[0].set_ylabel(r"$\Gamma$ [dB]", size=12)

    axs[1].plot(freq, s2db(self.dut_lna.s11_open), c=colors[1], label=r'$\Gamma_{open}$')
    axs[1].plot(freq, s2db(self.dut_lna.s11_short), c=colors[2], label=r'$\Gamma_{short}$')
    axs[1].legend(fontsize=14)
    axs[1].set_ylabel(r"$\Gamma$ [dB]", size=12)
    axs[1].set_xlabel(r"Frequency [MHz]", size=12)

    axs[1].set_xlim([0, np.max(freq) + 2])
    return fig


def plot_raw_phases_40dbm(self):
    fig = plt.figure(figsize=(8.25, 4))
    freq = self.dut_lna.s11_freq

    fig.suptitle(r"Raw $\varphi$ at $-40 \; dBm$", size=20)
    fig.tight_layout()

    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_lna.s11_lna))), color=colors[0], label=r'$\varphi_{LNA}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_lna.s11_match))), color=colors[4], label=r'$\varphi_{match}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_lna.s11_open))), color=colors[1], label=r'$\varphi_{open}$')
    plt.plot(freq, np.rad2deg(np.unwrap(np.angle(self.dut_lna.s11_short))), color=colors[2], label=r'$\varphi_{short}$')
    plt.legend(fontsize=14)
    plt.ylabel(r"$\varphi$, [deg]", size=12)
    plt.xlabel(r"Frequency [MHz]", size=12)

    plt.xlim([0, np.max(freq) + 2])

    return fig


def plot_raw_thermistors(self):
    fig = plt.figure(figsize=(8.25, 6))
    time = self.spec.therm.time
    lna = self.spec.therm.lna
    vna_load = self.spec.therm.vna_load
    ambient_load = self.spec.therm.ambient_load
    back_end = self.spec.therm.back_end

    plt.plot(time[lna > 0], lna[lna > 0], label="LNA")
    plt.plot(time[vna_load > 0], vna_load[vna_load > 0], label="VNA load")
    plt.plot(time[ambient_load > 0], ambient_load[ambient_load > 0], label="Ambient load")
    plt.plot(time[back_end > 0], back_end[back_end > 0], label="Back-end")

    if np.sum(lna > 0) > 0:
        lna_zi = np.argwhere(lna == 0).flatten()
    else:
        lna_zi = np.array([])
    if np.sum(ambient_load > 0) > 0:
        ambient_zi = np.argwhere(ambient_load == 0).flatten()
    else:
        ambient_zi = np.array([])
    if np.sum(vna_load > 0) > 0:
        vna_zi = np.argwhere(vna_load == 0).flatten()
    else:
        vna_zi = np.array([])
    if np.sum(back_end > 0) > 0:
        back_zi = np.argwhere(back_end == 0).flatten()
    else:
        back_zi = np.array([])

    zi = np.unique(np.concatenate((vna_zi, ambient_zi, lna_zi, back_zi)))

    for i in zi:
        i = int(i)
        x1 = time[i - 1] + (time[i] - time[i - 1]) / 2 if i > 0 else time[i]
        x2 = time[i] + (time[i + 1] - time[i]) / 2 if i < len(time) - 1 else time[i]
        plt.axvspan(x1, x2, color='black', linewidth=0, alpha=0.5, label="Missing data" if i == zi[0] else None)

    fmt = mdates.DateFormatter("%H:%M")
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.legend(fontsize=14, frameon=True, facecolor='w')
    fig.suptitle(f"Thermistors during spectra recording starting from   \n{str(self.spec.therm.time[0])}", size=20)
    fig.tight_layout()
    plt.xlabel(r"Time, [hh:mm]", size=12)
    plt.ylabel(r"Temperature, [K]", size=12)
    return fig
