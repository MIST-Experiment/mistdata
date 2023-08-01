import os
import warnings
from datetime import timedelta
from typing import Sequence

import h5py
import matplotlib.cm as mplcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .Spectrum import Spectrum
from .Thermistors import Thermistors
from .utils import ds2np

LSTH = 23.9344696  # Sidereal day in hours


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def _first_none_ind(arr: Sequence):
    for i, item in enumerate(arr):
        if item is None:
            return i
    raise RuntimeError("No more None-s left in the array")


class LSTBinnedTherm:
    """A class for binning thermistor data.

    Args:
        therm (Thermistors): The Thermistors object containing the raw data.
        nbins (int): Number of bins to use for binning (default is 720).

    Attributes:
        lna (ndarray): Binned LNA data.
        vna (ndarray): Binned VNA data.
        ambient (ndarray): Binned ambient data.
        backend (ndarray): Binned backend data.
        temp_count (ndarray): Number of temperature readings in each bin.
        bin_time (ndarray): Time intervals for each bin.

    """
    def __init__(self, therm: Thermistors, nbins: int = 720):
        ndays_lst = np.sum((therm.lst < np.roll(therm.lst, 1)[0:]))
        ndays_time = int((therm.time[-1] - therm.time[0]).total_seconds() / 3600 / LSTH) + 2
        ndays = max(ndays_lst, ndays_time)
        self.lna = np.empty(ndays * nbins)
        self.vna = np.empty(ndays * nbins)
        self.ambient = np.empty(ndays * nbins)
        self.backend = np.empty(ndays * nbins)
        self.temp_count = np.empty((ndays * nbins), dtype=int)
        self.bin_time = np.linspace(0, LSTH, nbins + 1, endpoint=True)
        bin_delta = timedelta(hours=LSTH / nbins)

        flat_bin_idx = np.searchsorted(self.bin_time[1:], therm.lst[0])
        left_time = therm.time[0] - timedelta(hours=therm.lst[0] - self.bin_time[flat_bin_idx])
        right_time = therm.time[0] + timedelta(hours=self.bin_time[flat_bin_idx + 1] - therm.lst[0])
        self.lna[:flat_bin_idx] = np.nan
        self.vna[:flat_bin_idx] = np.nan
        self.ambient[:flat_bin_idx] = np.nan
        self.backend[:flat_bin_idx] = np.nan
        self.temp_count[:flat_bin_idx] = 0

        bin_start_ind, bin_end_ind = 0, 0
        lna = np.where(therm.lna == 0, np.nan, therm.lna)
        vna_load = np.where(therm.vna_load == 0, np.nan, therm.vna_load)
        ambient_load = np.where(therm.ambient_load == 0, np.nan, therm.ambient_load)
        back_end = np.where(therm.back_end == 0, np.nan, therm.back_end)
        with tqdm(total=len(therm.lst)) as pbar:
            while bin_end_ind < len(therm.lst):
                bin_start_ind = np.searchsorted(therm.time, left_time)
                bin_end_ind = np.searchsorted(therm.time, right_time)
                if bin_start_ind != bin_end_ind:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        self.lna[flat_bin_idx] = np.nanmean(lna[bin_start_ind:bin_end_ind])
                        self.vna[flat_bin_idx] = np.nanmean(vna_load[bin_start_ind:bin_end_ind])
                        self.ambient[flat_bin_idx] = np.nanmean(ambient_load[bin_start_ind:bin_end_ind])
                        self.backend[flat_bin_idx] = np.nanmean(back_end[bin_start_ind:bin_end_ind])
                    self.temp_count[flat_bin_idx] = lna[bin_start_ind:bin_end_ind].shape[0]
                else:
                    self.lna[flat_bin_idx] = np.nan
                    self.vna[flat_bin_idx] = np.nan
                    self.ambient[flat_bin_idx] = np.nan
                    self.backend[flat_bin_idx] = np.nan
                    self.temp_count[flat_bin_idx] = 0
                flat_bin_idx += 1
                left_time = right_time
                right_time += bin_delta
                pbar.update(bin_end_ind - bin_start_ind)

        self.lna[flat_bin_idx:] = np.nan
        self.vna[flat_bin_idx:] = np.nan
        self.ambient[flat_bin_idx:] = np.nan
        self.backend[flat_bin_idx:] = np.nan
        self.temp_count[flat_bin_idx:] = 0

        self.lna = self.lna.reshape((ndays, nbins))
        self.vna = self.vna.reshape((ndays, nbins))
        self.ambient = self.ambient.reshape((ndays, nbins))
        self.backend = self.backend.reshape((ndays, nbins))
        self.temp_count = self.temp_count.reshape((ndays, nbins))

    def save(self, saveto: str = "binned_therm.bmist"):
        """
        Save the data to HDF file.

        :param saveto: Path and name of the file.
        """
        head, tail = os.path.split(saveto)
        if not os.path.exists(head) and len(head) > 0:
            os.makedirs(head)
        if not saveto.endswith(".bmist"):
            saveto += ".bmist"
        file = h5py.File(saveto, mode="w")
        file.create_dataset("bin_time", data=self.bin_time)
        file.create_dataset("lna", data=self.lna)
        file.create_dataset("vna", data=self.vna)
        file.create_dataset("ambient", data=self.ambient)
        file.create_dataset("backend", data=self.backend)
        file.create_dataset("temp_count", data=self.temp_count)
        file.close()

    @classmethod
    def load(cls, path: str):
        """
        Load data from file.

        :param path: Path to a file (file extension is not required).
        :return: :class:`LSTBinnedSpectra` recovered from a file.
        """
        if not path.endswith(".bmist"):
            path += ".bmist"
        obj = cls.__new__(cls)
        with h5py.File(path, mode="r") as file:
            obj.bin_time = ds2np(file.get("bin_time"))
            obj.lna = ds2np(file.get("lna"))
            obj.vna = ds2np(file.get("vna"))
            obj.ambient = ds2np(file.get("ambient"))
            obj.backend = ds2np(file.get("backend"))
            obj.temp_count = ds2np(file.get("temp_count"))
        return obj

    def plot_bin(self, data, plotdays=None, ylim=None):
        ndays = self.temp_count.shape[0]
        cm = plt.get_cmap('tab20b')
        cNorm = mcolors.Normalize(vmin=0, vmax=ndays - 1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        days = np.arange(ndays) + 1
        plotdays = plotdays or days
        fig = plt.figure(figsize=(10, 4))
        for i, daydata in enumerate(data):
            if i + 1 not in plotdays:
                continue
            ncolor = days[i] - 1
            color = scalarMap.to_rgba(ncolor)
            ls = '-'
            if i == 1:
                color = 'black'
                ls = '--'
            if np.sum(np.isnan(daydata)) > len(daydata)*0.95:
                continue
            plt.plot(self.bin_time[:-1], daydata, lw=2.5, label=f"{days[i]} day", color=color, ls=ls)
        if ylim is not None:
            plt.ylim(ylim)
        leg = plt.legend(loc='lower right')
        for i, line in enumerate(leg.get_lines()):
            if not i == 1:
                line.set_linewidth(4.0)
        # plt.title(f"LST={lst:04.2f} h")
        plt.xlabel("LST, [h]")
        plt.ylabel(r"T, [K]")


class LSTBinnedSpectra:
    """
    This class represents a container for binned spectra data.

    Methods:
        save(saveto: str = "binned_data.bmist") -> None:
            Save the data to HDF file.

        load(path: str) -> LSTBinnedSpectra:
            Load data from file.

        plot_occupancy(x0=0, x1=LSTH) -> None:
            Plot the occupancy of bins.

        plot_bin(lst: float, xlim=(25, 125), ylim=None, percent=False) -> None:
            Plot the bin with the given LST.

    Attributes:
        bin_time (ndarray): The bin boundaries in LST.
        freq (ndarray): The frequencies of the spectra.
        temp (ndarray): The binned spectra temperatures.
        temp_std (ndarray): The standard deviation of the binned spectra temperatures.
        temp_count (ndarray): The number of spectra in each bin.

    """
    def __init__(self, spec: Spectrum, nbins: int = 720):
        nfreq = spec.freq.size
        ndays_lst = np.sum((spec.lst < np.roll(spec.lst, 1)[0:])) + 1
        ndays_time = int((spec.time_end - spec.time_start).total_seconds() / 3600 / LSTH) + 2
        ndays = max(ndays_lst, ndays_time)
        self.temp = np.empty((ndays * nbins, nfreq))
        self.temp_std = np.empty((ndays * nbins, nfreq))
        self.temp_count = np.empty((ndays * nbins), dtype=int)
        self.bin_time = np.linspace(0, LSTH, nbins + 1, endpoint=True)
        self.freq = spec.freq
        bin_delta = timedelta(hours=LSTH / nbins)

        flat_bin_idx = np.searchsorted(self.bin_time[1:], spec.lst[0])
        left_time = spec.time[0] - timedelta(hours=spec.lst[0] - self.bin_time[flat_bin_idx])
        right_time = spec.time[0] + timedelta(hours=self.bin_time[flat_bin_idx + 1] - spec.lst[0])
        self.temp[:flat_bin_idx, :] = np.nan
        self.temp_std[:flat_bin_idx, :] = np.nan
        self.temp_count[:flat_bin_idx] = 0

        bin_start_ind, bin_end_ind = 0, 0
        with tqdm(total=len(spec.lst)) as pbar:
            while bin_end_ind < len(spec.lst):
                bin_start_ind = np.searchsorted(spec.time, left_time)
                bin_end_ind = np.searchsorted(spec.time, right_time)
                if bin_start_ind != bin_end_ind:
                    bin_specs = spec.t_antenna[bin_start_ind: bin_end_ind]
                    self.temp[flat_bin_idx, :] = np.average(bin_specs, axis=0)
                    self.temp_std[flat_bin_idx, :] = np.std(bin_specs, axis=0)
                    self.temp_count[flat_bin_idx] = bin_specs.shape[0]
                else:
                    self.temp[flat_bin_idx, :] = np.nan
                    self.temp_std[flat_bin_idx, :] = np.nan
                    self.temp_count[flat_bin_idx] = 0
                flat_bin_idx += 1
                left_time = right_time
                right_time += bin_delta
                pbar.update(bin_end_ind - bin_start_ind)

        self.temp[flat_bin_idx:, :] = np.nan
        self.temp_std[flat_bin_idx:, :] = np.nan
        self.temp_count[flat_bin_idx:] = 0

        self.temp = self.temp.reshape((ndays, nbins, nfreq))
        self.temp_std = self.temp_std.reshape((ndays, nbins, nfreq))
        self.temp_count = self.temp_count.reshape((ndays, nbins))

    def save(self, saveto: str = "binned_data.bmist"):
        """
        Save the data to HDF file.

        :param saveto: Path and name of the file.
        """
        head, tail = os.path.split(saveto)
        if not os.path.exists(head) and len(head) > 0:
            os.makedirs(head)
        if not saveto.endswith(".bmist"):
            saveto += ".bmist"
        file = h5py.File(saveto, mode="w")
        file.create_dataset("bin_time", data=self.bin_time)
        file.create_dataset("freq", data=self.freq)
        file.create_dataset("temp", data=self.temp)
        file.create_dataset("temp_std", data=self.temp_std)
        file.create_dataset("temp_count", data=self.temp_count)
        file.close()

    @classmethod
    def load(cls, path: str):
        """
        Load data from file.

        :param path: Path to a file (file extension is not required).
        :return: :class:`LSTBinnedSpectra` recovered from a file.
        """
        if not path.endswith(".bmist"):
            path += ".bmist"
        obj = cls.__new__(cls)
        with h5py.File(path, mode="r") as file:
            obj.bin_time = ds2np(file.get("bin_time"))
            obj.freq = ds2np(file.get("freq"))
            obj.temp = ds2np(file.get("temp"))
            obj.temp_std = ds2np(file.get("temp_std"))
            obj.temp_count = ds2np(file.get("temp_count"))
        return obj

    def plot_occupancy(self, x0=0, x1=LSTH):
        """
            Plots the occupancy of bins and temperature counts for each day in the LSTBinnedSpectra object.

            :param x0: The lower limit of the x-axis range for the plot. Default is 0.
            :param x1: The upper limit of the x-axis range for the plot. Default is LSTH.
            :return: None
        """
        ndays = self.temp.shape[0]
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=(max(int(ndays / 2), 1), max(ndays - 1, 1)),
                               figsize=(10, 5))

        steps = np.sum(self.temp_count > 0, axis=0)
        step_x = self.bin_time[:-1] + (self.bin_time[1] - self.bin_time[0]) / 2
        # step_y_ticks = np.arange(0, steps.max()+1, step=int(ndays//6))
        # ax[0].set_yticks(step_y_ticks)
        ax[0].bar(step_x, steps, width=self.bin_time[1] - self.bin_time[0])
        ax[0].set_ylabel("Bins available")
        ax[0].set_ylim((np.max(np.min(steps) - 1, 0), np.max(steps) + 1))
        ax[0].xaxis.grid(True, ls=':', c='k', alpha=0.5)

        # cmap1 = plt.cm.get_cmap('Greys', np.max(self.temp_count))
        cmap = plt.cm.get_cmap('PiYG')
        cmap1 = truncate_colormap(cmap, 0.5, 1)
        map_ticks = np.arange(0, ndays + 1)
        lst_ticks = np.arange(0, 24)
        im = ax[1].imshow(self.temp_count[::-1], interpolation='none', aspect='auto', cmap=cmap1,
                          extent=[0, self.bin_time[-1], 0, ndays])

        ax[1].set_yticks(map_ticks[1:] - 0.5, map_ticks[1:])
        ax[1].set_xticks(lst_ticks[1:])
        ax[1].set_ylabel("Day number")
        ax[1].set_xlabel("LST, [h]")
        ax[1].xaxis.grid(True, ls=':', c='k', alpha=0.5)
        ax[1].set_xlim((x0, x1))

    def plot_bin(self, lst: float, xlim=(25, 125), ylim=None, percent=False):
        ndays = self.temp.shape[0]
        cm = plt.get_cmap('tab20b')
        cNorm = mcolors.Normalize(vmin=0, vmax=ndays - 1)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

        bin_ind = np.searchsorted(self.bin_time[1:], lst)
        specs = []
        days = []
        for i, spec in enumerate(self.temp[:, bin_ind, :]):
            if not np.all(np.isnan(spec)):
                specs.append(spec)
                days.append(i + 1)
        days = np.asarray(days)
        specs = np.asarray(specs)
        ref_spec_ind = np.argwhere(days == 2)
        ref_spec = specs[ref_spec_ind].reshape(-1)

        ind_lim = np.searchsorted(self.freq, xlim)
        freq_adj = self.freq[ind_lim[0]:ind_lim[1]]
        for i, spec in enumerate(specs):
            if i == ref_spec_ind:
                plt.plot(freq_adj, freq_adj * 0, lw=2, ls='--', c='k', label=f"{2} day")
                continue
            spec = spec - ref_spec
            if percent:
                spec = spec / ref_spec * 100
            ncolor = days[i] - 1
            color = scalarMap.to_rgba(ncolor)
            plt.plot(freq_adj, spec[ind_lim[0]:ind_lim[1]], lw=1, label=f"{days[i]} day",
                     color=color)
        if ylim is not None:
            plt.ylim(ylim)
        leg = plt.legend(loc='lower right')
        for i, line in enumerate(leg.get_lines()):
            if i != ref_spec_ind:
                line.set_linewidth(4.0)
        plt.title(f"LST={lst:04.2f} h")
        plt.xlabel("Frequency, [MHz]")
        if percent:
            plt.ylabel(r"$\Delta T$, [%]")
        else:
            plt.ylabel(r"$\Delta T$, [K]")
