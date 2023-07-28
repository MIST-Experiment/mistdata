import copy
import gzip as gz
import itertools
import os
from datetime import datetime
from multiprocessing import Pool
from typing import List

import h5py as h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.ndimage import median_filter
from tqdm import tqdm

from . import Thermistors
from .DUTLNA import DUTLNA
from .DUTRecIn import DUTRecIn
from .Spectrum import Spectrum


def par_func(pars):
    return pars[0](*pars[1:])


def _extract_time(array):
    return datetime(*array[1:7].real.astype(np.int64))


def _assign_or_stack(var, array):
    if var is None:
        var = array
    else:
        var = np.vstack((var, array))
    return var


class MISTData:
    """
    The MISTData class represents the MIST data with information about DUTRecIn, DUTLNA, and Spectrum.

    Attributes:
    - dut_recin (DUTRecIn): The DUTRecIn object.
    - dut_lna (DUTLNA): The DUTLNA object.
    - spec (Spectrum): The Spectrum object.

    Methods:
    - __getitem__(self, item): Returns a new MISTData object with the specified slice of spec.
    - save(self, saveto: str = "data.mist"): Save the model to an HDF file.
    - load(cls, path: str): Load a model from a file and return the recovered MISTData object.
    - from_list(obj_list) -> "MISTData": Returns a new MISTData object by adding all the elements in the obj_list.
    - read_raw_many(cls, files: List[str], nproc=None): Read raw data from multiple files and return a MISTData object.
    - read_raw(cls, path: str): Read raw data from a file and return a MISTData object.
    - __add__(self, other): Adds two MISTData objects and returns a new MISTData object.
    - __eq__(self, other): Checks if two MISTData objects are equal.
    - plot_rfi(self, thresh=5, med_win=10, ax1_lims=(110, 114)): Plots the RFI (Radio Frequency Interference) using the
      MISTData object.
    """

    def __init__(
            self,
            dut_recin: DUTRecIn = DUTRecIn(),
            dut_lna: DUTLNA = DUTLNA(),
            spec: Spectrum = Spectrum(),
    ):
        self.dut_recin = dut_recin
        self.dut_lna = dut_lna
        self.spec = spec

    def __getitem__(self, item):
        return MISTData(
            dut_recin=self.dut_recin,
            dut_lna=self.dut_lna,
            spec=self.spec[item.start:item.stop:item.step],
        )

    def save(self, saveto: str = "data.mist"):
        """
        Save the model to HDF file.

        :param saveto: Path and name of the file.
        :return: None
        """
        head, tail = os.path.split(saveto)
        if not os.path.exists(head) and len(head) > 0:
            os.makedirs(head)
        if not saveto.endswith(".mist"):
            saveto += ".mist"

        file = h5py.File(saveto, mode="w")
        self.dut_recin.write_self_to_file(file)
        self.dut_lna.write_self_to_file(file)
        self.spec.write_self_to_file(file)
        file.close()

    @classmethod
    def load(cls, path: str):
        """
        Load a model from file.

        :param path: Path to a file (file extension is not required).
        :return: :class:`MISTData` recovered from a file.
        """
        if not path.endswith(".mist"):
            path += ".mist"
        with h5py.File(path, mode="r") as file:
            dutrecin = DUTRecIn.read_self_from_file(file)
            dutlna = DUTLNA.read_self_from_file(file)
            spec = Spectrum.read_self_from_file(file)
            obj = cls(dut_recin=dutrecin, dut_lna=dutlna, spec=spec)
        return obj

    @staticmethod
    def from_list(obj_list) -> "MISTData":
        """
        Construct a MISTData object from a list of objects.

        :param obj_list: A list of objects to be combined into a MISTData object.
        :return: A MISTData object containing the combined data from the input objects.
        """
        base = obj_list[0]
        for i in range(1, len(obj_list)):
            base += obj_list[i]
        return base

    @classmethod
    def read_raw_many(cls, files: List[str], nproc=None):
        """
        Reads multiple raw data files and returns a list of MISTData objects.

        :param files: A list of file paths to the raw data files.
        :param nproc: The number of processes to use for parallel processing. Default is None, which means no parallel
                      processing.
        :return: A list of MISTData objects.

        Example usage:

        files = ['data1.mist', 'data2.mist', 'data3.mist']
        data = MISTData.read_raw_many(files, nproc=2)
        """
        if nproc is not None:
            with Pool(processes=nproc) as pool:
                datafiles = list(
                    tqdm(
                        pool.imap(
                            par_func,
                            zip(itertools.repeat(MISTData.read_raw), files),
                        ),
                        total=len(files),
                    )
                )
        else:
            datafiles = []
            for f in tqdm(files):
                datafiles.append(MISTData.read_raw(f))
        return MISTData.from_raw_list(datafiles)

    @classmethod
    def read_raw(cls, path: str):
        """
        :class: MISTData

        The `read_raw` method reads raw data from a file and constructs an instance of the `MISTData` class.

        :param path: The path to the file containing the raw data.
        :return: An instance of the `MISTData` class.
        """
        with gz.open(path, "r") as fin:
            spec_therm = None
            spec_antenna = None
            spec_ambient = None
            spec_noise_source = None
            try:
                for line in fin:
                    line = line.decode("utf-8")
                    line = line.strip().split()
                    line_array = np.array([complex(i.replace("+-", "-")) for i in line])
                    # iteration = int(np.real(line_array[0]))
                    case = int(np.real(line_array[7]))

                    if case == 1:
                        time = _extract_time(line_array)
                        recin_therm = Thermistors(time, *line_array[[8, 9, 10, 13]].real)
                    elif case == 10:
                        recin_s11_freq = line_array[8:].real
                        recin_s11_freq_time = _extract_time(line_array)
                    elif case == 11:
                        recin_s11_open = line_array[8:]
                        recin_s11_open_time = _extract_time(line_array)
                    elif case == 12:
                        recin_s11_short = line_array[8:]
                        recin_s11_short_time = _extract_time(line_array)
                    elif case == 13:
                        recin_s11_match = line_array[8:]
                        recin_s11_match_time = _extract_time(line_array)
                    elif case == 14:
                        recin_s11_antenna = line_array[8:]
                        recin_s11_antenna_time = _extract_time(line_array)
                    elif case == 15:
                        recin_s11_ambient = line_array[8:]
                        recin_s11_ambient_time = _extract_time(line_array)
                    elif case == 16:
                        recin_s11_noise_source = line_array[8:]
                        recin_s11_noise_source_time = _extract_time(line_array)
                    elif case == 2:
                        time = _extract_time(line_array)
                        lna_therm = Thermistors(time, *line_array[8:12].real)
                    elif case == 20:
                        lna_s11_freq = line_array[8:].real
                        lna_s11_freq_time = _extract_time(line_array)
                    elif case == 21:
                        lna_s11_open = line_array[8:]
                        lna_s11_open_time = _extract_time(line_array)
                    elif case == 22:
                        lna_s11_short = line_array[8:]
                        lna_s11_short_time = _extract_time(line_array)
                    elif case == 23:
                        lna_s11_match = line_array[8:]
                        lna_s11_match_time = _extract_time(line_array)
                    elif case == 24:
                        lna_s11_lna = line_array[8:]
                        lna_s11_lna_time = _extract_time(line_array)
                    elif case == 3:
                        spec_therm = _assign_or_stack(spec_therm, line_array)
                    elif case == 30:
                        spec_freq = line_array[8:].real
                        # spec_freq_time = _extract_time(line_array)
                    elif case == 31:
                        spec_antenna = _assign_or_stack(spec_antenna, line_array)
                    elif case == 32:
                        spec_ambient = _assign_or_stack(spec_ambient, line_array)
                    elif case == 33:
                        spec_noise_source = _assign_or_stack(spec_noise_source, line_array)
            except EOFError:
                return None

            try:
                spec_therm_time = [_extract_time(arr) for arr in spec_therm]
                spec_therm_lna = spec_therm[:, 8].real
                spec_therm_vna_load = spec_therm[:, 9].real
                spec_therm_ambient_load = spec_therm[:, 10].real
                spec_therm_back_end = spec_therm[:, 13].real
                spec_t_antenna = spec_antenna[:, 8:].real
                spec_t_antenna_time = [_extract_time(arr) for arr in spec_antenna]
                spec_t_ambient = spec_ambient[:, 8:].real
                spec_t_ambient_time = [_extract_time(arr) for arr in spec_ambient]
                spec_t_noise_source = spec_noise_source[:, 8:].real
                spec_t_noise_source_time = [_extract_time(arr) for arr in spec_noise_source]
            except IndexError:
                # raise RuntimeError(f"Cannot read_raw file {path}")
                return None

            dut_recin = DUTRecIn(
                recin_therm,
                recin_s11_freq,
                recin_s11_freq_time,
                recin_s11_open,
                recin_s11_open_time,
                recin_s11_short,
                recin_s11_short_time,
                recin_s11_match,
                recin_s11_match_time,
                recin_s11_antenna,
                recin_s11_antenna_time,
                recin_s11_ambient,
                recin_s11_ambient_time,
                recin_s11_noise_source,
                recin_s11_noise_source_time,
            )
            dut_lna = DUTLNA(
                lna_therm,
                lna_s11_freq,
                lna_s11_freq_time,
                lna_s11_open,
                lna_s11_open_time,
                lna_s11_short,
                lna_s11_short_time,
                lna_s11_match,
                lna_s11_match_time,
                lna_s11_lna,
                lna_s11_lna_time,
            )
            spec_t = Thermistors(
                spec_therm_time,
                spec_therm_lna,
                spec_therm_vna_load,
                spec_therm_ambient_load,
                spec_therm_back_end,
            )
            spec = Spectrum(
                spec_t,
                spec_freq,
                spec_t_antenna,
                spec_t_antenna_time,
                spec_t_ambient,
                spec_t_ambient_time,
                spec_t_noise_source,
                spec_t_noise_source_time,
            )
        return cls(dut_recin, dut_lna, spec)

    def __add__(self, other):
        return MISTData(
            self.dut_recin + other.dut_recin,
            self.dut_lna + other.dut_lna,
            self.spec + other.spec,
        )

    def __eq__(self, other):
        if (
                self.dut_recin == other.dut_recin
                and self.dut_lna == other.dut_lna
                and self.spec == other.spec
        ):
            return True
        return False

    def plot_rfi(self, thresh=5, med_win=10, ax1_lims=(110, 114)):
        """
        Plots RFI removal steps for MISTData object. Based on https://arxiv.org/abs/2012.06521 paper.

        :param thresh: The threshold for flagging RFI, default is 5.
        :param med_win: The window size for median filter, default is 10.
        :param ax1_lims: The limits for the first subplot, default is (110, 114).

        :return: The matplotlib Figure object showing the RFI removal steps.
        """
        spec = self.spec.psd_antenna
        logdata = 10 * np.log10(spec)
        medians = np.median(logdata, axis=0)
        flattened = logdata - medians
        filtered = median_filter(flattened, [1, med_win])
        corrected = flattened - filtered
        MAD = np.median(np.abs(corrected - np.median(corrected)))
        flags = corrected - np.median(corrected) > thresh * MAD
        rfi_removed = np.ma.array(corrected, mask=flags)

        cmap = copy.copy(cm.get_cmap("YlOrRd_r"))
        cmap.set_bad("blue", 1.0)

        fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
        # =========== 1 ==========
        img = axs[0].imshow(
            logdata,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=ax1_lims[0],
            vmax=ax1_lims[1],
        )
        fig.colorbar(img, ax=axs[0], label="dB", aspect=7)
        axs[0].set_title("Raw Power Spectra")
        # ========================

        # =========== 2 ==========
        vmin2 = -1
        vmax2 = 1
        img = axs[1].imshow(
            flattened,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=vmin2,
            vmax=vmax2,
        )
        fig.colorbar(img, ax=axs[1], label="dB", aspect=7)
        axs[1].set_title("Step 1: Subtracting Channels' Medians")
        # ========================

        # =========== 3 ==========
        vmin3 = -0.1
        vmax3 = 0.1
        img = axs[2].imshow(
            corrected,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=vmin3,
            vmax=vmax3,
        )
        fig.colorbar(img, ax=axs[2], label="dB", aspect=7)
        axs[2].set_title("Step 2: Subtracting Median Filtered Autospectra")
        # ========================

        # =========== 4 ==========
        img = axs[3].imshow(
            rfi_removed,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=vmin3,
            vmax=vmax3,
        )
        fig.colorbar(img, ax=axs[3], label="dB", aspect=7)
        axs[3].set_title("Step 3: Flagging")
        axs[3].set_xlabel("Frequency (MHz)")
        # ========================

        plt.show()
        return fig
