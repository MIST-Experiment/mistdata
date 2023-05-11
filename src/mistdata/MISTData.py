import copy
import itertools
import os
from multiprocessing import Pool
from typing import List

import h5py as h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.ndimage import median_filter
from tqdm import tqdm

from .DUTLNA import DUTLNA
from .DUTRecIn import DUTRecIn
from .Spectrum import Spectrum


def par_func(pars):
    return pars[0](*pars[1:])


class MISTData:
    from .MISTData_methods.read_raw_data import read_raw
    read_raw = classmethod(read_raw)

    from .MISTData_methods.plotting_psd import plot_psd_antenna

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
        self_slice = copy.copy(self)
        self_slice.spec = self.spec[item.start:item.stop:item.step]
        return MISTData(
            dut_recin=self.dut_recin,
            dut_lna=self.dut_lna,
            spec=self.spec[item.start:item.stop:item.step],
        )

    def save(self, saveto: str = "data.mist"):
        """
        Save the model to HDF file.

        :param saveto: Path and name of the file.
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
        :return: :class:`IonModel` recovered from a file.
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
    def from_raw_list(obj_list):
        base = obj_list[0]
        for i in range(1, len(obj_list)):
            base += obj_list[i]
        return base

    @classmethod
    def read_raw_many(cls, files: List[str], nproc=None):
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
