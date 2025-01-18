import h5py
import numpy as np

from .Thermistors import Thermistors
from .utils import (
    add_sort_time_pair,
    combine_times,
    hdfdt2dtlist,
    dtlist2strlist,
    ds2np,
)


class DUTLNA:
    """
    Class containing data for LNA as a device under test.
    """

    def __init__(
        self,
        therm=Thermistors(),
        freq=None,
        freq_time=None,
        open_=None,
        open_time=None,
        short=None,
        short_time=None,
        match=None,
        match_time=None,
        lna=None,
        lna_time=None,
    ):
        self.therm = therm

        self.s11_freq = freq
        self.s11_open = open_
        self.s11_short = short
        self.s11_match = match
        self.s11_lna = lna

        self.s11_freq_time = freq_time
        self.s11_open_time = open_time
        self.s11_short_time = short_time
        self.s11_match_time = match_time
        self.s11_lna_time = lna_time

    def write_self_to_file(self, file: h5py.File):
        """
        Write the attributes of the DUTLNA object to an HDF5 file.

        :param file: The HDF5 file to write to.
        :return: None
        """
        grp = file.create_group("dutlna")

        grp.create_dataset("s11_freq", data=self.s11_freq)
        grp.create_dataset("s11_open", data=self.s11_open)
        grp.create_dataset("s11_short", data=self.s11_short)
        grp.create_dataset("s11_match", data=self.s11_match)
        grp.create_dataset("s11_lna", data=self.s11_lna)

        grp.create_dataset(
            "s11_freq_time", data=dtlist2strlist(self.s11_freq_time)
        )
        grp.create_dataset(
            "s11_open_time", data=dtlist2strlist(self.s11_open_time)
        )
        grp.create_dataset(
            "s11_short_time", data=dtlist2strlist(self.s11_short_time)
        )
        grp.create_dataset(
            "s11_match_time", data=dtlist2strlist(self.s11_match_time)
        )
        grp.create_dataset(
            "s11_lna_time", data=dtlist2strlist(self.s11_lna_time)
        )

        grp_therm = grp.create_group("dutlna_therm")
        self.therm.write_self_to_group(grp_therm)

    @classmethod
    def read_self_from_file(cls, file: h5py.File):
        """
        Read DUTLNA object from an HDF5 file.

        :param cls: The DUTLNA class.
        :param file: The HDF5 file object.
        :return: The DUTLNA object.
        """
        obj = cls()
        grp = file["dutlna"]
        obj.s11_freq = ds2np(grp.get("s11_freq"))
        obj.s11_open = ds2np(grp.get("s11_open"))
        obj.s11_short = ds2np(grp.get("s11_short"))
        obj.s11_match = ds2np(grp.get("s11_match"))
        obj.s11_lna = ds2np(grp.get("s11_lna"))

        obj.s11_freq_time = hdfdt2dtlist(grp.get("s11_freq_time"))
        obj.s11_open_time = hdfdt2dtlist(grp.get("s11_open_time"))
        obj.s11_short_time = hdfdt2dtlist(grp.get("s11_short_time"))
        obj.s11_match_time = hdfdt2dtlist(grp.get("s11_match_time"))
        obj.s11_lna_time = hdfdt2dtlist(grp.get("s11_lna_time"))

        obj.therm = Thermistors.read_self_from_group(grp["dutlna_therm"])
        return obj

    def cut_freq(self, freq_min=None, freq_max=None):
        """
        Cut the frequency range of the DUTLNA object.

        :param freq_min: The minimum frequency in MHz.
        :param freq_max: The maximum frequency in MHz.
        :return: None
        """
        if freq_min is None:
            freq_min = np.min(self.s11_freq)
        if freq_max is None:
            freq_max = np.max(self.s11_freq)
        mask = (self.s11_freq >= freq_min) & (self.s11_freq <= freq_max)
        self.s11_freq = self.s11_freq[mask]
        self.s11_open = self.s11_open[..., mask]
        self.s11_short = self.s11_short[..., mask]
        self.s11_match = self.s11_match[..., mask]
        self.s11_lna = self.s11_lna[..., mask]

    def __add__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self

        if not isinstance(other, DUTLNA):
            raise ValueError(
                "Addition defined only for objects of the same class"
            )
        if np.any(self.s11_freq != other.s11_freq):
            raise ValueError("Frequencies do not match")

        return DUTLNA(
            self.therm + other.therm,
            self.s11_freq,
            sorted(combine_times(self.s11_freq_time, other.s11_freq_time)),
            *add_sort_time_pair(
                self.s11_open,
                self.s11_open_time,
                other.s11_open,
                other.s11_open_time,
            ),
            *add_sort_time_pair(
                self.s11_short,
                self.s11_short_time,
                other.s11_short,
                other.s11_short_time,
            ),
            *add_sort_time_pair(
                self.s11_match,
                self.s11_match_time,
                other.s11_match,
                other.s11_match_time,
            ),
            *add_sort_time_pair(
                self.s11_lna,
                self.s11_lna_time,
                other.s11_lna,
                other.s11_lna_time,
            ),
        )

    def __radd__(self, other):
        return self + other

    def __eq__(self, other):
        if self.s11_open.shape != other.s11_open.shape:
            return False
        return (
            self.therm == other.therm
            and np.isclose(self.s11_freq, other.s11_freq).all()
            and np.isclose(self.s11_open, other.s11_open).all()
            and np.isclose(self.s11_short, other.s11_short).all()
            and np.isclose(self.s11_match, other.s11_match).all()
            and np.isclose(self.s11_lna, other.s11_lna).all()
            and np.all(self.s11_freq_time == other.s11_freq_time)
            and np.all(self.s11_open_time == other.s11_open_time)
            and np.all(self.s11_short_time == other.s11_short_time)
            and np.all(self.s11_match_time == other.s11_match_time)
            and np.all(self.s11_lna_time == other.s11_lna_time)
        )
