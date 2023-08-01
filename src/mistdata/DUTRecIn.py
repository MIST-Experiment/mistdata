from datetime import datetime

import h5py
import numpy as np

from .Thermistors import Thermistors
from .utils import add_sort_time_pair, hdfdt2dtlist, dtlist2strlist, ds2np


class DUTRecIn:
    """
    Class containing data for receiver input as a device under test.
    """
    def __init__(
            self,
            therm=Thermistors(),
            freq: np.ndarray = None,
            freq_time: datetime = None,
            open_: np.ndarray = None,
            open_time: datetime = None,
            short: np.ndarray = None,
            short_time: datetime = None,
            match: np.ndarray = None,
            match_time: datetime = None,
            antenna: np.ndarray = None,
            antenna_time: datetime = None,
            ambient: np.ndarray = None,
            ambient_time: datetime = None,
            noise_source: np.ndarray = None,
            noise_source_time: datetime = None,
    ):
        self.therm = therm

        self.s11_freq = freq
        self.s11_open = open_
        self.s11_short = short
        self.s11_match = match
        self.s11_antenna = antenna
        self.s11_ambient = ambient
        self.s11_noise_source = noise_source

        self.s11_freq_time = freq_time
        self.s11_open_time = open_time
        self.s11_short_time = short_time
        self.s11_match_time = match_time
        self.s11_antenna_time = antenna_time
        self.s11_ambient_time = ambient_time
        self.s11_noise_source_time = noise_source_time

    def write_self_to_file(self, file: h5py.File):
        """
        Write self to a HDF5 file.

        :param file: an open HDF5 file handle
        """
        grp = file.create_group("dutrecin")

        grp.create_dataset("s11_freq", data=self.s11_freq)
        grp.create_dataset("s11_open", data=self.s11_open)
        grp.create_dataset("s11_short", data=self.s11_short)
        grp.create_dataset("s11_match", data=self.s11_match)
        grp.create_dataset("s11_antenna", data=self.s11_antenna)
        grp.create_dataset("s11_ambient", data=self.s11_ambient)
        grp.create_dataset("s11_noise_source", data=self.s11_noise_source)

        grp.create_dataset("s11_freq_time", data=dtlist2strlist(self.s11_freq_time))
        grp.create_dataset("s11_open_time", data=dtlist2strlist(self.s11_open_time))
        grp.create_dataset("s11_short_time", data=dtlist2strlist(self.s11_short_time))
        grp.create_dataset("s11_match_time", data=dtlist2strlist(self.s11_match_time))
        grp.create_dataset("s11_antenna_time", data=dtlist2strlist(self.s11_antenna_time))
        grp.create_dataset("s11_ambient_time", data=dtlist2strlist(self.s11_ambient_time))
        grp.create_dataset("s11_noise_source_time", data=dtlist2strlist(self.s11_noise_source_time))

        grp_therm = grp.create_group("dutrecin_therm")
        self.therm.write_self_to_group(grp_therm)

    @classmethod
    def read_self_from_file(cls, file: h5py.File):
        """
        :param file: The h5py.File object representing the HDF5 file containing the DUTRecIn data.
        :return: An instance of the DUTRecIn class with data read from the file.
        """
        obj = cls()
        grp = file['dutrecin']
        obj.s11_freq = ds2np(grp.get("s11_freq"))
        obj.s11_open = ds2np(grp.get("s11_open"))
        obj.s11_short = ds2np(grp.get("s11_short"))
        obj.s11_match = ds2np(grp.get("s11_match"))
        obj.s11_antenna = ds2np(grp.get("s11_antenna"))
        obj.s11_ambient = ds2np(grp.get("s11_ambient"))
        obj.s11_noise_source = ds2np(grp.get("s11_noise_source"))

        obj.s11_freq_time = hdfdt2dtlist(grp.get("s11_freq_time"))
        obj.s11_open_time = hdfdt2dtlist(grp.get("s11_open_time"))
        obj.s11_short_time = hdfdt2dtlist(grp.get("s11_short_time"))
        obj.s11_match_time = hdfdt2dtlist(grp.get("s11_match_time"))
        obj.s11_antenna_time = hdfdt2dtlist(grp.get("s11_antenna_time"))
        obj.s11_ambient_time = hdfdt2dtlist(grp.get("s11_ambient_time"))
        obj.s11_noise_source_time = hdfdt2dtlist(grp.get("s11_noise_source_time"))

        obj.therm = Thermistors.read_self_from_group(grp["dutrecin_therm"])
        return obj

    def __add__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self

        if not isinstance(other, DUTRecIn):
            raise ValueError("Addition defined only for objects of the same class")
        return DUTRecIn(
            self.therm + other.therm,
            *add_sort_time_pair(
                self.s11_freq, self.s11_freq_time, other.s11_freq, other.s11_freq_time
            ),
            *add_sort_time_pair(
                self.s11_open, self.s11_open_time, other.s11_open, other.s11_open_time
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
                self.s11_antenna,
                self.s11_antenna_time,
                other.s11_antenna,
                other.s11_antenna_time,
            ),
            *add_sort_time_pair(
                self.s11_ambient,
                self.s11_ambient_time,
                other.s11_ambient,
                other.s11_ambient_time,
            ),
            *add_sort_time_pair(
                self.s11_noise_source,
                self.s11_noise_source_time,
                other.s11_noise_source,
                other.s11_noise_source_time,
            ),
        )

    def __radd__(self, other):
        return self + other

    def __eq__(self, other):
        if self.s11_open.shape != other.s11_open.shape:
            return False
        if (
                self.therm == other.therm
                and np.isclose(self.s11_freq, other.s11_freq).all()
                and np.isclose(self.s11_open, other.s11_open).all()
                and np.isclose(self.s11_short, other.s11_short).all()
                and np.isclose(self.s11_match, other.s11_match).all()
                and np.isclose(self.s11_antenna, other.s11_antenna).all()
                and np.isclose(self.s11_ambient, other.s11_ambient).all()
                and np.isclose(self.s11_noise_source, other.s11_noise_source).all()
                and np.asarray(self.s11_freq_time == other.s11_freq_time).all()
                and np.asarray(self.s11_open_time == other.s11_open_time).all()
                and np.asarray(self.s11_short_time == other.s11_short_time).all()
                and np.asarray(self.s11_match_time == other.s11_match_time).all()
                and np.asarray(self.s11_antenna_time == other.s11_antenna_time).all()
                and np.asarray(self.s11_ambient_time == other.s11_ambient_time).all()
                and np.asarray(self.s11_noise_source_time == other.s11_noise_source_time).all()
        ):
            return True
        return False
