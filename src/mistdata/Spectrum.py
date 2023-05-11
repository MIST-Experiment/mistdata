import copy

import h5py
import numpy as np

from .Thermistors import Thermistors
from .utils import add_sort_spec_pair
from .utils import hdfdt2dtlist, dtlist2strlist, ds2np


class Spectrum:
    def __init__(
            self,
            therm=Thermistors(),
            freq=None,
            psd_antenna=None,
            psd_antenna_time=None,
            psd_ambient=None,
            psd_ambient_time=None,
            psd_noise_source=None,
            psd_noise_source_time=None,
    ):
        self.therm = therm
        self.freq = freq
        self.psd_antenna = psd_antenna
        self.psd_antenna_time = psd_antenna_time
        self.psd_ambient = psd_ambient
        self.psd_ambient_time = psd_ambient_time
        self.psd_noise_source = psd_noise_source
        self.psd_noise_source_time = psd_noise_source_time
        self.t_antenna = None

    def __getitem__(self, item):
        return Spectrum(
            therm=self.therm,
            freq=self.freq,
            psd_antenna=self.psd_antenna[item.start:item.stop:item.step],
            psd_antenna_time=self.psd_antenna_time[item.start:item.stop:item.step],
            psd_ambient=self.psd_ambient[item.start:item.stop:item.step],
            psd_ambient_time=self.psd_ambient_time[item.start:item.stop:item.step],
            psd_noise_source=self.psd_noise_source[item.start:item.stop:item.step],
            psd_noise_source_time=self.psd_noise_source_time[item.start:item.stop:item.step]
        )

    def write_self_to_file(self, file: h5py.File):
        grp = file.create_group("spec")

        grp.create_dataset("freq", data=self.freq)
        grp.create_dataset("psd_antenna", data=self.psd_antenna)
        grp.create_dataset("psd_ambient", data=self.psd_ambient)
        grp.create_dataset("psd_noise_source", data=self.psd_noise_source)

        grp.create_dataset("psd_antenna_time", data=dtlist2strlist(self.psd_antenna_time))
        grp.create_dataset("psd_ambient_time", data=dtlist2strlist(self.psd_ambient_time))
        grp.create_dataset("psd_noise_source_time", data=dtlist2strlist(self.psd_noise_source_time))

        grp_therm = grp.create_group("spec_therm")
        self.therm.write_self_to_group(grp_therm)

    @classmethod
    def read_self_from_file(cls, file: h5py.File):
        obj = cls()
        grp = file['spec']
        obj.freq = ds2np(grp.get("freq"))
        obj.psd_antenna = ds2np(grp.get("psd_antenna"))
        obj.psd_ambient = ds2np(grp.get("psd_ambient"))
        obj.psd_noise_source = ds2np(grp.get("psd_noise_source"))

        obj.psd_antenna_time = hdfdt2dtlist(grp.get("psd_antenna_time"))
        obj.psd_ambient_time = hdfdt2dtlist(grp.get("psd_ambient_time"))
        obj.psd_noise_source_time = hdfdt2dtlist(grp.get("psd_noise_source_time"))

        obj.therm = Thermistors.read_self_from_group(grp["spec_therm"])
        return obj

    def calc_temp(self):
        self.t_antenna = 2000 * (self.psd_antenna - self.psd_ambient) / (self.psd_noise_source - self.psd_ambient) + 300

    def __add__(self, other):
        if not isinstance(other, Spectrum):
            raise ValueError("Addition defined only for objects of the same class")
        return Spectrum(
            self.therm + other.therm,
            self.freq,
            *add_sort_spec_pair(
                self.psd_antenna,
                self.psd_antenna_time,
                other.psd_antenna,
                other.psd_antenna_time
            ),
            *add_sort_spec_pair(
                self.psd_ambient,
                self.psd_ambient_time,
                other.psd_ambient,
                other.psd_ambient_time,
            ),
            *add_sort_spec_pair(
                self.psd_noise_source,
                self.psd_noise_source_time,
                other.psd_noise_source,
                other.psd_noise_source_time,
            ),
        )

    def __eq__(self, other):
        if self.psd_antenna.shape != other.psd_antenna.shape:
            return False
        if (
                self.therm == other.therm
                and np.isclose(self.freq, other.freq).all()
                and np.isclose(self.psd_antenna, other.psd_antenna).all()
                and np.isclose(self.psd_ambient, other.psd_ambient).all()
                and np.isclose(self.psd_noise_source, other.psd_noise_source).all()
                and np.asarray(self.psd_antenna_time == other.psd_antenna_time).all()
                and np.asarray(self.psd_ambient_time == other.psd_ambient_time).all()
                and np.asarray(self.psd_noise_source_time == other.psd_noise_source_time).all()
        ):
            return True
        return False
