from typing import List

import h5py
import numpy as np

from .utils import hdfdt2dtlist, dtlist2strlist, ds2np, combine_times


class Thermistors:
    def __init__(
            self, time=None, lna=None, vna_load=None, ambient_load=None, back_end=None
    ):
        self.time = time
        self.lna = np.array(lna)
        self.vna_load = np.array(vna_load)
        self.ambient_load = np.array(ambient_load)
        self.back_end = np.array(back_end)

    def write_self_to_group(self, grp: h5py.Group):
        grp.create_dataset("lna", data=self.lna)
        grp.create_dataset("vna_load", data=self.vna_load)
        grp.create_dataset("ambient_load", data=self.ambient_load)
        grp.create_dataset("back_end", data=self.back_end)
        grp.create_dataset("time", data=dtlist2strlist(self.time))

    @classmethod
    def read_self_from_group(cls, grp: h5py.Group):
        return cls(
            time=hdfdt2dtlist(grp.get("time")),
            lna=ds2np(grp.get("lna")),
            vna_load=ds2np(grp.get("vna_load")),
            ambient_load=ds2np(grp.get("ambient_load")),
            back_end=ds2np(grp.get("back_end")),
        )

    def __add__(self, other):
        if not isinstance(other, Thermistors):
            raise ValueError("Addition defined only for objects of the same class")

        time = combine_times(self.time, other.time)
        idxs = np.argsort(time)
        return Thermistors(
            [time[i] for i in idxs],
            np.hstack((self.lna, other.lna))[idxs],
            np.hstack((self.vna_load, other.vna_load))[idxs],
            np.hstack((self.ambient_load, other.ambient_load))[idxs],
            np.hstack((self.back_end, other.back_end))[idxs],
        )

    def __eq__(self, other):
        if self.time != other.time:
            return False
        if (
                np.asarray(self.time == other.time).all()
                and np.isclose(self.lna, other.lna).all()
                and np.isclose(self.vna_load, other.vna_load).all()
                and np.isclose(self.ambient_load, other.ambient_load).all()
                and np.isclose(self.back_end, other.back_end).all()
        ):
            return True
        return False
