import copy
from pytz import timezone

import h5py
import numpy as np

from .utils import hdfdt2dtlist, dtlist2strlist, ds2np, combine_times, dt2lst


class Thermistors:
    """
    :class:`Thermistors` represents a collection of data measured from multiple thermistors.

    .. attribute:: time

        A list of datetime objects representing the timestamps of the measurements.

    .. attribute:: lna

        A numpy array representing the values measured from the LNA (Low Noise Amplifier).

    .. attribute:: vna_load

        A numpy array representing the values measured from the VNA (Vector Network Analyzer) load.

    .. attribute:: ambient_load

        A numpy array representing the values measured from the ambient load.

    .. attribute:: back_end

        A numpy array representing the values measured from the back-end.

    .. attribute:: lst

        A numpy array representing the Local Solar Time corresponding to each measurement.

    :param time: A list of datetime objects representing the timestamps of the measurements.
    :type time: list(datetime.datetime), optional
    :param lna: A list of values measured from the LNA (Low Noise Amplifier).
    :type lna: list(float), optional
    :param vna_load: A list of values measured from the VNA (Vector Network Analyzer) load.
    :type vna_load: list(float), optional
    :param ambient_load: A list of values measured from the ambient load.
    :type ambient_load: list(float), optional
    :param back_end: A list of values measured from the back-end.
    :type back_end: list(float), optional

    """
    def __init__(
            self, time=None, lna=None, vna_load=None, ambient_load=None, back_end=None
    ):
        self.time = time
        self.lna = np.array(lna)
        self.vna_load = np.array(vna_load)
        self.ambient_load = np.array(ambient_load)
        self.back_end = np.array(back_end)
        self.lst = None

    def to_lst(self, lon: float, tz: timezone, inplace: bool = False):
        """
        Convert time values to local sidereal time.

        :param lon: The longitude of the location.
        :param tz: The timezone of the location.
        :param inplace: (Optional) If True, modifies the current object.
                        If False, returns a new object with modified time values.
        :return: None if inplace is True, otherwise a new object with modified time values.
        """
        obj = self if inplace else copy.copy(self)
        obj.lst = np.array([dt2lst(tz.localize(dti), lon) for dti in self.time])
        return None if inplace else obj

    def write_self_to_group(self, grp: h5py.Group):
        """
        Writes the Thermistors object to the given h5py.Group.

        :param grp: The h5py.Group to write the data to.
        """
        grp.create_dataset("lna", data=self.lna)
        grp.create_dataset("vna_load", data=self.vna_load)
        grp.create_dataset("ambient_load", data=self.ambient_load)
        grp.create_dataset("back_end", data=self.back_end)
        grp.create_dataset("time", data=dtlist2strlist(self.time))

    @classmethod
    def read_self_from_group(cls, grp: h5py.Group):
        """
            Read self from group.

            :param grp: The group containing the thermistors data.
            :return: An instance of the Thermistors class.
            :rtype: Thermistors
        """
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
