import copy
from datetime import datetime
from typing import Sequence
import warnings

import h5py
import numpy as np
import numpy.ma as ma
from pytz import timezone

from .Thermistors import Thermistors
from .plotting import plot_spec, plot_spec_rows, plot_spec_stats
from .utils import add_sort_spec_pair, dt2lst
from .utils import hdfdt2dtlist, dtlist2strlist, ds2np
from . import rfi


class Spectrum:
    """
    Class representing a spectrum.
    Attributes:
        therm (Thermistors): Thermistors object representing the thermometers
        used in the measurement.
        freq (numpy.ndarray): The frequency values.
        psd_antenna (numpy.ndarray): The PSD antenna values.
        _psd_antenna_time (numpy.ndarray): The PSD antenna time values.
        psd_ambient (numpy.ndarray): The PSD ambient values.
        psd_ambient_time (numpy.ndarray): The PSD ambient time values.
        psd_noise_source (numpy.ndarray): The PSD noise source values.
        psd_noise_source_time (numpy.ndarray): The PSD noise source times.
        lst (numpy.ndarray): The local sidereal time values.
        t_antenna (numpy.ma.masked_array): The temperature of the antennas.
        time_start (datetime): The start time of the antenna PSD values.
        time_end (datetime): The end time of the antenna PSD values.
    """

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
        self._psd_antenna_time = psd_antenna_time
        self.psd_ambient = psd_ambient
        self.psd_ambient_time = psd_ambient_time
        self.psd_noise_source = psd_noise_source
        self.psd_noise_source_time = psd_noise_source_time
        self.lst = None
        self.t_antenna = None
        if psd_antenna_time is None:
            self.time_start = None
            self.time_end = None
        else:
            self.time_start = min(psd_antenna_time)
            self.time_end = max(psd_antenna_time)
        self.rfi_flags = np.zeros_like(self.psd_antenna, dtype=bool)

    def __getitem__(self, item):
        return Spectrum(
            therm=self.therm,
            freq=self.freq,
            psd_antenna=self.psd_antenna[item],
            psd_antenna_time=self.psd_antenna_time[item],
            psd_ambient=self.psd_ambient[item],
            psd_ambient_time=self.psd_ambient_time[item],
            psd_noise_source=self.psd_noise_source[item],
            psd_noise_source_time=self.psd_noise_source_time[item],
        )

    @property
    def psd_antenna_time(self):
        return self._psd_antenna_time

    @psd_antenna_time.setter
    def psd_antenna_time(self, timearr):
        if isinstance(timearr[0], datetime):
            self.time_start = timearr[0]
            self.time_end = timearr[-1]
        self._psd_antenna_time = timearr

    @property
    def time(self):
        return self._psd_antenna_time

    def flag_mad(self, limit=5):
        """
        :param limit: The limit for the median absolute deviation (MAD)
        flagging. Default value is 5.
        :return: None

        The `flag_mad` method flags data points in the `t_antenna` attribute of
        the `Spectrum` object based on their deviation from the median.

        The method calculates the median of each column in the `t_antenna`
        data, and then computes the absolute difference between each data
        point and its column median. It then calculates the median of these
        absolute differences across all columns, which gives the MAD value.

        Data points that have an absolute difference from the column median
        greater than `limit * mad` are flagged as masked in the `t_antenna`
        data, using the `ma.masked_array` function.

        Note that this method modifies the `t_antenna` attribute of the
        `Spectrum` object in-place.

        """
        warnings.warn(
            "Spectrum.flag_mad is deprecated and will be removed in a future"
            " release. Use methods from rfi.py instead.",
            DeprecationWarning,
        )
        rfi.flag_mad(self, limit=limit, inplace=True)

    def flag_rfi(self, flags):
        self.rfi_flags = np.logical_or(self.rfi_flags, flags)
        self.t_antenna = ma.masked_array(self.t_antenna, flags)
        self.psd_antenna = ma.masked_array(self.psd_antenna, flags)

    def write_self_to_file(self, file: h5py.File):
        """
        :param file: An instance of h5py.File to write the spectrum data to.
        :return: None
        """
        grp = file.create_group("spec")

        grp.create_dataset("freq", data=self.freq)
        grp.create_dataset("psd_antenna", data=self.psd_antenna)
        grp.create_dataset("psd_ambient", data=self.psd_ambient)
        grp.create_dataset("psd_noise_source", data=self.psd_noise_source)

        grp.create_dataset(
            "psd_antenna_time", data=dtlist2strlist(self.psd_antenna_time)
        )
        grp.create_dataset(
            "psd_ambient_time", data=dtlist2strlist(self.psd_ambient_time)
        )
        grp.create_dataset(
            "psd_noise_source_time",
            data=dtlist2strlist(self.psd_noise_source_time),
        )

        grp_therm = grp.create_group("spec_therm")
        self.therm.write_self_to_group(grp_therm)

    @classmethod
    def read_self_from_file(cls, file: h5py.File):
        """
        Reads a Spectrum object from a file.

        :param file: The h5py.File object representing the file to read from.
        :return: The Spectrum object read from the file.
        """
        obj = cls()
        grp = file["spec"]
        obj.freq = ds2np(grp.get("freq"))
        obj.psd_antenna = ds2np(grp.get("psd_antenna"))
        obj.psd_ambient = ds2np(grp.get("psd_ambient"))
        obj.psd_noise_source = ds2np(grp.get("psd_noise_source"))
        obj.psd_antenna_time = hdfdt2dtlist(grp.get("psd_antenna_time"))
        obj.psd_ambient_time = hdfdt2dtlist(grp.get("psd_ambient_time"))
        obj.psd_noise_source_time = hdfdt2dtlist(
            grp.get("psd_noise_source_time")
        )
        obj.therm = Thermistors.read_self_from_group(grp["spec_therm"])

        shapes = (
            obj.psd_antenna.shape[0],
            obj.psd_ambient.shape[0],
            obj.psd_noise_source.shape[0],
        )
        min_shape = min(shapes)
        if not shapes[0] == shapes[1] == shapes[2]:
            warnings.warn(
                "Antenna, ambient and ambient+noise spectra have different"
                f"row count, specifically {shapes}."
                f" Cropping all arrays to length {min_shape}."
            )
            obj.psd_antenna = obj.psd_antenna[:min_shape]
            obj.psd_ambient = obj.psd_ambient[:min_shape]
            obj.psd_noise_source = obj.psd_noise_source[:min_shape]
            obj.psd_antenna_time = obj.psd_antenna_time[:min_shape]
            obj.psd_ambient_time = obj.psd_ambient_time[:min_shape]
            obj.psd_noise_source_time = obj.psd_noise_source_time[:min_shape]
        return obj

    @property
    def nspec(self):
        return np.atleast_2d(self.psd_antenna).shape[0]

    def cut_freq(self, freq_min=None, freq_max=None):
        """
        Cut the frequency range of the spectrum.

        :param freq_min: The minimum frequency value to keep in MHz.
        :param freq_max: The maximum frequency value to keep in MHz.
        :return: None
        """
        if freq_min is None:
            freq_min = np.min(self.freq)
        if freq_max is None:
            freq_max = np.max(self.freq)
        mask = (self.freq >= freq_min) & (self.freq <= freq_max)
        self.freq = self.freq[mask]
        self.psd_antenna = self.psd_antenna[..., mask]
        self.psd_ambient = self.psd_ambient[..., mask]
        self.psd_noise_source = self.psd_noise_source[..., mask]
        if self.t_antenna is not None:
            self.t_antenna = self.t_antenna[..., mask]

    def calc_temp(self):
        """
        Calculates the antenna temperature using the formula:
        temp = 2000 * (psd_antenna - psd_ambient) /
                  (psd_noise_source - psd_ambient) + 300
        and places it in Spectrum.t_antenna
        :return: None
        """
        # TODO: Fix bigger size of ambient and throw a warning
        self.t_antenna = (
            2000
            * (self.psd_antenna - self.psd_ambient)
            / (self.psd_noise_source - self.psd_ambient)
            + 300
        )

    def to_lst(self, lon: float, tz: timezone, inplace: bool = False):
        """
        Convert antenna time to Local Sidereal Time (LST) for each data point.

        :param lon: Longitude in degrees.
        :param tz: Timezone object from the pytz library.
        :param inplace: Optional boolean indicating if the calculation should
        be performed in-place.
            If False, a new Spectrum object will be returned with the
            calculated LST values.
        :return: None if inplace is True, otherwise a new Spectrum object with
        calculated LST values.
        """
        obj = self if inplace else copy.copy(self)
        obj.lst = np.array(
            [dt2lst(tz.localize(dti), lon) for dti in self.psd_antenna_time]
        )
        return None if inplace else obj

    def plot_psd_antenna(self, **kwargs):
        return plot_spec(
            self.freq, self.psd_antenna_time, self.psd_antenna, **kwargs
        )

    def plot_psd_ambient(self, **kwargs):
        return plot_spec(
            self.freq, self.psd_ambient_time, self.psd_ambient, **kwargs
        )

    def plot_psd_noise_source(self, **kwargs):
        return plot_spec(
            self.freq,
            self.psd_noise_source_time,
            self.psd_noise_source,
            **kwargs,
        )

    def plot_temp(self, **kwargs):
        if self.t_antenna is None:
            self.calc_temp()

        if "vmin" not in kwargs:
            kwargs["vmin"] = 100
        if "vmax" not in kwargs:
            kwargs["vmax"] = 1200
        return plot_spec(
            self.freq, self.psd_antenna_time, self.t_antenna, **kwargs
        )

    def plot_psd_antenna_rows(self, rows: Sequence, **kwargs):
        return plot_spec_rows(self.freq, self.psd_antenna, rows, **kwargs)

    def plot_psd_antenna_stats(self, **kwargs):
        return plot_spec_stats(self.freq, self.psd_antenna, **kwargs)

    def __add__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self

        if not isinstance(other, Spectrum):
            raise ValueError(
                "Addition defined only for objects of the same class"
            )
        # TODO: add derivative values
        return Spectrum(
            self.therm + other.therm,
            self.freq,
            *add_sort_spec_pair(
                self.psd_antenna,
                self.psd_antenna_time,
                other.psd_antenna,
                other.psd_antenna_time,
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

    def __radd__(self, other):
        return self + other

    def __eq__(self, other):
        if self.psd_antenna.shape != other.psd_antenna.shape:
            return False
        if (
            self.therm == other.therm
            and np.isclose(self.freq, other.freq).all()
            and np.isclose(self.psd_antenna, other.psd_antenna).all()
            and np.isclose(self.psd_ambient, other.psd_ambient).all()
            and np.isclose(self.psd_noise_source, other.psd_noise_source).all()
            and np.asarray(
                self.psd_antenna_time == other.psd_antenna_time
            ).all()
            and np.asarray(
                self.psd_ambient_time == other.psd_ambient_time
            ).all()
            and np.asarray(
                self.psd_noise_source_time == other.psd_noise_source_time
            ).all()
        ):
            return True
        return False
