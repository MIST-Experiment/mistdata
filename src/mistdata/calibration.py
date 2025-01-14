"""
High level tools for MIST calibration.
"""

from copy import deepcopy
from functools import cached_property
import numpy as np
import pickle
from .fitting import Fit
from .s11 import AntennaS11, ReceiverS11


class MISTCalibration:

    # frequency range to do calibration over
    fmin = 23
    fmax = 107

    def __init__(
        self,
        mistdata,
        cal_data,
        t_assumed_L=300,
        t_assumed_LNS=2300,
    ):
        """
        Class holding parameters and methods needed for MIST calibration,
        going from antenna PSD to sky temperature.

        Parameters
        ----------
        mistdata : MISTData
            MISTData object containing the data to be calibrated.
        cal_data : dict
            Dictionary containing the calibration data. Possible keys are
            'gamma_a', 'gamma_r', 'pathA_sparams', 'pathB_sparams',
            'pathC_sparams', 'nw_params', and 'C_params'. See notes.
        t_assumed_L : float
            Assumed temperature of the load in Kelvin. Default is 300 K.
        t_assumed_LNS : float
            Assumed temperature of the load + noise source in Kelvin. Default
            is 2300 K.

        Notes
        -----
        The calibration data dictionary can either directly contain the
        calibrated S11 parameters of the antenna and receiver, with keys
        'gamma_a' and 'gamma_r', or the S-parameters of the internal paths
        with keys 'pathA_sparams', 'pathB_sparams', and 'pathC_sparams'. If
        the paths are given, the calibration will be done automatically.

        The calibration data dictionary must also contain the noise wave
        parameters with keys 'nw_params', and the corrections to the noise
        wave parameters with keys 'C_params'.

        Note that all the data in cal_data should have a frequency axis
        that is the same as the S11 frequency axis in mistdata, that is,
        mistdata.dut_recin.s11_freq.

        """
        self.mistdata = deepcopy(mistdata)

        # s11 parameters
        try:
            gamma_a = cal_data["gamma_a"]
        except KeyError:
            pathA_sparams = cal_data["pathA_sparams"]
            gamma_a = AntennaS11(self.mistdata.dut_recin, pathA_sparams).s11
        try:
            gamma_r = cal_data["gamma_r"]
        except KeyError:
            pathB_sparams = cal_data["pathB_sparams"]
            pathC_sparams = cal_data["pathC_sparams"]
            gamma_r = ReceiverS11(
                self.mistdata.dut_lna, pathB_sparams, pathC_sparams
            ).s11
        gamma_a = np.atleast_2d(gamma_a["antenna"])

        self.t_assumed_L = t_assumed_L
        self.t_assumed_LNS = t_assumed_LNS

        # noise wave parameters
        self.nw_params = cal_data["nw_params"]
        self.C_params = cal_data["C_params"]  # corrections to nw parameters

        # broadcasting: there's one VNA measurement per file, many spectra
        self.nfiles = self.gamma_a.shape[0]
        self.nspec = self.mistdata.spec.psd_antenna.shape[0]
        spec_per_file = self.nspec // self.nfiles
        shape = (self.nfiles, spec_per_file, self.nfreq)
        # spectral measurements are now nfiles x nspec_per_file x nfreq
        self.spec.psd_noise_source.shape = shape
        self.spec.psd_ambient.shape = shape
        self.spec.psd_antenna.shape = shape
        # raw s11 measurements are now nfiles x 1 x nfreq_s11
        self._gamma_a = gamma_a[:, np.newaxis, :]
        self._gamma_r = gamma_r[np.newaxis, np.newaxis, :]

        # fit s11 spectra with self.fit_s11
        self.fit = {}
        self.gamma_a = None
        self.gamma_r = None

    def save(self, filename):
        """
        Save the calibration object to a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to save the calibration object to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load a calibration object from a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to load the calibration object from.

        Returns
        -------
        MISTCalibration
            Loaded calibration object.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def s11_freq(self):
        return self.mistdata.dut_recin.s11_freq

    @property
    def nfreq_s11(self):
        return self.s11_freq.size

    @property
    def freq(self):
        return self.mistdata.spec.freq

    @property
    def nfreq(self):
        return self.freq.size

    def fit_s11(self, device, model="dpss", nterms=10):
        """
        Fit the S11 parameters of the antenna to a model.

        Parameters
        ----------
        device : str
            Which spectra to fit. Either 'antenna' or 'receiver'.
        model : str
            Which model to fit S11 spectra to. Either 'dpss' or 'fourier'.
        nterms : int
            Number of terms to use in the fit.

        """
        if device == "antenna":
            gamma = self._gamma_a[:, 0, :]  # squeeze axis 1 since it's 1
        elif device == "receiver":
            gamma = self._gamma_r[:, 0, :]
        else:
            raise ValueError("Device must be 'antenna' or 'receiver'")

        # the spectra are the last axis of gamma
        mdl = np.empty_like(gamma)
        fits = []
        for i in range(gamma.shape[0]):
            fit = Fit(self.s11_freq, gamma[i], model, nterms, sigma=1)
            fit.fit()
            mdl[i] = fit.predict(self.freq)
            fits.append(fit)
        self.fit[device] = fits

        if device == "antenna":
            self.gamma_a = mdl[:, np.newaxis, :]  # add axis 1 back
        else:
            self.gamma_r = mdl[:, np.newaxis, :]

    def cut_freq(self, fmin=None, fmax=None):
        """
        Restrict the frequency range of the calibration data.

        Parameters
        ----------
        fmin : float
            Minimum frequency in MHz.
        fmax : float
            Maximum frequency in MHz.

        """
        if fmin is None:
            fmin = np.min((self.freq.min(), self.s11_freq.min()))
        if fmax is None:
            fmax = np.max((self.freq.max(), self.s11_freq.max()))

        mask = (self.freq >= fmin) & (self.freq <= fmax)
        s11_mask = (self.s11_freq >= fmin) & (self.s11_freq <= fmax)

        self._gamma_a = self._gamma_a[:, :, s11_mask]
        self._gamma_r = self._gamma_r[:, :, s11_mask]
        if self.gamma_a is not None:
            self.gamma_a = self.gamma_a[:, :, mask]
        if self.gamma_r is not None:
            self.gamma_r = self.gamma_r[:, :, mask]

        self.mistdata.cut_freq(
            freq_min=self.fmin, freq_max=self.fmax, inplace=True
        )

    @cached_property
    def k_params(self):
        """
        Calculate the k parameters for the calibration, given in Equations
        8-11 of the MIST instrument paper (Monsalve et al. 2024).

        Returns
        -------
        dict
            Dictionary containing the k parameters. The keys are
            'k0', 'kU', 'kC', 'kS'.
        """
        # intermediate parameters
        xa = 1 - np.abs(self.gamma_a) ** 2
        xr = 1 - np.abs(self.gamma_r) ** 2
        F = np.sqrt(xr) / (1 - self.gamma_a * self.gamma_r)  # eq 12
        alpha = np.angle(self.gamma_a * F)  # eq 13

        _k_params = {}
        _k_params["k0"] = xr / (xa * np.abs(F) ** 2)
        kU = np.abs(self.gamma_a) ** 2 / xa
        _k_params["kU"] = kU
        _k_params["kC"] = kU / np.abs(F) * np.cos(alpha)
        _k_params["kS"] = kU / np.abs(F) * np.sin(alpha)
        _k_params["F"] = F
        _k_params["alpha"] = alpha
        return _k_params

    @property
    def receiver_gain(self):
        """
        Calculate the receiver gain of the receiver, given in Equation 6 of
        the MIST instrument paper (Monsalve et al. 2024).

        Returns
        -------
        float
            Receiver gain.
        """
        pdiff = self.spec.psd_noise_source - self.spec.psd_ambient
        tdiff = self.t_assumed_LNS - self.t_assumed_L
        k0 = self.k_params["k0"]
        C1 = self.C_params["C1"]
        return pdiff / (k0 * C1 * tdiff)

    @property
    def receiver_temp(self):
        """
        Calculate the receiver temperature of the receiver, given in
        Equation 7 of the MIST instrument paper (Monsalve et al. 2024).

        Returns
        -------
        float
            Receiver temperature in Kelvin.
        """
        t1 = self.spec.psd_ambient / self.receiver_gain
        t2 = self.k_params["k0"] * (self.t_assumed_L - self.C_params["C2"])
        U = self.k_params["kU"] * self.nw_params["TU"]
        C = self.k_params["kC"] * self.nw_params["TC"]
        S = self.k_params["kS"] * self.nw_params["TS"]
        return t1 - t2 + U + C + S

    @property
    def antenna_temp(self):
        """
        Calculate the antenna temperature of the antenna, given in
        Equation 5 of the MIST instrument paper (Monsalve et al. 2024).
        """
        return self.spec.psd_antenna / self.receiver_gain - self.receiver_temp

    # XXX belongs somewhere else, probably in MISTData
    def calc_Tsky(self, Tphys=0, eta_rad=1, eta_beam=1, eta_balun=1):
        """
        Calculate the sky temperature given the radiation efficiency,
        beam efficiency, balun efficiency, and the temperature of the
        passive loss sources. To exclude any of the effects, simply leave
        the variable as their default values. This is useful e.g. if there
        is no balun.

        Parameters
        ----------
        Tphys : float
            Physical temperature associated with passive loss sources.
        eta_rad : float
            Radiation efficiency.
        eta_beam : float
            Beam efficiency.
        eta_balun : float
            Balun efficiency.

        Returns
        -------
        float
            Sky temperature in Kelvin.

        """
        eta = eta_rad * eta_beam * eta_balun
        return (self.antenna_temp - (1 - eta) * Tphys) / eta
