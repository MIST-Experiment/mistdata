"""
High level tools for MIST calibration.
"""

from functools import cached_property
import numpy as np
from .s11 import AntennaS11, ReceiverS11
from .utils import fourier_series, fit_fourier


class MISTCalibration:

    def __init__(
        self, mistdata, cal_data, t_assumed_L=300, t_assumed_LNS=2300
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
        'gamma_a' and 'gamma_r', or the s parameters of the internal paths
        with keys 'pathA_sparams', 'pathB_sparams', and 'pathC_sparams'. If
        the paths are given, the calibration will be done automatically.

        The calibration data dictionary must also contain the noise wave
        parameters with keys 'nw_params', and the corrections to the noise
        wave parameters with keys 'C_params'.

        """
        self.mistdata = mistdata
        self.spec = mistdata.spec
        self.t_assumed_L = t_assumed_L
        self.t_assumed_LNS = t_assumed_LNS

        # s11 parameters
        try:
            gamma_a = cal_data["gamma_a"]
        except KeyError:
            pathA_sparams = cal_data["pathA_sparams"]
            gamma_a = AntennaS11(mistdata.dut_recin, pathA_sparams).s11
        try:
            gamma_r = cal_data["gamma_r"]
        except KeyError:
            pathB_sparams = cal_data["pathB_sparams"]
            pathC_sparams = cal_data["pathC_sparams"]
            gamma_r = ReceiverS11(
                mistdata.dut_lna, pathB_sparams, pathC_sparams
            ).s11
        # fit the s11 parameters to a fourier series over 25-105 MHz
        deg = 5
        fmin = 25
        fmax = 105
        s11_freq = mistdata.dut_recin.s11_freq
        ydata = np.concatenate(
            (gamma_a["antenna"].real, gamma_a["antenna"].imag)
        )
        self._gamma_a = gamma_a["antenna"]
        popt = fit_fourier(
            s11_freq, ydata, deg, xmin=fmin, xmax=fmax, complex_data=True
        )
        self.gamma_a = fourier_series(mistdata.spec.freq, popt)

        ydata = np.concatenate((gamma_r.real, gamma_r.imag))
        self._gamma_r = gamma_r
        popt = fit_fourier(
            s11_freq, ydata, deg, xmin=fmin, xmax=fmax, complex_data=True
        )
        self.gamma_r = fourier_series(mistdata.spec.freq, popt)

        # noise wave parameters
        self.nw_params = cal_data["nw_params"]
        self.C_params = cal_data["C_params"]  # corrections to nw parameters

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
