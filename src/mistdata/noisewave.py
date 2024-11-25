"""
Tools for calculating noise wave parameters needed to calibrate MIST data.
Notations are following Monsalve et al. 2017 (M17), with some specific
quantities from Monsalve et al. 2024 (M24).
"""

from functools import partial
import numpy as np
from scipy.optimize import curve_fit

from mistdata import MISTCalibration


def _nw_model(freq, coeffs, open_cal, short_cal):
    """
    Model of antenna temperature given noise wave parameters. This is
    fit to measurements of the open and short cables, using Eq. 7 in M17.
    Inputs are the polynomial coefficients of the noise wave parameters,
    in order of decreasing degree (the last element is the constant term).

    Parameters
    ----------
    freq : array-like
        Frequency in Hz, the first M elements are the frequency points of
        the open cable measurements, and the next N elements are the
        frequency points of the short cable measurements. In practice, the
        frequency points of the open and short cables are the same.
    coeffs : array-like
        Polynomial coefficients that model the noise wave parameters. This
        must have a length of 3 times the degree of the polynomial. For
        a polynomial of degree N, the coefficients are ordered as follows:
        [Tunc_N, ..., Tunc_0, Tcos_N, ..., Tcos_0, Tsin_N, ..., Tsin_0].
    open_cal : MISTCalibration
        Calibration object for the open cable.
    short_cal : MISTCalibration
        Calibration object for the short cable.

    Returns
    -------
    T_open_short : ndarray
        Concatenated calibrated temperature spectra of the open and short
        cables. The first M elements are the calibrated temperature spectrum
        of the open cable, and the next N elements are the calibrated
        temperature spectrum of the short cable.

    """
    Tunc_pars, Tcos_pars, Tsin_pars = np.split(coeffs, 3)
    Tunc = np.polyval(Tunc_pars, freq)
    Tcos = np.polyval(Tcos_pars, freq)
    Tsin = np.polyval(Tsin_pars, freq)
    open_cal.nw_params = {"TU": Tunc, "TC": Tcos, "TS": Tsin}
    short_cal.nw_params = {"TU": Tunc, "TC": Tcos, "TS": Tsin}
    T_open = open_cal.antenna_temp
    T_short = short_cal.antenna_temp
    return np.concatenate((T_open, T_short))


class NoiseWave:

    # degree of the polynomial model for the noise wave parameters (from M17)
    nw_poly_deg = 7

    def __init__(
        self,
        hot_data,
        ambient_data,
        open_data,
        short_data,
        pathA_sparams,
        gamma_r,
    ):
        """
        Parameters
        ----------
        hot_data : MISTData
            MISTData object containing the hot load data.
        ambient_data : MISTData
            MISTData object containing the ambient load data.
        open_data : MISTData
            MISTData object containing the open cable data.
        short_data : MISTData
            MISTData object containing the short cable data
        pathA_sparams : array-like
            S-parameters of the path A in Figure 19 of M24. This is needed
            to shift the reference plane of the S11-measurements of the
            calibrators connected to the receiver input.
        gamma_r : complex
            Reflection coefficient looking in to the receiver input.

        """
        # initialize noise wave parameters with zeros
        nw_params = {"TU": 0, "TC": 0, "TS": 0}  # Tunc, Tcos, Tsin
        C_params = {"C1": 0, "C2": 0}  # corrections C1, C2
        cal_data = {
            "pathA_sparams": pathA_sparams,
            "gamma_r": gamma_r,
            "nw_params": nw_params,
            "C_params": C_params,
        }
        t_LNS = 350  # assumed noise temperature of load + noise source

        self.hot_cal = MISTCalibration(hot_data, cal_data, t_assumed_LNS=t_LNS)
        self.ambient_cal = MISTCalibration(
            ambient_data, cal_data, t_assumed_LNS=t_LNS
        )
        self.open_cal = MISTCalibration(
            open_data, cal_data, t_assumed_LNS=t_LNS
        )
        self.short_cal = MISTCalibration(
            short_data, cal_data, t_assumed_LNS=t_LNS
        )

        self.freq = self.open_cal.freq

        # physical temperature of the calibrators XXX
        self.T_hot = None
        self.T_amb = None
        self.T_open = None
        self.T_short = None

    @property
    def C_params(self):
        return self.hot_cal.C_params

    @property
    def nw_params(self):
        return self.hot_cal.nw_params

    def _update_C(self):
        """
        Update the C parameters in the calibration data. This uses Eq. 10
        and Eq. 11 in M17.

        """
        c1 = self.C_params["C1"]
        c2 = self.C_params["C2"]
        Th_spec = self.hot_cal.antenna_temp  # T_H^i
        Ta_spec = self.ambient_cal.antenna_temp  # T_A^i

        c1_next = c1 * (self.T_hot - self.T_amb) / (Th_spec - Ta_spec)
        c2_next = c2 + Ta_spec - self.T_amb
        C_next = {"C1": c1_next, "C2": c2_next}

        self.hot_cal.C_params = C_next
        self.ambient_cal.C_params = C_next
        self.open_cal.C_params = C_next
        self.short_cal.C_params = C_next

    def _update_nw(self):
        """
        Use measurements of open and short cables to fit the noise wave
        parameters. This is a least squares fit to Eq. 7 in M17.

        """
        mdl = partial(
            _nw_model, open_cal=self.open_cal, short_cal=self.short_cal
        )
        xdata = np.concatenate((self.open_cal.freq, self.short_cal.freq))
        ydata = np.concatenate(
            np.full(len(self.open_cal.freq), self.T_open),
            np.full(len(self.short_cal.freq), self.T_short),
        )
        p0 = [0] * self.nw_poly_deg * 3
        popt = curve_fit(mdl, xdata, ydata, p0=p0)[0]
        Tunc_pars, Tcos_pars, Tsin_pars = np.split(popt, 3)
        Tunc = np.polyval(Tunc_pars, self.freq)
        Tcos = np.polyval(Tcos_pars, self.freq)
        Tsin = np.polyval(Tsin_pars, self.freq)
        nw_params = {"TU": Tunc, "TC": Tcos, "TS": Tsin}

        self.hot_cal.nw_params = nw_params
        self.ambient_cal.nw_params = nw_params
        self.open_cal.nw_params = nw_params
        self.short_cal.nw_params = nw_params

    def iterate(self):
        """
        Iterate the calibration process until the C parameters converge.

        Returns
        -------
        C_params : dict
            Converged C parameters.
        nw_params : dict
            Noise wave parameters TU, TC, TS as a function of frequency.

        """
        for i in range(3):
            self._update_C()
            self._update_nw()

        return self.C_params, self.nw_params
