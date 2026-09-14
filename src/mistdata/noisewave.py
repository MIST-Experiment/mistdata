"""
Noise-wave calibration of the MIST receiver.

The calibration model is Equations 5-13 of the MIST instrument paper
(Monsalve et al. 2024, M24), as implemented in MISTCalibration. For a
calibrator with physical temperature T, it reads

    T = k0 (C1 (T_LNS - T_L) Q + T_L - C2) - kU TU - kC TC - kS TS,

where Q = (P - P_L) / (P_LNS - P_L) is the ratio of the measured PSDs,
T_L and T_LNS are the assumed temperatures of the internal load and load +
noise source, and k0, kU, kC, kS depend on the reflection coefficients of
the calibrator and the receiver.

The model is linear in C1, C2, TU, TC and TS. Modelling each as a
polynomial in frequency, a single linear least-squares fit to the hot load,
ambient load, open cable and shorted cable measurements gives all five at
once. This replaces the iteration between the C parameters and the noise
wave parameters in Monsalve et al. 2017.
"""

import numpy as np

C_KEYS = ("C1", "C2")
NW_KEYS = ("TU", "TC", "TS")


class NoiseWave:

    def __init__(self, calibrators, temperatures, npoly=7):
        """
        Parameters
        ----------
        calibrators : dict
            MISTCalibration object for each calibrator, e.g. with keys
            'hot', 'ambient', 'open' and 'short'. The S11s must be fit with
            MISTCalibration.fit_s11 and all calibrators must share the same
            frequency axis. Their noise wave and C parameters are not used.
        temperatures : dict
            Physical temperature in Kelvin of each calibrator, with the same
            keys as calibrators. Either a float or an array with one value
            per frequency.
        npoly : int
            Number of polynomial terms used to model each of C1, C2, TU, TC
            and TS in frequency.

        """
        self.calibrators = calibrators
        self.temperatures = temperatures
        self.npoly = npoly

        self.freq = next(iter(calibrators.values())).freq
        for name, cal in calibrators.items():
            if not np.array_equal(cal.freq, self.freq):
                raise ValueError(
                    f"Calibrator '{name}' has a different frequency axis."
                )

        self.coeffs = None
        # calibrated minus physical temperature of each calibrator, set by
        # solve, shape (nfiles, nfreq)
        self.residuals = None

    @property
    def basis(self):
        """
        Legendre polynomials evaluated on the frequency axis mapped to
        [-1, 1]. Shape is (nfreq, npoly).
        """
        fmin, fmax = self.freq.min(), self.freq.max()
        x = (2 * self.freq - fmin - fmax) / (fmax - fmin)
        return np.polynomial.legendre.legvander(x, self.npoly - 1)

    def _design(self, cal, temperature):
        """
        Rows of the least-squares problem for one calibrator. There is one
        block of rows per file, using the time-averaged PSDs of that file.

        Returns
        -------
        A : ndarray
            Design matrix of shape (nfiles * nfreq, 5 * npoly), the columns
            multiply the polynomial coefficients of C1, C2, TU, TC and TS.
        b : ndarray
            Data vector of shape (nfiles * nfreq,).

        """
        spec = cal.mistdata.spec
        # PSDs are (nfiles, nspec_per_file, nfreq), k params (nfiles, 1, nfreq)
        p_ant = spec.psd_antenna.mean(axis=1)
        p_load = spec.psd_ambient.mean(axis=1)
        p_lns = spec.psd_noise_source.mean(axis=1)
        q = (p_ant - p_load) / (p_lns - p_load)
        k = {key: cal.k_params[key][:, 0] for key in ("k0", "kU", "kC", "kS")}
        dT = cal.t_assumed_LNS - cal.t_assumed_L

        columns = [k["k0"] * dT * q, -k["k0"], -k["kU"], -k["kC"], -k["kS"]]
        blocks = []
        for i in range(q.shape[0]):
            blocks.append(
                np.hstack([col[i, :, None] * self.basis for col in columns])
            )
        A = np.vstack(blocks)
        T = np.broadcast_to(temperature, q.shape)
        b = (T - k["k0"] * cal.t_assumed_L).ravel()
        return A, b

    def solve(self, sigma=None):
        """
        Fit C1, C2, TU, TC and TS to the calibrator measurements.

        Parameters
        ----------
        sigma : dict
            Uncertainty in Kelvin of the calibrated temperature of each
            calibrator, used to weight the fit. Either a float or an array
            with one value per frequency. By default all calibrators have
            equal weight.

        Returns
        -------
        C_params : dict
            C1 and C2 as a function of frequency.
        nw_params : dict
            Noise wave parameters TU, TC and TS as a function of frequency.

        """
        nfreq = self.freq.size
        designs = {
            name: self._design(cal, self.temperatures[name])
            for name, cal in self.calibrators.items()
        }
        A, b = [], []
        for name, (A_cal, b_cal) in designs.items():
            s = 1.0 if sigma is None else sigma[name]
            w = 1 / np.broadcast_to(s, (b_cal.size // nfreq, nfreq)).ravel()
            A.append(A_cal * w[:, None])
            b.append(b_cal * w)
        A = np.vstack(A)
        b = np.concatenate(b)

        # normalize the columns to improve the conditioning
        norm = np.linalg.norm(A, axis=0)
        x = np.linalg.lstsq(A / norm, b, rcond=None)[0] / norm

        # rows of the design matrix are temperatures, so A x - b is the
        # calibrated minus the physical temperature
        self.residuals = {
            name: (A_cal @ x - b_cal).reshape(-1, nfreq)
            for name, (A_cal, b_cal) in designs.items()
        }

        keys = C_KEYS + NW_KEYS
        self.coeffs = dict(zip(keys, x.reshape(len(keys), self.npoly)))
        params = {key: self.basis @ c for key, c in self.coeffs.items()}
        C_params = {key: params[key] for key in C_KEYS}
        nw_params = {key: params[key] for key in NW_KEYS}
        return C_params, nw_params
