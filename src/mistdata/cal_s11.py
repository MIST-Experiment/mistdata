"""
Tools for calibrating S11 measurements.

The calibration formalism is described in the MIST instrument paper (Monsalve
et al. 2024; M24). Equations related to OSL calibration of reflection 
coefficient measurements are described in Monsalve et al. 2016 (M16).
"""

def impedance_to_gamma(Z, Z0):
    """
    Convert impedance to reflection coefficient.

    Parameters
    ----------
    Z : complex
        Impedance.
    Z0 : complex
        Characteristic impedance.

    Returns
    -------
    complex
        Reflection coefficient.

    """
    return (Z - Z0) / (Z + Z0)

def calc_Z_off(Z0, delta_1ghz, f_Hz):
    """
    Calculate the impedance of the offset for the calibration standard. See
    M16, Eq. 20.

    Parameters
    ----------
    Z0 : float
        Characteristic impedance.
    delta_1ghz : float
        One-way loss at 1 GHz.
    f_Hz : float
        Frequency in Hz.

    Returns
    -------
    complex
        Impedance of the offset.

    
    """
    x = delta_1ghz / (4 * np.pi * f_Hz) * np.sqrt(f_Hz / 1e9)
    return Z0 + (1 - 1j) * x

def calc_l_x_gamma(Z0, delta_1ghz, delay, f_Hz):
    """
    Calculate the product of the length and propagation constant of the offset
    for the calibration standard. See M16, Eq. 21.

    Parameters
    ----------
    Z0 : float
        Characteristic impedance.
    delta_1ghz : float
        One-way loss at 1 GHz.
    delay : float
        One-way delay in seconds.
    f_Hz : float
        Frequency in Hz.

    Returns
    -------
    float
        Product of length and propagation constant.

    """
    x = delay * delta_1ghz / (2 * Z0) * np.sqrt(f_Hz / 1e9)
    return 2j * np.pi * f_Hz * delay + (1 + 1j) * x

def sparams_from_calkit(gamma_true, gamma_meas):
    """
    Get the S-parameters from an open-short-match calibration. See M16, Eq. 3.

    Parameters
    ----------
    gamma_true : array-like
        True reflection coefficients for the open, short, and match standards.
        These are the unprimed quantities in Eq. 3.
    gamma_meas : array-like
        Measured reflection coefficients for the open, short, and match. These
        are the primed quantities in Eq. 3.

    Returns
    -------
    sparams : ndarray
        S-parameters in the form [S11, S12 * S21, S22]. We only care about
        the product of S12 and S21, not their individual values.

    """
    gamma_true = np.array(gamma_true)
    gamma_meas = np.array(gamma_meas)
    # matrix to invert in eq 3
    mat = np.column_stack((np.ones(3), gamma_true, gamma_true*gamma_meas))
    sparams = np.linalg.lstsq(mat, gamma_meas, rcond=None)[0]
    sparams[1] += sparams[0] * sparams[2]  # need to do this to get S12 * S21
    return sparams



def embed_sparams(sparams, gamma):
    """
    Embed S-parameters into a network with reflection coefficient gamma. See
    M16, Eq. 1.

    Parameters
    ----------
    sparams : array-like
        S-parameters in the form [S11, S12 * S21, S22]. We only care about
        the product of S12 and S21, not their individual values.
    gamma : complex
        Intrinsic reflection coefficient.

    Returns
    -------
    gamma_prime : complex
        Embedded reflection coefficient, measured at reference plane.

    """
    s11, s12s21, s22 = sparams
    gamma_prime = s11 + s12s21 * gamma / (1 - s22 * gamma)
    return gamma_prime

def de_embed_sparams(sparams, gamma_prime):
    """
    De-embed S-parameters from a network with measured reflection coefficient
    gamma_prime. See M16, Eq. 2.

    Parameters
    ----------
    sparams : array-like
        S-parameters in the form [S11, S12 * S21, S22]. We only care about
        the product of S12 and S21, not their individual values.
    gamma_prime : complex
        Measured reflection coefficient.

    Returns
    -------
    gamma : complex
        Intrinsic reflection coefficient.

    """
    s11, s12s21, s22 = sparams
    d = gamma_prime - s11
    gamma = d / (s12s21 + s22 * d)
    return gamma

class CalStandard:

    def __init__(self, Z_ter, Z_off, l_x_gamma, Z0=50):
        """
        Useful measured and derived values for a calibration standard. Note 
        that all quantities are in SI units.

        Parameters
        ----------
        f_Hz : float
            Frequency in Hz.
        Z_ter : complex
            Impedance of termination.
        Z_off : complex
            Impedance of offset.
        l_x_gamma : float
           Product of length and propagation constant of offset, denoted
           by l and lowercase gamma in M16. Only the product shows up in
           the equations, its value can be calcuclated with Eq 21 in M16.
        Z0 : float
            Characteristic impedance.

        """
        self.Z0 = Z0
        self.Z_ter = Z_ter
        self.Z_off = Z_off
        self.l_x_gamma = l_x_gamma

    @property
    def gamma_ter(self):
        """
        Reflection coefficient for the termination.

        Returns
        -------
        complex
            Reflection coefficient.

        """
        return impedance_to_gamma(self.Z_ter, self.Z0)

    @property
    def gamma_off(self, f_Hz):
        """
        Reflection coefficient for the offset.

        Parameters
        ----------
        f_Hz : float
            Frequency in Hz.

        Returns
        -------
        complex
            Reflection coefficient.

        """
        return impedance_to_gamma(self.Z_off, self.Z0)

        
    @property
    def gamma(self):
        """
        Combined reflection coefficient for impedance of termination and lossy
        characteristic impedance of offset. See M16, Eq. 18.

        Returns
        -------
        complex
            Reflection coefficient.

        """
        
        exp = np.exp(-2j * self.l_x_gamma)
        # numerator
        num1 = self.gamma_off * (1 - exp) + self.gamma_ter * exp
        num2 = self.gamma_off**2 * self.gamma_ter
        num = num1 - num2
        # denominator
        den_bracket = self.gamma_off * exp + self.gamma_ter * (1 - exp)
        den = 1 - self.gamma_off * den_bracket
        return num / den


class CalKit:

    def __init__(self, f_Hz, Z0=50):
        """
        Useful measured and derived values for a calibration kit. Note that all
        quantities are in SI units.

        Parameters
        ----------
        f_Hz : float
            Frequency in Hz.
        Z0 : float
            Characteristic impedance.

        """
        self.Z0 = Z0
        self.f_Hz = f_Hz

    @property
    def omega(self):
        return 2 * np.pi * self.freq_Hz
    
    def _add_standard(self, Z_ter, delta_1ghz, delay):
        """
        Add a calibration standard to the calibration kit. Users should use
        the add_open and add_short methods instead.

        Parameters
        ----------
        Z_ter : float
            Impedance of termination in ohms.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        Returns
        -------
        CalStandard

        """
        Z_off = calc_Z_off(self.Z0, delta_1ghz, self.f_Hz)
        lxg = calc_l_x_gamma(self.Z0, delta_1ghz, delay, self.f_Hz)
        return CalStandard(Z_ter, Z_off, lxg, Z0=self.Z0)

    def add_open(self, C_open, delta_1ghz, delay):
        """
        Add an open standard to the calibration kit.

        Parameters
        ----------
        C_open : float
            Capacitance of open in farads.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        """
        Z_ter = -1j / (self.omega * C_open)
        self.open = self._add_standard(Z_ter, delta_1ghz, delay)

    def add_short(self, L_short, delta_1ghz, delay):
        """
        Add a short standard to the calibration kit.

        Parameters
        ----------
        L_short : float
            Inductance of short in henries.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        """
        Z_ter = 1j * self.omega * L_short
        self.short = self._add_standard(Z_ter, delta_1ghz, delay)

    def add_match(self, Z_match, delta_1ghz, delay):
        """
        Add a match standard to the calibration kit. This is usually assumed
        to be 50 ohms with no loss or delay; however, the actual values may
        differ from this assumption. See M16 for more information.

        Parameters
        ----------
        Z_match : float
            Impedance of match in ohms. This is usally assumed to be 50 ohms;
            however, MIST measures this following M16.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        """
        self.match = self._add_standard(Z_match, delta_1ghz, delay)

    def VNA_sparams(self, measured_cal_standards):
        """
        Get the VNA s-parameters using measurements of the open, short, and 
        match calibration standards.

        Parameters
        ----------
        measured_cal_standards : array-like
            Measured reflection coefficients for the open, short, and match
            standards. Shape is (3, nfreq).

        Returns
        -------
        sparams : ndarray
            S-parameters in the form [S11, S12 * S21, S22]. This can be de-
            embed to calibrate S11 measurements.

        """
        gamma_true = [self.open.gamma, self.short.gamma, self.match.gamma]
        return sparams_from_osl(gamma_true, measured_cal_standards)


class Keysight85033E(CalKit):

    def __init__(self, freq_Hz, match_resistance=50):
        """
        Nominal values for the Keysight 85033E calibration kit used in MIST.
        These values are reported in M16.

        Parameters
        ----------
        freq_Hz : float
            Frequency in Hz.
        match_resistance : float
            Resistance of match standard in ohms.

        """
        Z0 = 50

        # open standard
        C_coeffs = (-1.597e-46, 2.317e-35, -3.101e-25, 4.943e-14)
        C_open = np.polyval(C_coeffs, freq_Hz)
        open_delay = 29.243e-12
        open_loss = 2.2e9
        
        # short standard
        L_coeffs = (-1.000e-44, 2.171e-33, -1.085e-22, 2.077e-12)
        L_short = np.polyval(L_coeffs, freq_Hz)
        short_delay = 31.785e-12
        short_loss = 2.36e9

        # load / match standard
        match_delay = 38e-12  # 38 ps
        match_loss = 2.3e9

        super().__init__(freq_Hz, Z0=Z0)
        self.add_open(C_open, open_loss, open_delay)
        self.add_short(L_short, short_loss, short_delay)
        self.add_match(match_resistance, match_loss, match_delay)
