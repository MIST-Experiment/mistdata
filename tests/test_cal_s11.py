import numpy as np

from mistdata import cal_s11


def test_impedance_to_gamma():
    # perfect match gives 0 reflection
    Z = 50
    Z0 = 50
    gamma = cal_s11.impedance_to_gamma(Z, Z0)
    assert np.isclose(gamma, 0)

    # open circuit, reflection is 1 with phase 0
    Z = np.inf
    assert np.isclose(cal_s11.impedance_to_gamma(Z, Z0), 1)

    # short circuit, reflection is 1 with phase 180
    Z = 0
    assert np.isclose(cal_s11.impedance_to_gamma(Z, Z0), -1)


def test_calc_Z_off():
    Z0 = 50
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz

    # no loss gives Zoff = Z0
    assert np.allclose(cal_s11.calc_Z_off(Z0, 0, f_Hz), Z0)

    # lossy line
    delta = 2e9  # loss
    Zoff = cal_s11.calc_Z_off(Z0, delta, f_Hz)
    dZ = Zoff - Z0
    # real and imag parts are equal in magnitude, opposite in sign
    assert np.allclose(dZ.real, -dZ.imag)
    # dZ scale with 1/sqrt(f)
    assert np.allclose(dZ, dZ[0] * np.sqrt(f_Hz[0] / f_Hz))


def test_calc_l_x_gamma():
    Z0 = 50
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    omega = 2 * np.pi * f_Hz

    # no loss means l*gamma is just due to delay
    delay = 30e-12  # 30 ps
    assert np.allclose(
        cal_s11.calc_l_x_gamma(Z0, 0, delay, f_Hz), 1j * omega * delay
    )

    # lgamma is 0 if there's no delay
    delta = 2e9  # loss
    assert np.allclose(cal_s11.calc_l_x_gamma(Z0, delta, 0, f_Hz), 0)

    # generic case with loss and delay
    lxg_dly = 1j * omega * delay  # l*gamma due to delay
    d = cal_s11.calc_l_x_gamma(Z0, delta, delay, f_Hz) - lxg_dly
    assert np.allclose(d.real, d.imag)  # remainder has equal real/imag
    # and scale with sqrt(f)
    assert np.allclose(d, d[0] * np.sqrt(f_Hz / f_Hz[0]))


def test_gamma():
    Z0 = 50
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    Z_ters = [0, Z0, np.inf, 125]  # short, match, open, random
    # no loss means gamma = gamma_termination
    Z_off = Z0
    for Z_ter in Z_ters:
        cal_std = cal_s11.CalStandard(Z_ter, Z_off, 0, Z0=Z0)
        assert np.allclose(cal_std.gamma, cal_std.gamma_ter)

    # infinite loss gives gamma = gamma_offset
    Z_off = 1e12 / np.sqrt(f_Hz) * (1 - 1j)  # very high loss
    for Z_ter in Z_ters:
        cal_std = cal_s11.CalStandard(Z_ter, Z_off, np.inf, Z0=Z0)
        assert np.allclose(cal_std.gamma, cal_std.gamma_off)

    # Z_ter = Z0
    delta = 2e9  # loss
    delay = 30e-12  # delay
    Z_off = cal_s11.calc_Z_off(Z0, delta, f_Hz)
    lxg = cal_s11.calc_l_x_gamma(Z0, delta, delay, f_Hz)
    cal_std = cal_s11.CalStandard(Z0, Z_off, lxg, Z0=Z0)
    exp = np.exp(-2 * lxg)
    gamma_off = cal_std.gamma_off
    gamma_expected = gamma_off * (1 - exp) / (1 - gamma_off**2 * exp)
    assert np.allclose(cal_std.gamma, gamma_expected)

    # Z_off = Z0
    for Z_ter in Z_ters:
        cal_std = cal_s11.CalStandard(Z_ter, Z0, lxg, Z0=Z0)
        assert np.allclose(cal_std.gamma, cal_std.gamma_ter * exp)

def test_calkit():
    # nominal values
    delta = 2e9  # loss
    delay = 30e-12  # delay
    Z0 = 50  # characteristic impedance
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    calkit = cal_s11.CalKit(f_Hz, Z0=Z0)
    assert np.allclose(calkit.Z0, Z0)
    assert np.allclose(calkit.omega, 2 * np.pi * f_Hz)
    # eq 20 in Monsalve et al 2016
    Z_off = Z0 + (1-1j) * delta / (4 * np.pi * f_Hz) * np.sqrt(f_Hz/1e9)
    gamma_off = cal_s11.impedance_to_gamma(Z_off, Z0)

    # ideal case for open: C_open = 0 -> gamma_open = 1
    calkit.add_open(0, 0, 0)
    assert np.all(calkit.open.gamma == 1)
    assert np.all(calkit.open.gamma_off == 0)
    assert np.all(calkit.open.gamma_ter == 1)

    # ideal case for short: L_short = 0 -> gamma_short = -1
    calkit.add_short(0, 0, 0)
    assert np.all(calkit.short.gamma == -1)
    assert np.all(calkit.short.gamma_off == 0)
    assert np.all(calkit.short.gamma_ter == -1)

    # ideal case for match: Z_match = Z0 -> gamma_match = 0
    calkit.add_match(Z0, 0, 0)
    assert np.all(calkit.match.gamma == 0)
    assert np.all(calkit.match.gamma_off == 0)
    assert np.all(calkit.match.gamma_ter == 0)

    # frequency dependent C_open
    C_open = 1e-15 + 1e-27 * f_Hz + 1e-36 * f_Hz**2 + 1e-45 * f_Hz**3
    calkit.add_open(C_open, delta, delay)
    # eq 22 in Monsalve et al 2016
    Z_open = -1j/(2*np.pi*f_Hz*C_open)
    gamma_ter_open = cal_s11.impedance_to_gamma(Z_open, Z0)
    assert np.allclose(calkit.open.gamma_ter, gamma_ter_open)
    assert np.allclose(calkit.open.gamma_off, gamma_off)
    
    # frequency dependent L_short
    L_short = 1e-12 + 1e-24 * f_Hz + 1e-33 * f_Hz**2 + 1e-42 * f_Hz**3
    calkit.add_short(L_short, delta, delay)
    # eq 23 in Monsalve et al 2016
    Z_short = 1j*2*np.pi*f_Hz*L_short
    gamma_ter_short = cal_s11.impedance_to_gamma(Z_short, Z0)
    assert np.allclose(calkit.short.gamma_ter, gamma_ter_short)
    assert np.allclose(calkit.short.gamma_off, gamma_off)

    # match with loss and delay and Z != Z0
    Z_match = Z0 * 1.05  # 5% mismatch
    calkit.add_match(Z_match, delta, delay)
    gamma_ter_match = cal_s11.impedance_to_gamma(Z_match, Z0)
    assert np.allclose(calkit.match.gamma_ter, gamma_ter_match)
    assert np.allclose(calkit.match.gamma_off, gamma_off)

# Standard definitions for the Keysight/Agilent 85033E 3.5 mm kit (plug).
#
# C and L coefficients are from Keysight, the load's offset delay is different
# following M16 and Monsalve et al. 2024. M16 measured 38.8ps +- 2.1ps.
# We're using 38ps.
OPEN_DELAY = 29.243e-12
OPEN_LOSS = 2.2e9
OPEN_C_COEFFS = (-0.1597e-45, 23.17e-36, -310.1e-27, 49.43e-15)

SHORT_DELAY = 31.785e-12
SHORT_LOSS = 2.36e9
SHORT_L_COEFFS = (-0.01e-42, 2.171e-33, -108.5e-24, 2.077e-12)

MATCH_DELAY = 38e-12  # Keysight nominal is 0 ps
MATCH_LOSS = 2.3e9


def m16_standard_gamma(f_Hz, Z0, delay, delta, Z_ter):
    """Reflection coefficient of an offset standard, M16 eq. 18.

    ``gamma_l`` below is M16 eq. 21 and ``Z_off`` is eq. 20. Written out
    independently of ``cal_s11`` so that the test checks the implementation
    rather than restating it.
    """
    Z_off = Z0 + (1 - 1j) * delta / (4 * np.pi * f_Hz) * np.sqrt(f_Hz / 1e9)
    gamma_off = cal_s11.impedance_to_gamma(Z_off, Z0)
    gamma_ter = cal_s11.impedance_to_gamma(Z_ter, Z0)
    gamma_l = 1j * 2 * np.pi * f_Hz * delay + (1 + 1j) * delay * delta / (
        2 * Z0
    ) * np.sqrt(f_Hz / 1e9)
    e = np.exp(-2 * gamma_l)
    gamma = gamma_off * (1 - e - gamma_off * gamma_ter) + gamma_ter * e
    gamma /= 1 - gamma_off * (e * gamma_off + gamma_ter * (1 - e))
    return gamma_off, gamma_ter, gamma


def test_keysight_standard_definitions():
    """The kit carries the published 85033E values.

    Separate from the equation test below, and deliberately tight: a parameter
    that drifts from the reference must fail here, naming the standard, rather
    than surfacing as an unexplained mismatch in a reflection coefficient. Two
    of the three errors this file previously carried -- the short's L1
    coefficient, off by a factor of 10, and the open's delay, off by 1 fs --
    moved ``gamma`` by less than ``np.allclose``'s default tolerance and so
    went unnoticed for the life of the test.

    ``CalStandard`` keeps the parameters only through ``Z_ter`` (the
    termination, i.e. C, L or R), ``Z_off`` (the offset, i.e. the loss) and
    ``l_x_gamma`` (delay and loss together), so those are what is checked. The
    expressions are written out rather than taken from ``cal_s11`` so that this
    test does not restate the code it is checking.

    Every check is at ``rtol=1e-12`` with ``atol=0``: the constants above are
    the kit's own values, confirmed parameter by parameter against the EDGES
    reference implementation, so there is no rounding to leave room for.

    ``atol=0`` matters. ``np.allclose``'s default ``atol`` is 1e-8, which is
    not scale-free, and the short's ``Z_ter`` is only ~1.6e-3 ohm at 125 MHz,
    so the default would swamp a relative error of a few parts per million in
    L -- the same way the default tolerance hid the factor-of-10 L1 error in
    the first place.
    """
    f_Hz = np.arange(1, 126) * 1e6
    Z_match = 50.025
    calkit = cal_s11.Keysight85033E(f_Hz, match_resistance=Z_match)
    Z0 = calkit.Z0
    rtol = 1e-12

    def same(got, want):
        # atol=0: see the docstring
        assert np.allclose(got, want, rtol=rtol, atol=0)

    def Z_off(loss):
        return Z0 + (1 - 1j) * loss / (4 * np.pi * f_Hz) * np.sqrt(f_Hz / 1e9)

    def l_x_gamma(loss, delay):
        return 2j * np.pi * f_Hz * delay + (1 + 1j) * delay * loss / (
            2 * Z0
        ) * np.sqrt(f_Hz / 1e9)

    C_open = np.polyval(OPEN_C_COEFFS, f_Hz)
    same(calkit.open.Z_ter, -1j / (2 * np.pi * f_Hz * C_open))
    same(calkit.open.Z_off, Z_off(OPEN_LOSS))
    same(calkit.open.l_x_gamma, l_x_gamma(OPEN_LOSS, OPEN_DELAY))

    L_short = np.polyval(SHORT_L_COEFFS, f_Hz)
    same(calkit.short.Z_ter, 1j * 2 * np.pi * f_Hz * L_short)
    same(calkit.short.Z_off, Z_off(SHORT_LOSS))
    same(calkit.short.l_x_gamma, l_x_gamma(SHORT_LOSS, SHORT_DELAY))

    same(calkit.match.Z_ter, Z_match)
    same(calkit.match.Z_off, Z_off(MATCH_LOSS))
    same(calkit.match.l_x_gamma, l_x_gamma(MATCH_LOSS, MATCH_DELAY))


def test_keysight():
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    Z_match = 50.025
    calkit = cal_s11.Keysight85033E(f_Hz, match_resistance=Z_match)
    assert np.allclose(calkit.Z0, 50)
    Z0 = calkit.Z0

    C_open = np.polyval(OPEN_C_COEFFS, f_Hz)
    L_short = np.polyval(SHORT_L_COEFFS, f_Hz)

    # open
    gamma_off, gamma_ter, gamma = m16_standard_gamma(
        f_Hz, Z0, OPEN_DELAY, OPEN_LOSS, -1j / (2 * np.pi * f_Hz * C_open)
    )
    assert np.allclose(calkit.open.gamma_ter, gamma_ter)
    assert np.allclose(calkit.open.gamma_off, gamma_off)
    assert np.allclose(calkit.open.gamma, gamma)

    # short
    gamma_off, gamma_ter, gamma = m16_standard_gamma(
        f_Hz, Z0, SHORT_DELAY, SHORT_LOSS, 1j * 2 * np.pi * f_Hz * L_short
    )
    assert np.allclose(calkit.short.gamma_ter, gamma_ter)
    assert np.allclose(calkit.short.gamma_off, gamma_off)
    assert np.allclose(calkit.short.gamma, gamma)

    # match
    gamma_off, gamma_ter, gamma = m16_standard_gamma(
        f_Hz, Z0, MATCH_DELAY, MATCH_LOSS, Z_match
    )
    assert np.allclose(calkit.match.gamma_ter, gamma_ter)
    assert np.allclose(calkit.match.gamma_off, gamma_off)
    assert np.allclose(calkit.match.gamma, gamma)


def test_network_sparams():
    Nfreq = 125
    # ideal values for open, short, match
    gamma_true = np.empty((3, Nfreq), dtype=complex)
    gamma_true[0] = 1  # open
    gamma_true[1] = -1  # short
    gamma_true[2] = 0  # match
    # case with ideal values, network sparams no influence
    gamma_meas = np.empty((3, Nfreq), dtype=complex)
    gamma_meas[0] = 1  # open
    gamma_meas[1] = -1  # short
    gamma_meas[2] = 0  # match
    sparams = cal_s11.network_sparams(gamma_true, gamma_meas)
    # in this case s11=s22=0, s12=s21=1
    assert np.all(sparams[0] == 0)
    assert np.all(sparams[1] == 1)
    assert np.all(sparams[2] == 0)
    
