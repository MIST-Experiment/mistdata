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

def test_keysight():
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    Z_match = 50.025
    calkit = cal_s11.Keysight85033E(f_Hz, match_resistance=Z_match)
    assert np.allclose(calkit.Z0, 50)
    Z0 = calkit.Z0

    # values from Monsalve et al 2016 / EDGES memo 12
    C_open = 4.943e-14 - 3.101e-25 * f_Hz + 2.317e-35 * f_Hz**2 - 1.597e-46 * f_Hz**3
    L_short = 2.077e-12 - 10.85e-22 * f_Hz + 2.171e-33 * f_Hz**2 - 1.000e-44 * f_Hz**3

    # open
    delay = 29.242e-12
    delta = 2.2e9
    Z_off = Z0 + (1-1j) * delta / (4 * np.pi * f_Hz) * np.sqrt(f_Hz/1e9)
    gamma_off = cal_s11.impedance_to_gamma(Z_off, Z0)
    Z_ter = -1j/(2*np.pi*f_Hz*C_open)
    gamma_ter = cal_s11.impedance_to_gamma(Z_ter, Z0)
    # eq 21 in Monsalve et al 2016
    gamma_l = 1j * 2 * np.pi * f_Hz * delay + (1+1j) * delay * delta / (2 * Z0) * np.sqrt(f_Hz/1e9)
    gamma = gamma_off * (1 - np.exp(-2 * gamma_l) - gamma_off * gamma_ter) + gamma_ter * np.exp(-2 * gamma_l)
    gamma /= 1 - gamma_off * (np.exp(-2 * gamma_l) * gamma_off + gamma_ter * (1 - np.exp(-2 * gamma_l)))
    assert np.allclose(calkit.open.gamma_ter, gamma_ter)
    assert np.allclose(calkit.open.gamma_off, gamma_off)
    assert np.allclose(calkit.open.gamma, gamma)

    # short
    delay = 31.785e-12
    delta = 2.36e9
    Z_off = Z0 + (1-1j) * delta / (4 * np.pi * f_Hz) * np.sqrt(f_Hz/1e9)
    gamma_off = cal_s11.impedance_to_gamma(Z_off, Z0)
    Z_ter = 1j*2*np.pi*f_Hz*L_short
    gamma_ter = cal_s11.impedance_to_gamma(Z_ter, Z0)
    # eq 21 in Monsalve et al 2016
    gamma_l = 1j * 2 * np.pi * f_Hz * delay + (1+1j) * delay * delta / (2 * Z0) * np.sqrt(f_Hz/1e9)
    gamma = gamma_off * (1 - np.exp(-2 * gamma_l) - gamma_off * gamma_ter) + gamma_ter * np.exp(-2 * gamma_l)
    gamma /= 1 - gamma_off * (np.exp(-2 * gamma_l) * gamma_off + gamma_ter * (1 - np.exp(-2 * gamma_l)))
    assert np.allclose(calkit.short.gamma_ter, gamma_ter)
    assert np.allclose(calkit.short.gamma_off, gamma_off)
    assert np.allclose(calkit.short.gamma, gamma)

    # match
    delay = 38.8e-12
    delta = 2.3e9
    Z_off = Z0 + (1-1j) * delta / (4 * np.pi * f_Hz) * np.sqrt(f_Hz/1e9)
    gamma_off = cal_s11.impedance_to_gamma(Z_off, Z0)
    Z_ter = Z_match
    gamma_ter = cal_s11.impedance_to_gamma(Z_ter, Z0)
    # eq 21 in Monsalve et al 2016
    gamma_l = 1j * 2 * np.pi * f_Hz * delay + (1+1j) * delay * delta / (2 * Z0) * np.sqrt(f_Hz/1e9)
    gamma = gamma_off * (1 - np.exp(-2 * gamma_l) - gamma_off * gamma_ter) + gamma_ter * np.exp(-2 * gamma_l)
    gamma /= 1 - gamma_off * (np.exp(-2 * gamma_l) * gamma_off + gamma_ter * (1 - np.exp(-2 * gamma_l)))
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
    
