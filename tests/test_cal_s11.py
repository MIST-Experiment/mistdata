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
