"""Tests for mistdata.calibration."""

import numpy as np
import pytest

from mistdata.calibration import MISTCalibration


def make_cal(gamma_a, gamma_r):
    """A MISTCalibration carrying only what k_params reads.

    ``k_params`` is a cached_property that uses ``gamma_a`` and ``gamma_r``
    and nothing else, so it can be exercised without a MISTData object.
    """
    cal = object.__new__(MISTCalibration)
    cal.gamma_a = gamma_a
    cal.gamma_r = gamma_r
    return cal


def m24_k_params(gamma_a, gamma_r):
    """The K parameters of Monsalve et al. 2024 (M24), written out.

    These are M24's equations for K_0, K_U, K_C, K_S, F and alpha, the ones
    ``MISTCalibration.k_params`` says it implements. Transcribed here
    independently so the test checks the implementation rather than
    restating it.

    Note the first power of |Gamma_A| in K_C and K_S: M24 has

        K_C = |Gamma_A| / ((1 - |Gamma_A|^2) |F|) cos(alpha),

    not |Gamma_A|^2. The same first power appears in Monsalve et al. 2017
    eq. 7 and in the EDGES reference implementation
    (``edges.cal.noise_waves.get_K``), and it must: T_U, T_C and T_S are
    properties of the receiver, so a factor that depends on the source's
    reflection coefficient cannot be absorbed into them.
    """
    xa = 1 - np.abs(gamma_a) ** 2
    xr = 1 - np.abs(gamma_r) ** 2
    F = np.sqrt(xr) / (1 - gamma_a * gamma_r)
    alpha = np.angle(gamma_a * F)
    prefactor = np.abs(gamma_a) / (xa * np.abs(F))
    return {
        "k0": xr / (xa * np.abs(F) ** 2),
        "kU": np.abs(gamma_a) ** 2 / xa,
        "kC": prefactor * np.cos(alpha),
        "kS": prefactor * np.sin(alpha),
        "F": F,
        "alpha": alpha,
    }


# |Gamma_A| across the MIST calibrators: ambient and hot loads are well
# matched, the open and shorted cables are not, and the antenna sits between.
GAMMA_A_MAGNITUDES = [0.0, 0.01, 0.05, 0.11, 0.5, 0.8, 0.95]


@pytest.mark.parametrize("mag", GAMMA_A_MAGNITUDES)
def test_k_params_match_m24(mag):
    """k_params reproduces M24 eqs. for K_0, K_U, K_C and K_S."""
    freq = np.linspace(23e6, 107e6, 401)
    gamma_a = mag * np.exp(2j * np.pi * freq / 40e6)
    gamma_r = 0.1 * np.exp(-2j * np.pi * freq / 55e6)

    got = make_cal(gamma_a, gamma_r).k_params
    want = m24_k_params(gamma_a, gamma_r)
    for key in ("k0", "kU", "kC", "kS", "F", "alpha"):
        assert np.allclose(got[key], want[key], rtol=1e-12, atol=0), key


def test_kc_ks_are_first_order_in_gamma_a():
    """K_C and K_S scale as |Gamma_A|, not |Gamma_A|^2.

    A direct, human-legible guard on the bug this file was written for.
    Since K_C and K_S share the prefactor |Gamma_A| / ((1-|Gamma_A|^2)|F|)
    and carry cos(alpha) and sin(alpha), their quadrature sum is that
    prefactor alone, which removes the phase and makes the power of
    |Gamma_A| directly testable. Halving |Gamma_A| in the small-|Gamma_A|
    limit, where 1/(1-|Gamma_A|^2) and |F| barely move, must halve it; the
    buggy second power would quarter it instead.

    Ratioing K_C itself will not do: alpha sweeps through +-pi/2 across the
    band, and near those frequencies cos(alpha) ~ 0, so its ratio is
    ill-conditioned however correct the code is.
    """
    freq = np.linspace(23e6, 107e6, 401)
    phase = np.exp(2j * np.pi * freq / 40e6)
    gamma_r = 0.1 * np.exp(-2j * np.pi * freq / 55e6)

    def prefactor(mag):
        k = make_cal(mag * phase, gamma_r).k_params
        return np.hypot(k["kC"], k["kS"])

    # rtol 5e-3, not tighter: |F| itself moves with Gamma_A at the
    # |Gamma_A Gamma_R| ~ 2e-3 level, so the ratio spans 1.9995 to 2.0026.
    # The buggy second power gives 4, so the margin here is a factor ~100.
    assert np.allclose(prefactor(0.02) / prefactor(0.01), 2.0, rtol=5e-3)
    # kU is genuinely second order, which is the contrast that matters
    big = make_cal(0.02 * phase, gamma_r).k_params
    small = make_cal(0.01 * phase, gamma_r).k_params
    assert np.allclose(big["kU"] / small["kU"], 4.0, rtol=5e-3)


def test_k_params_match_m17_noise_wave_coefficients():
    """k_params agrees with the M17 form used by the EDGES implementation.

    ``edges.cal.noise_waves.get_K`` returns K1..K4, the coefficients of
    T_ant, T_unc, T_cos and T_sin in M17 eq. 7. Our model is written with
    the T_ant coefficient divided out, so k0 = 1/K1, kU = K2/K1, kC = K3/K1
    and kS = K4/K1. Transcribed from that function, this is an independent
    check of the same physics through a different arrangement of it.
    """
    freq = np.linspace(23e6, 107e6, 401)
    gamma_a = 0.3 * np.exp(2j * np.pi * freq / 33e6)
    gamma_r = 0.1 * np.exp(-2j * np.pi * freq / 55e6)

    gain = 1 - np.abs(gamma_r) ** 2
    F = np.sqrt(gain) / (1 - gamma_a * gamma_r)
    alpha = np.angle(gamma_a * F)
    f_ratio = np.abs(F)
    fgant = np.abs(gamma_a) * f_ratio / gain

    K2 = fgant**2 * gain
    K1 = f_ratio**2 / gain - K2
    K3 = fgant * np.cos(alpha)
    K4 = fgant * np.sin(alpha)

    got = make_cal(gamma_a, gamma_r).k_params
    assert np.allclose(got["k0"], 1 / K1, rtol=1e-12, atol=0)
    assert np.allclose(got["kU"], K2 / K1, rtol=1e-12, atol=0)
    assert np.allclose(got["kC"], K3 / K1, rtol=1e-12, atol=0)
    assert np.allclose(got["kS"], K4 / K1, rtol=1e-12, atol=0)
