import types

import numpy as np
from numpy.testing import assert_allclose

from mistdata import MISTCalibration, NoiseWave

FREQ = np.linspace(25, 105, 321)  # MHz
X = (FREQ - 65) / 40

# true calibration parameters, smooth in frequency
TRUE_C = {"C1": 1.05 + 0.02 * X**2, "C2": -1.5 + 0.3 * X}
TRUE_NW = {"TU": 196 + X**3, "TC": -17 - 8 * X, "TS": 1 + 10 * X**2}
# placeholders the calibrators are constructed with; solve must not use them
PLACEHOLDER_C = {"C1": 1, "C2": 0}
PLACEHOLDER_NW = {"TU": 0, "TC": 0, "TS": 0}

GAMMA_R = 0.05 * np.exp(-2j * np.pi * FREQ * 2e-3)  # 2 ns delay
CABLE_DELAY = 78.5e-3  # microseconds, as measured for the MISTIC cables
GAMMA = {
    "hot": np.full(FREQ.size, 0.004 * np.exp(0.3j)),
    "ambient": np.full(FREQ.size, 0.003 * np.exp(-0.5j)),
    "open": 0.93 * np.exp(-2j * np.pi * FREQ * CABLE_DELAY),
    "short": -0.92 * np.exp(-2j * np.pi * FREQ * CABLE_DELAY),
}
TEMPS = {"hot": 390.9, "ambient": 299.2, "open": 299.2, "short": 299.2}
# unequal number of spectra per calibrator, as in the 2024-10-24 data
NSPEC = {"hot": 97, "ambient": 92, "open": 65, "short": 160}


def make_cal(gamma_a, psd_antenna, nw_params, C_params):
    """
    MISTCalibration on a minimal stand-in for MISTData that holds only the
    fields MISTCalibration reads when the S11s are supplied directly.
    psd_antenna has shape (nspec, nfreq); the load and load + noise source
    PSDs are constant.
    """
    shape = psd_antenna.shape
    mistdata = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            freq=FREQ,
            psd_antenna=psd_antenna.copy(),
            psd_ambient=np.ones(shape),
            psd_noise_source=np.full(shape, 2.0),
        ),
        dut_recin=types.SimpleNamespace(s11_freq=FREQ),
    )
    cal_data = {
        "gamma_a": {"antenna": gamma_a},
        "gamma_r": GAMMA_R,
        "nw_params": nw_params,
        "C_params": C_params,
    }
    cal = MISTCalibration(mistdata, cal_data)
    # S11s are already on the spectral grid, stand in for fit_s11
    cal.gamma_a = cal._gamma_a
    cal.gamma_r = cal._gamma_r
    return cal


def true_psd(name):
    """
    Antenna PSD that the true parameters calibrate to the physical
    temperature of the calibrator. Antenna temperature is affine in the
    PSD, so evaluate MISTCalibration at PSD = 0 and 1 and invert.
    """
    t0, t1 = (
        make_cal(
            GAMMA[name], np.full((1, FREQ.size), p), TRUE_NW, TRUE_C
        ).antenna_temp[0, 0]
        for p in (0.0, 1.0)
    )
    return (TEMPS[name] - t0) / (t1 - t0)


def assert_true_params(C_params, nw_params):
    for key, val in TRUE_C.items():
        assert_allclose(C_params[key], val, rtol=1e-6)
    for key, val in TRUE_NW.items():
        assert_allclose(nw_params[key], val, rtol=1e-6, atol=1e-6)


def test_solve_recovers_true_parameters():
    cals = {}
    for name in GAMMA:
        psd = np.tile(true_psd(name), (NSPEC[name], 1))
        cals[name] = make_cal(GAMMA[name], psd, PLACEHOLDER_NW, PLACEHOLDER_C)

    C_params, nw_params = NoiseWave(cals, TEMPS).solve()

    assert_true_params(C_params, nw_params)


def test_solve_averages_spectra_over_time():
    # scatter each spectrum by +d, -d, 0, ... so only the time average is
    # the true PSD
    cals = {}
    for name in GAMMA:
        nspec = 3 * (NSPEC[name] // 3)
        scatter = np.tile([0.2, -0.2, 0.0], nspec // 3)[:, None]
        psd = true_psd(name) * (1 + scatter)
        cals[name] = make_cal(GAMMA[name], psd, PLACEHOLDER_NW, PLACEHOLDER_C)

    C_params, nw_params = NoiseWave(cals, TEMPS).solve()

    assert_true_params(C_params, nw_params)
