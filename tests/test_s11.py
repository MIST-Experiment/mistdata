import types

import numpy as np
from numpy.testing import assert_allclose

from mistdata import MISTCalibration
from mistdata.DUTLNA import DUTLNA
from mistdata.s11 import ReceiverS11

FREQ = np.linspace(1, 125, 497)  # MHz


def embed(sparams, gamma):
    """M16 eq. 1: Gamma at port 2 seen from port 1, S = [S11, S12S21, S22]."""
    s11, s12s21, s22 = sparams
    return s11 + s12s21 * gamma / (1 - s22 * gamma)


def network(delay, refl):
    """Reciprocal two-port, [S11, S12 S21, S22], delay in microseconds."""
    s21 = 0.9 * np.exp(-2j * np.pi * FREQ * delay)
    return np.array(
        [
            refl * np.exp(1j * FREQ / 30),
            s21**2,
            -refl * np.exp(-1j * FREQ / 50),
        ]
    )


VNA = network(3e-3, 0.08)  # VNA error network
PATH_B = network(5e-3, 0.03)  # internal reference plane -> PSD output
PATH_C = network(2e-3, 0.02)  # receiver input -> PSD output
GAMMA_L = 0.1 * np.exp(-2j * np.pi * FREQ * 4e-3)  # at the PSD output


def make_dut(gamma_l):
    """Raw LNA-side sweep: standards and the LNA seen through B and the VNA."""
    ones = np.ones_like(gamma_l)
    return DUTLNA(
        freq=FREQ,
        open_=embed(VNA, ones),
        short=embed(VNA, -ones),
        match=embed(VNA, 0 * ones),
        lna=embed(VNA, embed(PATH_B, gamma_l)),
    )


def test_receiver_s11_recovers_lna_reflection():
    """
    Calibrate with the internal OSL standards, de-embed path B, embed path
    C (Monsalve et al. 2024): the result is the LNA reflection seen from
    the receiver input, for one sweep and for two stacked sweeps.
    """
    want = embed(PATH_C, GAMMA_L)
    got = ReceiverS11(make_dut(GAMMA_L), PATH_B, PATH_C).s11
    assert_allclose(got, want, rtol=0, atol=1e-12)
    two = np.array([GAMMA_L, 0.5 * GAMMA_L])
    got2 = ReceiverS11(make_dut(two), PATH_B, PATH_C).s11
    assert got2.shape == (2, FREQ.size)
    assert_allclose(got2, embed(PATH_C, two), rtol=0, atol=1e-12)


def test_calibration_uses_paths_b_and_c():
    """
    MISTCalibration computes the receiver S11 from paths B and C when
    cal_data has no 'gamma_r'.
    """
    nspec, nfreq = 4, 64
    spec_freq = np.linspace(25, 105, nfreq)
    mistdata = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            freq=spec_freq,
            psd_antenna=np.ones((nspec, nfreq)),
            psd_ambient=np.ones((nspec, nfreq)),
            psd_noise_source=np.full((nspec, nfreq), 2.0),
        ),
        dut_recin=types.SimpleNamespace(s11_freq=FREQ),
        dut_lna=make_dut(GAMMA_L),
    )
    cal_data = {
        "gamma_a": {"antenna": 0.1 * np.ones(FREQ.size, complex)},
        "pathB_sparams": PATH_B,
        "pathC_sparams": PATH_C,
        "nw_params": {"TU": 0, "TC": 0, "TS": 0},
        "C_params": {"C1": 1, "C2": 0},
    }
    cal = MISTCalibration(mistdata, cal_data)
    assert_allclose(
        cal._gamma_r[0, 0], embed(PATH_C, GAMMA_L), rtol=0, atol=1e-12
    )


def test_calibration_paths_b_and_c_with_two_files():
    """
    With one VNA sweep per file, the receiver S11 from paths B and C has one
    row per file, like the antenna S11, and the S11 fits and k parameters
    broadcast per file.
    """
    nfiles, per_file, nfreq = 2, 3, 64
    spec_freq = np.linspace(25, 105, nfreq)
    gamma_l = np.array([GAMMA_L, 0.5 * GAMMA_L])
    mistdata = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            freq=spec_freq,
            psd_antenna=np.ones((nfiles * per_file, nfreq)),
            psd_ambient=np.ones((nfiles * per_file, nfreq)),
            psd_noise_source=np.full((nfiles * per_file, nfreq), 2.0),
        ),
        dut_recin=types.SimpleNamespace(s11_freq=FREQ),
        dut_lna=make_dut(gamma_l),
    )
    gamma_a = 0.1 * np.ones((nfiles, FREQ.size), complex)
    cal_data = {
        "gamma_a": {"antenna": gamma_a},
        "pathB_sparams": PATH_B,
        "pathC_sparams": PATH_C,
        "nw_params": {"TU": 0, "TC": 0, "TS": 0},
        "C_params": {"C1": 1, "C2": 0},
    }
    cal = MISTCalibration(mistdata, cal_data)
    assert cal._gamma_r.shape == (nfiles, 1, FREQ.size)
    assert_allclose(
        cal._gamma_r[:, 0], embed(PATH_C, gamma_l), rtol=0, atol=1e-12
    )
    cal.fit_s11("antenna", nterms=8)
    cal.fit_s11("receiver", nterms=20)
    assert cal.gamma_r.shape == (nfiles, 1, nfreq)
    assert cal.k_params["k0"].shape == (nfiles, 1, nfreq)
