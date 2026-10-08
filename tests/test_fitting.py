import numpy as np
from numpy.testing import assert_allclose

from mistdata.fitting import FitDPSS

# S11 frequency grid of the MIST VNA, in MHz
S11_FREQ = np.arange(1, 125.25, 0.25)


def cable_like(freq):
    """
    Band-limited reflection coefficient with the 78.5 ns round-trip delay
    and |Gamma| ~ 0.93 of the MISTIC calibration cables, plus a weaker
    short reflection. Frequency in MHz, delays in microseconds.
    """
    return 0.93 * np.exp(-2j * np.pi * freq * 78.5e-3) + 0.02 * np.exp(
        -2j * np.pi * freq * 5e-3
    )


def test_dpss_predict_between_fit_points():
    # channels of the spectrometer grid that lie inside the S11 grid
    spec_freq = np.arange(4096) * 125 / 4096
    x_new = spec_freq[(spec_freq >= 1) & (spec_freq <= 125)]

    fit = FitDPSS(
        S11_FREQ, cable_like(S11_FREQ), eval_cutoff=1e-14, fc=0, fhw=0.4
    )
    fit.fit()

    assert_allclose(fit.predict(x_new), cable_like(x_new), rtol=0, atol=1e-5)


def test_fitdpss_recovers_coefficients_off_zero_centre():
    """
    At fc != 0 the design matrix is complex; least squares must use Q^H.
    The sign convention: the basis carries exp(+2 pi i (x - xc) fc).
    """
    x = np.linspace(1, 125, 497)
    rng = np.random.default_rng(3)
    a_true = rng.normal(size=40) + 1j * rng.normal(size=40)
    for fc in (0.05, -0.05):
        ref = FitDPSS(x, np.zeros(x.size, complex), nterms=40, fc=fc, fhw=0.2)
        y = ref.A @ a_true
        fit = FitDPSS(x, y, nterms=40, fc=fc, fhw=0.2)
        fit.fit()
        assert_allclose(fit.popt, a_true, rtol=0, atol=1e-10)
        assert np.sqrt(np.mean(np.abs(fit.residuals) ** 2)) < 1e-10


def test_least_squares_matches_lstsq_complex():
    from mistdata.fitting import least_squares

    rng = np.random.default_rng(4)
    A = rng.normal(size=(60, 12)) + 1j * rng.normal(size=(60, 12))
    y = rng.normal(size=60) + 1j * rng.normal(size=60)
    for sigma in (0.3, rng.uniform(0.5, 2, size=60)):
        w = np.ones(60) / sigma
        want = np.linalg.lstsq(w[:, None] * A, w * y, rcond=None)[0]
        assert_allclose(least_squares(A, y, sigma), want, rtol=1e-12)


def test_fitdpss_fc_sign_convention():
    """
    A reflection exp(-2 pi i x tau) (a delay tau > 0) is centred by
    fc = -tau: the basis carries exp(+2 pi i (x - xc) fc). The tone is fit
    to the noise floor at fc = -tau and not at all at fc = +tau.
    """
    x = np.linspace(1, 125, 497)
    tau = 0.1
    y = np.exp(-2j * np.pi * x * tau)
    rms = {}
    for fc in (-tau, tau):
        fit = FitDPSS(x, y, nterms=20, fc=fc, fhw=0.05)
        fit.fit()
        rms[fc] = np.sqrt(np.mean(np.abs(fit.residuals) ** 2))
    assert rms[-tau] < 1e-3
    assert rms[tau] > 0.5
