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


def _prolate_count(nf, nw, cutoff):
    """Eigenvalues >= cutoff of the prolate matrix (Slepian 1978)."""
    W = nw / nf
    m = np.arange(nf)
    d = m[:, None] - m[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        B = np.sin(2 * np.pi * W * d) / (np.pi * d)
    B[d == 0] = 2 * W
    return int(np.count_nonzero(np.linalg.eigvalsh(B) >= cutoff))


def test_nterms_counts_eigenvalues_at_or_above_cutoff():
    """
    get_nterms_DPSS returns the number of DPSS vectors whose eigenvalue is
    >= the cutoff. Before the fix it returned the index of the last one,
    one fewer (7, 10, 14 here).
    """
    from scipy.signal import windows

    from mistdata.fitting import get_nterms_DPSS

    x = np.arange(64.0)
    fhw = 4 / 63
    nw = (x[-1] - x[0]) * fhw  # NW = 4, as get_nterms_DPSS computes it
    evals = windows.dpss(64, nw, Kmax=64, return_ratios=True)[1]
    for cutoff, want in [(0.5, 8), (1e-3, 11), (1e-8, 15)]:
        n = get_nterms_DPSS(x, fhw, cutoff)
        assert n == want == np.count_nonzero(evals >= cutoff)
        assert evals[n - 1] >= cutoff > evals[n]


def test_nterms_vna_grid():
    """
    Term counts on the MIST VNA grid (497 points, 1-125 MHz) for four
    (eval_cutoff, fhw) pairs, cross-checked against the eigenvalues of the
    prolate matrix. At cutoff 1e-14 the boundary eigenvalues (2.8e-14,
    3.6e-15) are near the eigvalsh floor (~1e-15), so that case allows
    +-1 and the others are exact.
    """
    from mistdata.fitting import get_nterms_DPSS

    x = np.linspace(1, 125, 497)
    bw = x[-1] - x[0]
    cases = [
        (1e-14, 0.4, 118, 1),
        (1e-12, 0.4, 116, 0),
        (1e-6, 0.6, 158, 0),
        (1e-10, 0.1, 36, 0),
    ]
    for cutoff, fhw, want, tol in cases:
        assert abs(get_nterms_DPSS(x, fhw, cutoff) - want) <= tol
        assert abs(_prolate_count(x.size, bw * fhw, cutoff) - want) <= tol


def test_fitdpss_eval_cutoff_uses_all_qualifying_vectors():
    x = S11_FREQ
    fit = FitDPSS(x, cable_like(x), eval_cutoff=1e-6, fhw=0.6)
    assert fit.nterms == 158 and fit.A.shape == (x.size, 158)


def test_nterms_raises_when_none_qualify():
    import pytest
    from scipy.signal import windows

    from mistdata.fitting import get_nterms_DPSS

    lam0 = windows.dpss(64, 0.5, Kmax=1, return_ratios=True)[1][0]
    assert lam0 < 0.99
    with pytest.raises(ValueError):
        get_nterms_DPSS(np.arange(64.0), 0.5 / 63, 0.99)


def test_extra_term_adds_projection_on_added_vector():
    """
    The DPSS columns are orthonormal, so one more term leaves the first N
    coefficients unchanged and adds (u_N . y) u_N to the model.
    """
    from scipy.signal import windows

    x = np.linspace(1, 125, 497)
    rng = np.random.default_rng(5)
    y = rng.normal(size=x.size) + 1j * rng.normal(size=x.size)
    n, fhw = 117, 0.4
    small = FitDPSS(x, y, nterms=n, fhw=fhw)
    big = FitDPSS(x, y, nterms=n + 1, fhw=fhw)
    small.fit()
    big.fit()
    u = windows.dpss(x.size, (x[-1] - x[0]) * fhw, Kmax=n + 1)[n]
    assert_allclose(big.popt[:n], small.popt, rtol=0, atol=1e-13)
    assert_allclose(big.yhat - small.yhat, (u @ y) * u, rtol=0, atol=1e-13)


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
