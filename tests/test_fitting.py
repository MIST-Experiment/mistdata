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
