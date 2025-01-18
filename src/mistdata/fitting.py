import numpy as np
from scipy.signal import windows


def least_squares(A, y, sigma):
    """
    Solve the linear least squares problem.

    Parameters
    ----------
    A : array-like
        The design matrix.
    y : array-like
        The y data.
    sigma : float or array-like
        The uncertainty in the y data. Either scalar (constant noise), an
        array-like of the same length as y, or a matrix specifying the full
        covariance matrix of the data.

    Returns
    -------
    xhat : ndarray
        The least squares solution.

    """
    if np.isscalar(sigma):
        W = np.eye(y.size) / sigma**2
    elif np.size(sigma) == y.size:
        W = np.diag(1 / sigma**2)
    else:
        W = np.linalg.inv(sigma)

    Q, R = np.linalg.qr(np.sqrt(W) @ A, mode="reduced")
    xhat = np.linalg.solve(R, Q.T @ np.sqrt(W) @ y)
    return xhat


def get_nterms_DPSS(x, fhw, eval_cutoff):
    """
    Find the number of DPSS vectors with eigenvalues above the given
    cutoff. This uses the estimate provided by Slepian 1978 and
    Karnik 2020, as implemented by Aaron Ewall-Wice in the hera_filters
    package.

    Parameters
    ----------
    x : array-like
        The x data, normally frequency channels.
    fhw : float
        The half-width of the DPSS window. Must have inverse units of x.
    eval_cutoff : float
        Minimum eigenvalue to include in the model.

    Returns
    -------
    nterms : int
        The number of DPSS vectors to include.

    """
    nf = np.size(x)
    bw = x[-1] - x[0]
    # arguments for the log terms
    a = 4 * nf
    b = 4 / (eval_cutoff * (1 - eval_cutoff))
    Nw = 2 * bw * fhw + 2 / np.pi**2 * np.log(a) * np.log(b)
    Kmax = int(np.min([Nw, nf]))  # this is an upper bound for nterms
    # we compute the eigenvalues and keep the ones above the cutoff
    evals = windows.dpss(nf, bw * fhw, Kmax=Kmax, return_ratios=True)[1]
    nterms = np.max(np.where(evals >= eval_cutoff))
    return nterms


class Fit:

    def __init__(self, x, y, nterms, sigma=1):
        """
        Base class for fitting a model to the given data. Use the subclasses
        FitDPSS and FitFourier instead.
        """
        self.x = x
        self.y = y
        self.nterms = nterms
        self.sigma = sigma
        self.A = self.design_matrix()

    def design_matrix(self, x=None):
        """
        Construct the design matrix. Let x be None for fitting, and provide a
        new x for prediction. This method must be provided by the subclass.

        Parameters
        ----------
        x : array-like
            The x values to construct the design matrix for. If None, use the
            x values provided during initialization.

        Returns
        -------
        A : ndarray
            The design matrix. Shape is (x.size, nterms).

        """
        raise NotImplementedError("Subclass must implement design_matrix.")

    def fit(self):
        """
        Fit the model to the data.

        """
        self.popt = least_squares(self.A, self.y, self.sigma)
        self.yhat = self.A @ self.popt
        self.residuals = self.y - self.yhat

    def predict(self, x):
        """
        Predict the model at the given x values.

        Parameters
        ----------
        x : array-like
            The x values to predict the model at.

        Returns
        -------
        y : ndarray
            The predicted y values.

        """
        A = self.design_matrix(x)
        y = A @ self.popt
        return y


class FitDPSS(Fit):

    def __init__(
        self, x, y, nterms=None, eval_cutoff=None, sigma=1, fc=0, fhw=0.2
    ):
        """
        Fit a model to the given data using Discrete Prolate Spheroidal
        Sequences (DPSS).

        Parameters
        ----------
        x : array-like
            The x data, normally frequency channels.
        y : array-like
            The y data, normally spectra. Must have the same length as x.
        nterms : int
            The number of DPSS vectors to include in the model. This can
            be determined automatically by providing eval_cutoff instead.
        eval_cutoff : float
            Minimum eigenvalue to include in the model. If provided, nterms
            will be determined automatically.
        sigma : array-like
            The uncertainty in the y data. Either scalar (constant noise),
            an array-like of the same length as y, or a matrix specifying the
            full covariance matrix of the data.
        fc : float
            The center of the DPSS window. Must have inverse units of x.
        fhw : float
            The half-width of the DPSS window.Must have inverse units of x.

        Notes
        -----
        A useful way to determine ``fc'' and ``fhw'' is to fourier transform
        ``y'' and plot the magnitude of this delay spectrum. The center and
        half-width of the DPSS window should be chosen to capture the peak
        of the delay spectrum.

        """
        if eval_cutoff is not None:
            nterms = get_nterms_DPSS(x, fhw, eval_cutoff)
        if nterms is None:
            raise ValueError("Must provide nterms or eval_cutoff.")
        self.fc = fc
        self.fhw = fhw
        super().__init__(x, y, nterms, sigma=sigma)

    def design_matrix(self, x=None):
        """
        Construct the design matrix. Let x be None for fitting, and provide a
        new x for prediction.

        Parameters
        ----------
        x : array-like
            The x values to construct the design matrix for. If None, use the
            x values provided during initialization.

        Returns
        -------
        A : ndarray
            The design matrix. Shape is (x.size, nterms).

        """
        if x is None:
            x = self.x
        nf = x.size
        bw = x[-1] - x[0]
        xc = x[nf // 2]
        arg = 2j * np.pi * (x[:, None] - xc) * self.fc
        # each row is a dpss vector
        dpss_vec = windows.dpss(nf, bw * self.fhw, Kmax=self.nterms).T
        # the dpss vectors are normalized (2-norm) meaning the A matrix will
        # depend on a factor of sqrt(nf) that we need to account for
        dpss_vec *= np.sqrt(nf)
        A = dpss_vec * np.exp(arg)
        return A


class FitFourier(Fit):

    def __init__(self, x, y, nterms, sigma=1, normalize=True):
        """
        Fit a model to the given data using Fourier series.

        Parameters
        ----------
        x : array-like
            The x data, normally frequency channels.
        y : array-like
            The y data, normally spectra. Must have the same length as x.
        nterms : int
            The number of cosine terms to include in the model. The total
            number of parameters will be 2*nterms+1 (constant term and sine
            and cosine terms).
        sigma : array-like
            The uncertainty in the y data. Either scalar (constant noise),
            an array-like of the same length as y, or a matrix specifying the
            full covariance matrix of the data.
        normalize : bool
            Normalize the frequency channels by 2*pi / bandwidth.

        """
        super().__init__(x, y, nterms, sigma=sigma)
        self.normalize = normalize

    def design_matrix(self, x=None):
        """
        Construct the design matrix. Let x be None for fitting, and provide a
        new x for prediction.

        Parameters
        ----------
        x : array-like
            The x values to construct the design matrix for. If None, use the
            x values provided during initialization.

        Returns
        -------
        A : ndarray
            The design matrix. Shape is (x.size, 2*nterms+1).

        """

        if self.normalize:
            x = 2 * np.pi * x / (x[-1] - x[0])
        A = np.empty((x.size, 2 * self.nterms + 1))
        A[:, 0] = 1
        for i in range(self.nterms):
            arg = (i + 1) * x
            A[:, 2 * i + 1] = np.cos(arg)
            A[:, 2 * i + 2] = np.sin(arg)
        return A
