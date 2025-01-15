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


class Fit:

    def __init__(
        self, x, y, model, nterms, sigma=1, normalize="christian", fc=None, fhw=None
    ):
        """
        Fit a model to the given data.

        Parameters
        ----------
        x : array-like
            The x data, normally frequency channels.
        y : array-like
            The y data, normally spectra. Must have the same length as x.
        model : str
            The model to fit. Must be ``fourier'' or ``dpss''.
        nterms : int
            The number of terms in the model. For the Fourier model, this
            refers to the number of cosine terms, meaning the total number of
            parameters will be 2*nterms + 1.
        sigma : array-like
            The uncertainty in the y data. Either scalar (constant noise),
            an array-like of the same length as y, or a matrix specifying the
            full covariance matrix of the data.
        normalize : bool
            Normalize the frequency channels by 2*pi / bandwidth. Only used
            if ``model'' is 'fourier'.
        fc : float
            The center of the DPSS window. Only used if ``model'' is 'dpss'.
            Must have inverse units of x.
        fhw : float
            The half-width of the DPSS window. Only used if ``model'' is
            'dpss'. Must have inverse units of x.

        """
        self.x = x
        self.y = y
        self.model = model
        self.nterms = nterms
        self.sigma = sigma
        self.normalize = normalize
        self.fc = fc
        self.fhw = fhw
        self.A = self.design_matrix()

    def design_matrix(self, x=None):
        """
        Construct the design matrix. Let x be None for fitting, and provide a
        new x for prediction.
        """
        if x is None:
            x = self.x
        else:
            x = np.copy(x)
        if self.model == "dpss":
            return self._design_matrix_dpss(x)
        if self.normalize == "christian":
            x = 2 * np.pi * x / (x[-1] - x[0])
        elif self.normalize == "raul":
            x = (x - x.max()) / x[x.size // 2]
        return self._design_matrix_fourier(x=x)

    def _design_matrix_fourier(self, x):
        A = np.empty((x.size, 2 * self.nterms + 1))
        A[:, 0] = 1
        for i in range(self.nterms):
            arg = (i + 1) * x
            A[:, 2 * i + 1] = np.cos(arg)
            A[:, 2 * i + 2] = np.sin(arg)
        return A

    def _design_matrix_dpss(self, x):
        nf = x.size
        bw = x[-1] - x[0]
        xc = x[nf // 2]
        arg = 2j * np.pi * (x[:, None] - xc) * self.fc
        dpss_vec = windows.dpss(nf, bw * self.fhw, Kmax=self.nterms).T
        A = dpss_vec * np.exp(arg)
        return A

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
