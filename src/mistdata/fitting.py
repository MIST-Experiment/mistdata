import numpy as np
from scipy.signal import windows

def design_matrix(x, model, nterms, fc=None, fhw=None):
    """
    Construct the design matrix for the given model.

    Parameters
    ----------
    x : array-like
        The x data, normally frequency channels. Must be equally spaced if
        ``model'' is 'dpss'.
    model : str
        The model to fit. Must be 'fourier' or 'dpss'.
    nterms : int
        The number of terms in the model. For the Fourier model, this refers
        to the number of cosine terms, meaning the total number of parameters
        will be 2*nterms + 1.
    fc : float
        The center of the DPSS window. Only used if ``model'' is
        'dpss'. Must have inverse units of x.
    fhw : float
        The half-width of the DPSS window. Only used if ``model'' is 'dpss'.
        Must have inverse units of x.

    Returns
    -------
    A : ndarray
        The design matrix for the given model. Rows correspond to data points
        and columns correspond to parameters.

    """
    if model == "fourier":
        A = np.zeros((x.size, 2 * nterms + 1))
        A[:, 0] = 1
        period = np.max(x) - np.min(x)
        for i in range(nterms):
            arg = 2 * np.pi * (i + 1) * x / period
            A[:, 2 * i + 1] = np.cos(arg)
            A[:, 2 * i + 2] = np.sin(arg)
    elif model == "dpss":
        nf = x.size  # window length
        bw = x[-1] - x[0]  # full bandwidth
        xc = x[nf // 2]
        dpss_vec = windows.dpss(nf, bw * fhw, Kmax=nterms).T  # (nf, nterms)
        A = dpss_vec * np.exp(2j * np.pi * (x[:, None] - xc) * fc)
    else:
        raise ValueError("model must be 'fourier' or 'dpss'")
    return A


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

    def __init__(self, x, y, model, nterms, fc=None, fhw=None, sigma=1):
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
        fc : float
            The center of the DPSS window. Only used if ``model'' is 'dpss'.
            Must have inverse units of x.
        fhw : float
            The half-width of the DPSS window. Only used if ``model'' is
            'dpss'. Must have inverse units of x.
        sigma : float
            The uncertainty in the y data. Either scalar (constant noise),
            an array-like of the same length as y, or a matrix specifying the
            full covariance matrix of the data.

        """
        self.x = x
        self.y = y
        self.model = model
        self.nterms = nterms
        self.sigma = sigma
        self.fc = fc
        self.fhw = fhw
        self.A = design_matrix(x, model, nterms, fc=fc, fhw=fhw)

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
        A = design_matrix(x, self.model, self.nterms, fc=self.fc, fhw=self.fhw)
        y = A @ self.popt
        return y
