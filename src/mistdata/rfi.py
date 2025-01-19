import datetime
import numpy as np
import scipy.ndimage
from astropy.convolution import convolve
from .fitting import FitDPSS


def flag_times(spec, tstart=None, tstop=None, inplace=True):
    """
    Flag all frequency channels within a given time range.

    Parameters
    ----------
    spec : MISTData.Spectrum
    tstart : datetime.datetime or float
        The start time of the range to flag. If a float, it is interpreted
        as the unix timestamp and converted to a datetime object with
        datetime.datetime.fromtimestamp. If None, the start time is set to
        the beginning of the observation.
    tstop : datetime.datetime or float
        The end time of the range to flag. See `tstart` for details. If None,
        the end time is set to the end of the observation.
    inplace : bool
        If True, the data in `spec' is modified in place. If False,
        the flags are returned as a new array.

    Returns
    -------
    flags : np.ndarray
        A boolean array with the same shape as `spec.t_antenna' that is True
        for data points that are within the specified time range. Only
        returned if `inplace` is False.

    """
    times = np.array(spec.psd_antenna_time)
    if tstart is None:
        tstart = spec.psd_antenna_time[0]
    elif isinstance(tstart, float):
        tstart = datetime.datetime.fromtimestamp(tstart)
    if tstop is None:
        tstop = spec.psd_antenna_time[-1]
    elif isinstance(tstop, float):
        tstop = datetime.datetime.fromtimestamp(tstop)
    flags = (times >= tstart) & (times <= tstop)
    # add freq axis for broadcasting
    flags = np.broadcast_to(flags[:, None], spec.t_antenna.shape)
    if inplace:
        spec.flag_rfi(flags)
    else:
        return flags


def flag_mad(spec, limit=5, inplace=True):
    """
    Flag outliers using the median absolute deviation (MAD) method.

    Parameters
    ----------
    spec : MISTData.Spectrum
    limit : float
        The threshold for flagging outliers.
    inplace : bool
        If True, the data in `spec' is modified in place. If False,
        the flags are returned as a new array.

    Returns
    -------
    flags : np.ndarray
        A boolean array with the same shape as `spec.t_antenna' that is True
        for data points that are outliers. Only returned if `inplace` is False.

    Notes
    -----
    This method flags data points in the `t_antenna' attribute of the
    MISTData.Spectrum object based on their deviation from the median.

    The method calculates the median of each colum in `t_antenna' and
    then computes the absolute difference between each data point and
    its column median.. It then calculates the median of these absolute
    differences across all columns, which gives the MAD value.

    Data points that have an absolute difference from their column median
    greater than `limit' times the MAD value are flagged as outliers.

    """
    data = spec.t_antenna
    mdata = np.median(data, axis=0)
    absdiff = np.abs(data - mdata)
    mad = np.median(absdiff, axis=0)
    flags = absdiff > limit * mad
    if inplace:
        spec.flag_rfi(flags)
    else:
        return flags


def median_filter(spec, size=None, footprint=None, limit=100, inplace=True):
    """
    Apply a median filter to the data to find and flag RFI.

    Parameters
    ----------
    spec : MISTData.Spectrum
    size : int or tuple of ints
        The size of the filter. If a single integer, the filter is a square
        with side length `size'. If a tuple, it is the shape of the filter.
    footprint : array-like
        Shape of the filter. If not None, `size' is ignored.
    limit : float
        The threshold for flagging RFI in units of the noise level.
    inplace : bool
        If True, the data in `spec' is modified in place. If False,
        the flags are returned as a new array.

    Returns
    -------
    flags : np.ndarray
        A boolean array with the same shape as `spec.t_antenna' that is True
        for data points that are flagged as RFI. Only returned if `inplace` is
        False.

    Notes
    -----
    See the documentation for `scipy.ndimage.median_filter' for details on the
    `size' and `footprint' parameters.

    """
    data = spec.t_antenna
    noise_est = spec.noise_estimate()
    med = scipy.ndimage.median_filter(data, size=size, footprint=footprint)
    flags = np.abs(data - med) > limit * noise_est
    if inplace:
        spec.flag_rfi(flags)
    else:
        return flags


def mean_filter(
    spec, kernel, limit=100, time_thresh=0.4, freq_thresh=0.3, inplace=True
):
    """
    Apply a mean filter to the data to find and flag RFI.

    Parameters
    ----------
    spec : MISTData.Spectrum
    kernel : array-like
        The kernel to use for the filter.
    limit : float
       The threshold for flagging RFI in units of the noise level.
    time_thresh : float
        Fraction of time samples that must be flagged to flag the entire
        frequency channel.
    freq_thresh : float
        Fraction of frequency channels that must be flagged to flag the entire
        time sample.
    inplace : bool
        If True, the data in `spec' is modified in place. If False,
        the flags are returned as a new array.

    Returns
    -------
    flags : np.ndarray
        A boolean array with the same shape as `spec.t_antenna' that is True
        for data points that are flagged as RFI. Only returned if `inplace` is
        False.

    """
    data = spec.t_antenna
    noise_est = spec.noise_estimate()
    mean = convolve(data, kernel)
    flags = np.abs(data - mean) > limit * noise_est
    flags[:, flags.mean(axis=0) > time_thresh] = True
    flags[flags.mean(axis=1) > freq_thresh, :] = True
    if inplace:
        spec.flag_rfi(flags)
    else:
        return flags


def dpss_filter(spec, fhw, fc=0, eval_cutoff=1e-9, limit=6, inplace=True):
    """
    Fit DPSS modes to the data and flag outliers.

    Parameters
    ----------
    spec : MISTData.Spectrum
    fhw : float
        The half-width of the DPSS window. In inverse frequency units
        (usually microseconds for MIST data, since frequency is in MHz).
    fc : float
        The center frequency of the DPSS window. In inverse frequency units.
    eval_cutoff : float
        The cutoff for the eigenvalues of the DPSS modes. Modes with
        eigenvalues less than this value are discarded.
    limit : float
        The threshold for flagging outliers.
    inplace : bool
        If True, the data in `spec' is modified in place. If False,
        the flags are returned as a new array.

    Returns
    -------
    flags : np.ndarray
        A boolean array with the same shape as `spec.t_antenna' that is True
        for data points that are flagged as outliers. Only returned if
        `inplace` is False.

    """
    data = spec.t_antenna
    mdl = np.empty_like(data)
    res = np.empty_like(data)
    fit = FitDPSS(spec.freq, data[0], eval_cutoff=eval_cutoff, fc=fc, fhw=fhw) 
    for i, d in enumerate(data):
        print(f"Fitting DPSS to {i + 1}/{data.shape[0]}")
        fit.y = d
        fit.fit()
        mdl[i] = fit.yhat
        res[i] = fit.residuals
    # use smoothed data for noise estimate without RFI
    noise = spec.noise_estimate()
    noise_est = noise * mdl / data
    flags = np.abs(res) > limit * noise_est
    if inplace:
        spec.flag_rfi(flags)
    else:
        return flags
