from datetime import datetime
from typing import List, Union
import numpy as np
from scipy.optimize import curve_fit
import skyfield.api as sf

SF_TS = sf.load.timescale()


def add_sort_time_pair(arr1, time1, arr2, time2):
    """
    Combine two arrays and corresponding time arrays, sort them based on time,
    and returns the sorted arrays and time arrays.
    """
    time = combine_times(time1, time2)
    arr = np.squeeze(np.vstack((arr1, arr2)))
    idxs = np.argsort(time)
    return arr[idxs], [time[i] for i in idxs]


def add_sort_spec_pair(spec1, time1, spec2, time2):
    """
    This method takes in two sets of spectral data and their corresponding
    observation times, and combines them into a single set of spectral data and
    observation times. If the observation times of the second set end before
    the observation times of the first set start, the order of the sets will be
    swapped before combining.
    """
    if time2[-1] < time1[0]:
        time1, time2 = time2, time1
        spec1, spec2 = spec2, spec1
    time = combine_times(time1, time2)
    spec = np.vstack((spec1, spec2))
    # idxs = np.argsort(time)
    # return spec[idxs], [time[i] for i in idxs]
    return spec, time


def dtlist2strlist(dates: Union[datetime, List[datetime]]):
    """
    Convert a list of datetime objects to a list of ISO-formatted string
    representations.
    """
    if not isinstance(dates, List):
        dates = [dates]
    return [dt.isoformat() for dt in dates]


def hdfdt2dtlist(dates):
    """
    Convert an array of dates in string ISO format to a list of datetime
    objects.
    """
    strdates = [datetime.fromisoformat(dt) for dt in dates.asstr()[()]]
    if len(strdates) == 1:
        return strdates[0]
    return strdates


def ds2np(dataset):
    """
    Converts dataset read from file to float, array or None
    """
    return np.array(dataset)
    # print(np.array(dataset))
    # print(isinstance(dataset, float))
    # if dataset.ndim == 0:
    #     print(dataset.astype(float))
    #     return None
    # elif isinstance(dataset, float):
    #     return dataset
    # else:
    #     return dataset[:]


def combine_times(time1, time2):
    """
    Combine two time lists into a single list.
    """
    res = []
    res.extend(time1) if isinstance(time1, List) else res.append(time1)
    res.extend(time2) if isinstance(time2, List) else res.append(time2)
    return res


def dt2lst(dt: datetime, lon: float):
    """
    Calculate Local Sidereal Time (LST) given a datetime object and longitude.
    """
    t = SF_TS.from_datetime(dt)
    earth_loc = sf.wgs84.latlon(90, lon)
    return earth_loc.lst_hours_at(t)


def x_transform_fourier(x):
    """
    Transform x values to the range [0, 1] for Fourier series fitting.

    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def fourier_series(x, *params):
    """
    Evaluate a Fourier series at the given x values.

    Parameters
    ----------
    x : array-like
        The x values to evaluate the Fourier series at.
    params : array-like
        The parameters of the Fourier series. The first element is the
        constant term, the next N elements are the cosine coefficients, and
        the last N elements are the sine coefficients.

    Returns
    -------
    y : ndarray
        The values of the Fourier series at the given x values.

    """
    params = np.array(params)
    y = params[0]
    pcos, psin = np.split(params[1:], 2)
    N = pcos.size
    x = x_transform_fourier(x)
    for i in range(N):
        arg = (i + 1) * x
        y += pcos[i] * np.cos(arg) + psin[i] * np.sin(arg)
    return y


def fit_fourier(
    xdata, ydata, deg, xmin=None, xmax=None, complex_data=True, **kwargs
):
    """
    Fit a Fourier series to the given data.

    Parameters
    ----------
    xdata : array-like
        The x data.
    ydata : array-like
        The y data. Must have the same length as xdata.
    deg : int
        The degree of the Fourier series to fit.
    xmin : float
        The minimum x value to use in the fit.
    xmax : float
        The maximum x value to use in the fit.
    complex_data : bool
        Whether the data is complex. If True, the Fourier series will be fit
        to the real and imaginary parts of the data separately.
    kwargs : dict
        Additional keyword arguments to pass to scipy.optimize.curve_fit.

    Returns
    -------
    popt : ndarray
        The optimized parameters of the Fourier series, with length 2*deg + 1
        in order of constant term, the cosine coefficients, and the sine
        coefficients.

    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    p0 = kwargs.pop("p0", np.zeros(2 * deg + 1))
    if np.size(p0) != 2 * deg + 1:
        raise ValueError("p0 must have length 2*deg + 1")
    if xmin:
        ydata = ydata[xdata >= xmin]
        xdata = xdata[xdata >= xmin]
    if xmax:
        ydata = ydata[xdata <= xmax]
        xdata = xdata[xdata <= xmax]
    if complex_data:
        pr = curve_fit(
            fourier_series, xdata, ydata.real, p0=p0.real, **kwargs
        )[0]
        pi = curve_fit(
            fourier_series, xdata, ydata.imag, p0=p0.imag, **kwargs
        )[0]
        popt = pr + 1j * pi
    else:
        popt = curve_fit(fourier_series, xdata, ydata, p0=p0, **kwargs)[0]
    return popt
