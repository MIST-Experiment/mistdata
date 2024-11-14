from datetime import datetime, timedelta
from typing import List, Union

import numpy as np
import skyfield.api as sf

SF_TS = sf.load.timescale()

def add_sort_time_pair(arr1, time1, arr2, time2):
    """
    Combines two arrays and corresponding time arrays, sorts them based on time,
    and returns the sorted arrays and time arrays.
    """
    time = combine_times(time1, time2)
    arr = np.squeeze(np.vstack((arr1, arr2)))
    idxs = np.argsort(time)
    return arr[idxs], [time[i] for i in idxs]


def add_sort_spec_pair(spec1, time1, spec2, time2):
    """
    This method takes in two sets of spectral data and their corresponding observation times, and combines them into
    a single set of spectral data and observation times. If the observation times of the second set end before the
    observation times of the first set start, the order of the sets will be swapped before combining.
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
    Converts a list of datetime objects to a list of ISO-formatted string representations.
    """
    if not isinstance(dates, List):
        dates = [dates]
    return [dt.isoformat() for dt in dates]


def hdfdt2dtlist(dates):
    """
    Converts an array of dates in string ISO format to a list of datetime objects.
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

