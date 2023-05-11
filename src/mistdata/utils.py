from datetime import datetime
from typing import List, Union

import numpy as np


def add_sort_time_pair(arr1, time1, arr2, time2):
    time = combine_times(time1, time2)
    arr = np.hstack((arr1, arr2))
    idxs = np.argsort(time)
    return arr[idxs], [time[i] for i in idxs]


def add_sort_spec_pair(spec1, time1, spec2, time2):
    time = combine_times(time1, time2)
    spec = np.vstack((spec1, spec2))
    idxs = np.argsort(time)
    return spec[idxs], [time[i] for i in idxs]


def dtlist2strlist(dates: Union[datetime, List[datetime]]):
    if not isinstance(dates, List):
        dates = [dates]
    return [dt.isoformat() for dt in dates]


def hdfdt2dtlist(dates):
    """
    Converts written array of dates in string iso format to datetime objects.
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
    res = []
    res.extend(time1) if isinstance(time1, List) else res.append(time1)
    res.extend(time2) if isinstance(time2, List) else res.append(time2)
    return res




