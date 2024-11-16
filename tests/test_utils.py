from datetime import datetime
import numpy as np

from mistdata import utils

def test_add_sort_time_pair():
    t1 = datetime(2024, 10, 20, 23, 47, 37)
    t2 = datetime(2024, 10, 20, 8, 48, 20)
    freq = np.linspace(1, 125, num=497)
    sum_freq, sum_time = utils.add_sort_time_pair(freq, t1, freq, t2)
    assert sum_time == [t2, t1]  # should be sorted
    # sum_freq should be two copies of the same array
    assert np.all(sum_freq == [freq, freq])

    # check that freq is sorted if they differ
    freq2 = 2 * freq
    sum_freq, sum_time = utils.add_sort_time_pair(freq, t1, freq2, t2)
    assert sum_time == [t2, t1]  # should be sorted
    # sum_freq should be sorted based on time
    assert np.all(sum_freq == [freq2, freq])
