import gzip

import h5py
import numpy as np
import pytest

from mistdata import FREQ_AXIS_CONVENTION, MISTData


CHANNEL_WIDTH_MHZ = 125 / 4096
DAQ_FREQ = np.arange(4096) * CHANNEL_WIDTH_MHZ
VNA_FREQ = [25, 35]


def write_raw(path, spec_freq):
    with gzip.open(path, "wt") as raw:
        def row(case, values, second=0):
            header = [0, 2024, 10, 19, 0, 0, second, case]
            raw.write(" ".join(map(str, [*header, *values])) + "\n")

        row(1, [300] * 6)
        row(2, [300] * 4)
        for case in (10, 20):
            row(case, VNA_FREQ)
        for case in (*range(11, 17), *range(21, 25)):
            row(case, [0.1, 0.2])
        row(30, spec_freq)
        for second in (0, 1):
            row(3, [300] * 6, second)
            for case in (31, 32, 33):
                row(case, np.arange(4096) + case + second, second)


@pytest.fixture
def raw_path(tmp_path):
    path = tmp_path / "spectrum.gz"
    write_raw(path, DAQ_FREQ)
    return str(path)


def test_read_raw_frequency_axis_default_is_daq(raw_path):
    data = MISTData.read_raw(raw_path)
    assert FREQ_AXIS_CONVENTION == data.spec.freq_axis == "daq"
    np.testing.assert_array_equal(data.spec.freq, DAQ_FREQ)


@pytest.mark.parametrize("relabel_freq", [False, True])
def test_read_raw_frequency_axis_for_multiple_files(raw_path, relabel_freq):
    datafiles = MISTData.read_raw([raw_path, raw_path], relabel_freq=relabel_freq)
    assert len(datafiles) == 2
    for data in datafiles:
        assert data.spec.freq_axis == ("true" if relabel_freq else "daq")
        expected = DAQ_FREQ + relabel_freq * CHANNEL_WIDTH_MHZ
        np.testing.assert_array_equal(data.spec.freq, expected)
        np.testing.assert_array_equal(data.dut_recin.s11_freq, VNA_FREQ)
        np.testing.assert_array_equal(data.dut_lna.s11_freq, VNA_FREQ)
        for case, psd in (
            (31, data.spec.psd_antenna),
            (32, data.spec.psd_ambient),
            (33, data.spec.psd_noise_source),
        ):
            np.testing.assert_array_equal(psd[0], np.arange(4096) + case)
            np.testing.assert_array_equal(psd[1], psd[0] + 1)


@pytest.mark.parametrize("relabel_freq", [False, True])
def test_frequency_axis_survives_slicing_addition_and_cut(raw_path, relabel_freq):
    data = MISTData.read_raw(raw_path, relabel_freq=relabel_freq)
    axis = "true" if relabel_freq else "daq"
    expected = DAQ_FREQ + relabel_freq * CHANNEL_WIDTH_MHZ
    subset = data[:1]
    assert subset.spec.freq_axis == axis
    assert subset.spec.nspec == 1
    combined = data + data
    assert combined.spec.freq_axis == axis
    assert combined.spec.nspec == 4
    np.testing.assert_array_equal(combined.spec.freq, expected)

    cut = combined.cut_freq(30, 40, inplace=False)
    expected_cut = expected[(expected >= 30) & (expected <= 40)]
    assert cut.spec.freq_axis == axis
    assert cut[:1].spec.freq_axis == axis
    np.testing.assert_array_equal(cut.spec.freq, expected_cut)
    np.testing.assert_array_equal(combined.spec.freq, expected)


def test_add_refuses_mixed_frequency_axes_even_on_matching_grids(raw_path):
    daq = MISTData.read_raw(raw_path).spec
    true = MISTData.read_raw(raw_path, relabel_freq=True).spec
    for spec in (daq, true):
        spec.cut_freq(30, 40)
    np.testing.assert_array_equal(daq.freq, true.freq)
    with pytest.raises(ValueError, match="frequency axis conventions"):
        daq + true


@pytest.mark.parametrize("relabel_freq", [False, True])
def test_frequency_axis_hdf5_round_trip(raw_path, tmp_path, relabel_freq):
    data = MISTData.read_raw(raw_path, relabel_freq=relabel_freq)
    data.cut_freq(30, 40)
    path = str(tmp_path / "spectrum.mist")
    for _ in range(2):
        data.save(path)
        data = MISTData.load(path)
        assert data.spec.freq_axis == ("true" if relabel_freq else "daq")
        expected = DAQ_FREQ + relabel_freq * CHANNEL_WIDTH_MHZ
        np.testing.assert_array_equal(
            data.spec.freq, expected[(expected >= 30) & (expected <= 40)]
        )


def test_frequency_axis_reads_legacy_hdf5_without_marker(raw_path, tmp_path):
    data = MISTData.read_raw(raw_path)
    path = str(tmp_path / "legacy.mist")
    data.save(path)
    with h5py.File(path, "r+") as file:
        del file["spec"].attrs["freq_axis"]

    legacy = MISTData.load(path)
    assert legacy.spec.freq_axis is None
    np.testing.assert_array_equal(legacy.spec.freq, DAQ_FREQ)
    assert (legacy + data).spec.freq_axis == "daq"


def test_read_raw_refuses_a_second_shift(raw_path):
    write_raw(raw_path, DAQ_FREQ + CHANNEL_WIDTH_MHZ)
    with pytest.raises(ValueError, match="125 MHz"):
        MISTData.read_raw(raw_path, relabel_freq=True)
