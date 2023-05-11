import gzip as gz
import numpy as np
from datetime import datetime

from tqdm import tqdm

from ..Thermistors import Thermistors
from ..DUTRecIn import DUTRecIn
from ..DUTLNA import DUTLNA
from ..Spectrum import Spectrum


def _extract_time(array):
    return datetime(*array[1:7].real.astype(np.int64))


def _assign_or_stack(var, array):
    if var is None:
        var = array
    else:
        var = np.vstack((var, array))
    return var


def read_raw(cls, path: str, silent=True):
    with gz.open(path, "r") as fin:
        spec_therm = None
        spec_antenna = None
        spec_ambient = None
        spec_noise_source = None

        for line in fin:
            line = line.decode("utf-8")
            line = line.strip().split()
            line_array = np.array([complex(i.replace("+-", "-")) for i in line])
            # iteration = int(np.real(line_array[0]))
            case = int(np.real(line_array[7]))

            if case == 1:
                time = _extract_time(line_array)
                recin_therm = Thermistors(time, *line_array[[8, 9, 10, 13]].real)
            elif case == 10:
                recin_s11_freq = line_array[8:].real
                recin_s11_freq_time = _extract_time(line_array)
            elif case == 11:
                recin_s11_open = line_array[8:]
                recin_s11_open_time = _extract_time(line_array)
            elif case == 12:
                recin_s11_short = line_array[8:]
                recin_s11_short_time = _extract_time(line_array)
            elif case == 13:
                recin_s11_match = line_array[8:]
                recin_s11_match_time = _extract_time(line_array)
            elif case == 14:
                recin_s11_antenna = line_array[8:]
                recin_s11_antenna_time = _extract_time(line_array)
            elif case == 15:
                recin_s11_ambient = line_array[8:]
                recin_s11_ambient_time = _extract_time(line_array)
            elif case == 16:
                recin_s11_noise_source = line_array[8:]
                recin_s11_noise_source_time = _extract_time(line_array)
            elif case == 2:
                time = _extract_time(line_array)
                lna_therm = Thermistors(time, *line_array[8:12].real)
            elif case == 20:
                lna_s11_freq = line_array[8:].real
                lna_s11_freq_time = _extract_time(line_array)
            elif case == 21:
                lna_s11_open = line_array[8:]
                lna_s11_open_time = _extract_time(line_array)
            elif case == 22:
                lna_s11_short = line_array[8:]
                lna_s11_short_time = _extract_time(line_array)
            elif case == 23:
                lna_s11_match = line_array[8:]
                lna_s11_match_time = _extract_time(line_array)
            elif case == 24:
                lna_s11_lna = line_array[8:]
                lna_s11_lna_time = _extract_time(line_array)
            elif case == 3:
                spec_therm = _assign_or_stack(spec_therm, line_array)
            elif case == 30:
                spec_freq = line_array[8:].real
                # spec_freq_time = _extract_time(line_array)
            elif case == 31:
                spec_antenna = _assign_or_stack(spec_antenna, line_array)
            elif case == 32:
                spec_ambient = _assign_or_stack(spec_ambient, line_array)
            elif case == 33:
                spec_noise_source = _assign_or_stack(spec_noise_source, line_array)

        try:
            spec_therm_time = [_extract_time(arr) for arr in spec_therm]
            spec_therm_lna = spec_therm[:, 8].real
            spec_therm_vna_load = spec_therm[:, 9].real
            spec_therm_ambient_load = spec_therm[:, 10].real
            spec_therm_back_end = spec_therm[:, 13].real
            spec_t_antenna = spec_antenna[:, 8:].real
            spec_t_antenna_time = [_extract_time(arr) for arr in spec_antenna]
            spec_t_ambient = spec_ambient[:, 8:].real
            spec_t_ambient_time = [_extract_time(arr) for arr in spec_ambient]
            spec_t_noise_source = spec_noise_source[:, 8:].real
            spec_t_noise_source_time = [_extract_time(arr) for arr in spec_noise_source]
        except IndexError:
            raise RuntimeError(f"Cannot read_raw file {path}")

        dut_recin = DUTRecIn(
            recin_therm,
            recin_s11_freq,
            recin_s11_freq_time,
            recin_s11_open,
            recin_s11_open_time,
            recin_s11_short,
            recin_s11_short_time,
            recin_s11_match,
            recin_s11_match_time,
            recin_s11_antenna,
            recin_s11_antenna_time,
            recin_s11_ambient,
            recin_s11_ambient_time,
            recin_s11_noise_source,
            recin_s11_noise_source_time,
        )
        dut_lna = DUTLNA(
            lna_therm,
            lna_s11_freq,
            lna_s11_freq_time,
            lna_s11_open,
            lna_s11_open_time,
            lna_s11_short,
            lna_s11_short_time,
            lna_s11_match,
            lna_s11_match_time,
            lna_s11_lna,
            lna_s11_lna_time,
        )
        spec_t = Thermistors(
            spec_therm_time,
            spec_therm_lna,
            spec_therm_vna_load,
            spec_therm_ambient_load,
            spec_therm_back_end,
        )
        spec = Spectrum(
            spec_t,
            spec_freq,
            spec_t_antenna,
            spec_t_antenna_time,
            spec_t_ambient,
            spec_t_ambient_time,
            spec_t_noise_source,
            spec_t_noise_source_time,
        )
    return cls(dut_recin, dut_lna, spec)
