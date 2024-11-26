"""
Tools for MIST S11 measurements.
"""

import numpy as np
from .cal_s11 import embed_sparams, de_embed_sparams, network_sparams


class S11:

    def __init__(self, data):
        """
        Base class for S11 measurements. End users should use either
        AntennaS11, ReceiverS11 or CalStandardS11.

        Parameters
        ----------
        data : mistdata.MISTData.DUTrecIn or mistdata.MISTData.DUTLNA

        """
        self.freq = data.s11_freq  # in MHz
        self.data = data
        self.raw_s11_standards = {
            "open": self.data.s11_open,
            "short": self.data.s11_short,
            "match": self.data.s11_match,
        }

    @property
    def cal_s11_internal(self):
        """
        Return S11 data calibrated at the internal reference plane of the
        receiver. This is the first step in the calibration process.

        Returns
        -------
        calibrated_s11 : dict or ndarray
            The calibrated S11 data of same type as raw_s11. If raw_s11 is a
            dict, then calibrated_s11 is a dict with the same keys. If raw_s11
            is an ndarray, then calibrated_s11 is an ndarray.

        """
        # measured standards
        measured_standards = np.array(
            [
                self.raw_s11_standards["open"],
                self.raw_s11_standards["short"],
                self.raw_s11_standards["match"],
            ],
            dtype=complex,
        )
        # model for calibration network, open = 1, short = -1, match = 0
        model_s11 = np.empty_like(measured_standards)
        model_s11[0] = 1  # open
        model_s11[1] = -1  # short
        model_s11[2] = 0  # match
        # vna S-parameters
        vna_sparams = network_sparams(model_s11, measured_standards)

        raw_s11 = self.raw_s11
        if isinstance(raw_s11, dict):
            calibrated_s11 = {}
            for key, gamma in raw_s11.items():
                calibrated_s11[key] = de_embed_sparams(vna_sparams, gamma)
        else:
            calibrated_s11 = de_embed_sparams(vna_sparams, raw_s11)

        # use the same calibration on the standards for validation
        calibrated_s11_standards = {}
        for key, gamma in self.raw_s11_standards.items():
            calibrated_s11_standards[key] = de_embed_sparams(
                vna_sparams, gamma
            )
        self.cal_s11_standards = calibrated_s11_standards

        return calibrated_s11


class AntennaS11(S11):

    def __init__(self, data, pathA_sparams):
        """
        Class for antenna S11 measurements.

        Parameters
        ----------
        data : mistdata.MISTData.DUTrecIn
        pathA_sparams : array-like
            S-parameters of the path A in the receiver. See Fig 19 in the MIST
            instrument paper, Monsalve et al. 2024.

        """
        super().__init__(data)
        self.pathA_sparams = pathA_sparams

    @property
    def raw_s11(self):
        """
        Return the raw S11 data.

        Returns
        -------
        s11 : dict
            Dictionary containing the raw S11 data. Keys are 'antenna',
            'ambient', and 'noise_source'.

        """
        s11 = {
            "antenna": self.data.s11_antenna,
            "ambient": self.data.s11_ambient,
            "noise_source": self.data.s11_noise_source,
        }
        return s11

    @property
    def s11(self):
        """
        Return S11 data calibrated with the calibration kit and with the
        reference plane at the receiver input. The second step involves
        de-embedding the S-parameters of internal paths in the receiver (see
        Fig 19 in the MIST instrument paper, Monsalve et al. 2024).

        Returns
        -------
        calibrated_s11 : dict
            Dictionary containing the calibrated S11 data. Keys are 'antenna',
            'ambient', and 'noise_source'.

        """
        calibrated_s11 = {}
        # de-embed S-parameters of path A in the receiver
        for key, gamma in self.cal_s11_internal.items():
            calibrated_s11[key] = de_embed_sparams(self.pathA_sparams, gamma)

        return calibrated_s11


class ReceiverS11(S11):

    def __init__(self, data, pathB_sparams, pathC_sparams):
        """
        Class for receiver S11 measurements.

        Parameters
        ----------
        data : mistdata.MISTData.DUTrecIn
        pathB_sparams : array-like
            S-parameters of the path B in the receiver. See Fig 19 in the MIST
            instrument paper, Monsalve et al. 2024.
        pathC_sparams : array-like
            S-parameters of the path C in the receiver.

        """
        super().__init__(data)
        self.pathB_sparams = pathB_sparams
        self.pathC_sparams = pathC_sparams

    @property
    def raw_s11(self):
        """
        Return the raw S11 data.

        Returns
        -------
        s11 : ndarray
            The raw S11 data.

        """
        return self.data.s11_lna

    @property
    def s11(self):
        """
        Return S11 data calibrated with the calibration kit and with the
        reference plane at the receiver input. The second step involves
        de-embedding the S-parameters of internal paths in the receiver (see
        Fig 19 in the MIST instrument paper, Monsalve et al. 2024).

        Returns
        -------
        calibrated_s11 : ndarray
            The calibrated S11 data.

        """
        calibrated_s11 = {}
        # de-embed S-parameters of path B in the receiver and embed the
        # S-parameters of path C in the receiver
        gamma = self.cal_s11_internal["lna"]
        calibrated_s11 = de_embed_sparams(self.pathB_sparams, gamma)
        calibrated_s11 = embed_sparams(self.pathC_sparams, calibrated_s11)

        return calibrated_s11


class CalStandardS11(S11):

    @property
    def raw_s11(self):
        """
        Return the raw S11 data.

        Returns
        -------
        s11 : ndarray
            The raw S11 data.

        """
        return self.data.s11_antenna

    @property
    def s11(self):
        """
        Return S11 data calibrated at the internal reference plane. This is
        used to get the S-parameters of the internal path between the
        receiver input and the calibration network.

        Returns
        -------
        calibrated_s11 : ndarray
            The calibrated S11 data.

        """
        return self.cal_s11_internal
