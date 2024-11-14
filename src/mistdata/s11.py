"""
Tools for MIST S11 measurements.
"""

import numpy as np
from .cal_s11 import embed_sparams, de_embed_sparams, network_sparams


class S11:

    def __init__(self, data, model_gamma):
        """
        Base class for S11 measurements. End users should use either
        AntennaS11 or ReceiverS11.

        Parameters
        ----------
        data : mistdata.MISTData.DUTrecIn or mistdata.MISTData.DUTLNA
        model_gamma : dict
            Dictionary containing the model of reflection coefficients for the
            calibration network. Keys are 'open', 'short', and 'match'.

        """
        self.freq = data.s11_freq  # in MHz
        self.data = data
        for k, v in model_gamma.items():
            model_gamma[k] = np.atleast_1d(v)  # add frequency axis if scalar
        self.model_gamma = np.array(
            [model_gamma["open"], model_gamma["short"], model_gamma["match"]],
            dtype=complex,
        )

    @property
    def vna_sparams(self):
        """
        Return the S-parameters of the VNA.
        """
        gamma_meas = np.array(
            [self.data.s11_open, self.data.s11_short, self.data.s11_match]
        )
        return cal_s11.network_sparams(self.model_gamma, gamma_meas)

    @property
    def cal_s11_internal(self):
        """
        Return S11 data calibrated at the internal reference plane of the
        receiver. This is the first step in the calibration process.

        Returns
        -------
        calibrated_s11 : dict
            Dictionary containing the calibrated S11 data. Keys are the same
            as for raw_S11.

        """
        calibrated_s11 = {}
        for key, gamma in self.raw_s11.items():
            calibrated_s11[key] = cal_s11.de_embed_sparams(
                self.vna_sparams, gamma
            )

        return calibrated_s11


class AntennaS11(S11):

    def __init__(self, data, pathA_sparams, model_gamma):
        """
        Class for antenna S11 measurements.

        Parameters
        ----------
        data : mistdata.MISTData.DUTrecIn
        pathA_sparams : array-like
            S-parameters of the path A in the receiver. See Fig 19 in the MIST
            instrument paper, Monsalve et al. 2024.
        model_gamma : dict
            Dictionary containing the model of reflection coefficients for the
            calibration network. Keys are 'open', 'short', and 'match'.

        """
        super().__init__(data, model_gamma)
        self.pathA_sparams = pathA_sparams

    @property
    def raw_S11(self):
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
            calibrated_s11[key] = cal_s11.de_embed_sparams(
                self.pathA_sparams, gamma
            )

        return calibrated_s11


class ReceiverS11(S11):

    def __init__(self, data, pathB_sparams, pathC_sparams, model_gamma):
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
        model_gamma : dict
            Dictionary containing the model of reflection coefficients for the
            calibration network. Keys are 'open', 'short', and 'match'.

        """
        super().__init__(data, model_gamma)
        self.pathB_sparams = pathB_sparams
        self.pathC_sparams = pathC_sparams

    @property
    def raw_S11(self):
        """
        Return the raw S11 data.

        Returns
        -------
        s11 : dict
            Dictionary containing the raw S11 data. Only key is 'lna'.

        """
        s11 = {"lna": self.data.s11_lna}
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
            Dictionary containing the calibrated S11 data. Key is 'lna'.

        """
        calibrated_s11 = {}
        # de-embed S-parameters of path B in the receiver and embed the
        # S-parameters of path C in the receiver
        for key, gamma in self.cal_s11_internal.items():
            cal_gamma = cal_s11.de_embed_sparams(self.pathB_sparams, gamma)
            cal_gamma = cal_s11.embed_sparams(self.pathC_sparams, cal_gamma)
            calibrated_s11[key] = cal_gamma

        return calibrated_s11
