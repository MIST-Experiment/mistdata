"""
High level tools for MIST calibration.
"""

from . import cal_S11

class S11:

    def __init__(
        self, data, cal_kit=cal_s11.Keysight85033E, match_resistance=50
    ):
        """
        Base class for S11 measurements. End users should use either
        AntennaS11 or ReceiverS11.

        Parameters
        ----------
        data : mistdata.MISTData.DUTrecIn or mistdata.MISTData.DUTLNA
        cal_kit : mistdata.CalKit
            Class to be instantiated for the calibration kit.
        match_resistance : float
            Resistance of the match used in the calibration kit.

        """
        self.freq = data.s11_freq  # in MHz
        self.data = data
        freq_Hz = self.freq * 1e6
        self.cal_kit = cal_kit(freq_Hz, match_resistance=match_resistance)

        self._raw_S11 = {
            "open": self.data.s11_open,
            "short": self.data.s11_short,
            "match": self.data.s11_match,
        }

    def _cal_step1(self):
        """
        Calibrate the raw S11 data with the calibration kit. This is the first
        step in the calibration process. End users should use the s11 property
        of the AntennaS11 or ReceiverS11 classes directly.

        Returns
        -------
        calibrated_s11 : dict
            Dictionary containing the calibrated S11 data.

        """
        raw_s11 = self.raw_S11
        s11_open = raw_s11.pop("open")
        s11_short = raw_s11.pop("short")
        s11_match = raw_s11.pop("match")
        cal_s11 = np.array([s11_open, s11_short, s11_match])
        vna_sparams = self.cal_kit.VNA_sparams(cal_S11)

        calibrated_s11 = {}
        for key, gamma in raw_s11.items():
            calibrated_s11[key] = cal_S11.de_embed_sparams(vna_sparams, gamma)
        
        return calibrated_s11

class AntennaS11(S11):

    def __init__(
        self,
        data,
        pathA_sparams,
        cal_kit=cal_s11.Keysight85033E,
        match_resistance=50,
    ):
        """
        Class for antenna S11 measurements.

        Parameters
        ----------
        data : mistdata.MISTData.DUTrecIn
        pathA_sparams : array-like
            S-parameters of the path A in the receiver. See Fig 19 in the MIST
            instrument paper, Monsalve et al. 2024.
        cal_kit : mistdata.CalKit
            Class to be instantiated for the calibration kit.
        match_resistance : float
            Resistance of the match used in the calibration kit.

        """
        super().__init__(data, cal_kit, match_resistance)
        self.pathA_sparams = pathA_sparams

    @property
    def raw_S11(self):
        """
        Return the raw S11 data.

        Returns
        -------
        s11 : dict
            Dictionary containing the raw S11 data.
            Keys are 'open', 'short', 'match', 'antenna', 'ambient', and
            'noise_source'.

        """
        s11 = self._raw_S11
        s11["antenna"] = self.data.s11_antenna
        s11["ambient"] = self.data.s11_ambient
        s11["noise_source"] = self.data.s11_noise_source

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
        calibrated_s11 = self._cal_step1()  # open, short, match calibration
        # de-embed S-parameters of path A in the receiver
        for key, gamma in calibrated_s11.items():
            calibrated_s11[key] = cal_S11.de_embed_sparams(
                self.pathA_sparams, gamma
            )

        return calibrated_s11


class ReceiverS11(S11):

    def __init__(
        self,
        data,
        pathB_sparams,
        pathC_sparams,
        cal_kit=cal_s11.Keysight85033E,
        match_resistance=50,
    ):
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
        cal_kit : mistdata.CalKit
            Class to be instantiated for the calibration kit.
        match_resistance : float
            Resistance of the match used in the calibration kit.

        """
        super().__init__(data, cal_kit, match_resistance)
        self.pathB_sparams = pathB_sparams
        self.pathC_sparams = pathC_sparams

    @property
    def raw_S11(self):
        """
        Return the raw S11 data.

        Returns
        -------
        s11 : dict
            Dictionary containing the raw S11 data.
            Keys are 'open', 'short', 'match', 'lna'.

        """
        s11 = self._raw_S11
        s11["lna"] = self.data.s11_lna

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
        calibrated_s11 = self._cal_step1()  # open, short, match calibration
        # de-embed S-parameters of path B in the receiver and embed the
        # S-parameters of path C in the receiver
        for key, gamma in calibrated_s11.items():
            cal_gamma = cal_S11.de_embed_sparams(self.pathB_sparams, gamma)
            cal_gamma = cal_S11.embed_sparams(self.pathC_sparams, cal_gamma)
            calibrated_s11[key] = cal_gamma

        return calibrated_s11

