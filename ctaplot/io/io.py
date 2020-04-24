import json
import numpy as np
import pandas as pd
from astropy.utils import deprecated


@deprecated("08/10/2019", "The mc_run_header should be in the data file. Under `/simulation/run_config` for HDF5 files")
class mc_run_header():

    def __init__(self):
        pass

    def read(self, filename):
        """
        Read a json file containing a Monte-Carlo run header.

        Parameters
        ----------
        filename: string
            path to the json file

        Returns
        -------
        dict
        """
        with open(filename) as file:
            self.data = json.load(file)

    def get_e_range(self):
        """
        return simulated energy range

        Returns
        -------
        [e_min, e_max]
        """
        return self.data['E_range']

    def get_core_range(self):
        """
        return core range

        Returns
        -------
        [r_min, r_max]
        """
        return self.data['core_range']

    def get_core_area(self):
        """
        return simulated core area

        Returns
        -------
        area: float
        """
        if self.data['core_pos_mode'] == 1:
            return np.pi * (self.data['core_range'][1] - self.data['core_range'][0])**2
        else:
            raise Exception("Sorry I do not know this simulation mode")

    def get_number_simulated_events(self):
        """
        return the number of simulated events (number of simulated showers times number of use)

        Returns
        -------
        float
        """
        return self.data['n_showers'] * self.data['n_use']


def read_lst_dl1_data(filename, key='dl1/event/telescope/parameters/LST_LSTCam'):
    """
    Read lst dl1 data and return a dataframe with right keys for gammaboard

    Parameters
    ----------
    filename: path
    key: dataset path in file

    Returns
    -------
    `pandas.DataFrame`
    """

    data = pd.read_hdf(filename, key=key)
    # data.rename({
    #
    # })
    return data


def read_lst_dl2_data(filename, key='dl2/event/telescope/parameters/LST_LSTCam'):
    """
    Read lst dl1 data and return a dataframe with right keys for gammaboard

    Parameters
    ----------
    filename: path
    key: dataset path in file

    Returns
    -------
    `pandas.DataFrame`
    """

    data = pd.read_hdf(filename, key=key)

    data = data.rename(columns={
        "mc_alt": "mc_altitude",
        "mc_az": "mc_azimuth",
        "reco_alt": "reco_altitude",
        "reco_az": "reco_azimuth",
        "gammaness": "reco_gammaness",
    })

    if data['mc_energy'].min() > 0.1 and data['mc_energy'].max() < 10:
        # energy is probably in log(GeV)
        data['reco_energy'] = 10 ** (data['reco_energy'] - 3)
        data['mc_energy'] = 10**(data['mc_energy'] - 3)

    return data

