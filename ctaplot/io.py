import json
import numpy as np


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


