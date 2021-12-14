import pandas as pd
from astropy.utils.decorators import deprecated


@deprecated("30/11/2020", "In the future, this format will no longer be supported")
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


@deprecated("30/11/2020", "In the future, this format will no longer be supported")
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
        "true_alt": "mc_altitude",
        "mc_alt": "mc_altitude",
        "true_az": "mc_azimuth",
        "mc_az": "mc_azimuth",
        "reco_alt": "reco_altitude",
        "reco_az": "reco_azimuth",
        "gammaness": "reco_gammaness",
        "mc_type": "mc_particle",
    })

    if data['mc_energy'].min() > 0.1 and data['mc_energy'].max() < 10:
        # true_energy is probably in log(GeV)
        data['mc_energy'] = 10**(data['mc_energy'] - 3)
        data['mc_energy'] = 10**(data['mc_energy'] - 3)

    return data

