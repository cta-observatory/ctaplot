import pandas as pd


def read_lst_dl2_data(filename, key='dl2/event/telescope/parameters/LST_LSTCam'):
    """
    Read lst dl2 data and return a dataframe with right keys for gammaboard

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
        "true_alt": "mc_alt",
        "true_az": "mc_az",
        "reco_alt": "reco_alt",
        "reco_az": "reco_az",
        "gammaness": "reco_gammaness",
    })

    if data['true_energy'].min() > 0.1 and data['true_energy'].max() < 10:
        # energy is probably in log(GeV)
        data['reco_energy'] = 10 ** (data['reco_energy'] - 3)
        data['true_energy'] = 10**(data['true_energy'] - 3)

    return data


def read_glearn_dl2_data(filename):
    """
    Read glearn dl2 data and return a dataframe with right keys for gammaboard

    Parameters
    ----------
    filename: path

    Returns
    -------
    `pandas.DataFrame`
    """

    data = pd.read_hdf(filename, key='data')
    if 'mc_type' not in data:
        data = data.rename(columns={
            "mc_altitude": "mc_alt",
            "mc_azimuth": "mc_az",
            "reco_altitude": "reco_alt",
            "reco_azimuth": "reco_az",
            "mc_particle": "mc_type",
            "mc_impact_x": "mc_core_x",
            "mc_impact_y": "mc_core_y",
            "reco_impact_x": "reco_core_x",
            "reco_impact_y": "reco_core_y",
        })

    data['mc_core_x'] *= 1000
    data['mc_core_y'] *= 1000
    data['reco_core_x'] *= 1000
    data['reco_core_y'] *= 1000

    return data
