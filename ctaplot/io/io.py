import pandas as pd
import numpy as np


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
    if 'true_energy' in data.columns:
        data = data.rename(columns={
            "true_energy": "mc_energy",
            "true_alt": "mc_altitude",
            "true_az": "mc_azimuth",
            "reco_alt": "reco_altitude",
            "reco_az": "reco_azimuth",
            "gammaness": "reco_gammaness",
        })
    if 'mc_alt' in data.columns:
        data = data.rename(columns={
            'mc_alt': 'mc_altitude',
            'mc_az': 'mc_azimuth',
            'reco_alt': 'reco_altitude',
            'reco_az': 'reco_azimuth',
            'gammaness': 'reco_gammaness',
            'mc_type': 'mc_particle',
            'reco_type': 'reco_particle'
        })

    if data['mc_energy'].min() > 0.1 and data['mc_energy'].max() < 10:
        # energy is probably in log(GeV)
        data['reco_energy'] = 10 ** (data['reco_energy'] - 3)
        data['mc_energy'] = 10**(data['mc_energy'] - 3)

    return data


def read_pyirf_dl2_data(filename, key='dl2'):
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

    for col in ['reco_alt', 'reco_az', 'mc_alt', 'mc_az']:
        data[col] = np.deg2rad(data[col])

    data = data.rename(columns={
        'mc_alt': 'mc_altitude',
        'mc_az': 'mc_azimuth',
        'reco_alt': 'reco_altitude',
        'reco_az': 'reco_azimuth',
        'gammaness': 'reco_gammaness',
        'mc_type': 'mc_particle'
    })

    reco_particle = data.filter([col for col in data.columns if 'reco_proba' in col])
    reco_particle = reco_particle.idxmax(axis=1)
    reco_particle = reco_particle.apply(lambda x: int(x.split('_')[-1]))
    data['reco_particle'] = reco_particle

    if data['mc_energy'].min() > 0.1 and data['mc_energy'].max() < 10:
        # energy is probably in log(GeV)
        data['reco_energy'] = 10 ** (data['reco_energy'] - 3)
        data['mc_energy'] = 10**(data['mc_energy'] - 3)

    if 'pass_best_cutoff' in data.columns and 'pass_angular_cut' in data.columns:
        data = data.query('pass_best_cutoff == True and pass_angular_cut == True')

    return data
