import pandas as pd


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
        "true_alt": "mc_altitude",
        "true_az": "mc_azimuth",
        "reco_alt": "reco_altitude",
        "reco_az": "reco_azimuth",
        "gammaness": "reco_gammaness",
    })

    if data['true_energy'].min() > 0.1 and data['true_energy'].max() < 10:
        # energy is probably in log(GeV)
        data['reco_energy'] = 10 ** (data['reco_energy'] - 3)
        data['true_energy'] = 10**(data['true_energy'] - 3)

    return data

