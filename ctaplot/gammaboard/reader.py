from astropy.table import Table
import astropy.units as u
import pandas as pd


def read_params(filename):
    readers =  [LstchainDL2Reader, GammaLearnv01DL2Reader]
    for reader in readers:
        try:
            return reader(filename).read()
        except:
            continue
    raise ValueError(f"Can't read data from {filename}")


class BaseParamsReader:

    def __init__(self, filename, path):
        self.checker(filename)
        self.filename = filename
        self.path = path

    def read(self, **kwargs):
        self.data = self._reader(**kwargs)
        self.apply_name_mapping()
        self.apply_units()
        self.check_basic_columns()
        return self.data

    def _reader(self, **kwargs):
        return Table.read(self.filename, path=self.path, **kwargs)

    def checker(self, filename):
        """
        Check if the data is in the correct format.
        Add check about provider version.
        """
        if not (filename.endswith('.h5') or filename.endswith('.hdf5')):
            raise TypeError("Wrong format")

    def apply_name_mapping(self):
        for source_col_name, target_col_name in self.column_names_mapping.items():
            self.data.rename_column(source_col_name, target_col_name)

    def apply_units(self):
        for col_name, unit in self.unit_map.items():
            self.data[col_name].unit = unit

    @property
    def column_names_mapping(self):
        names_mapping = {}
        return names_mapping

    @property
    def unit_map(self):
        unit_map = {}
        return unit_map

    def check_basic_columns(self):
        column_names = ['true_altitude', 'reco_altitude',
                        'true_azimuth', 'reco_azimuth',
                        'true_energy', 'reco_energy',
                        'true_particle', 'reco_gammaness',
                        ]
        for name in column_names:
            if name not in self.data.colnames:
                raise ValueError(f"{name} not in columns")


class GammaLearnv01DL2Reader(BaseParamsReader):
    def __init__(self, filename):
        path = 'data'
        super().__init__(filename, path=path)

    def _reader(self, **kwargs):
        df = pd.read_hdf(self.filename, key=self.path)
        if 'reco_hadroness' in df.columns:
            df['reco_gammaness'] = 1 - df['reco_hadroness']
        return Table.from_pandas(df)

    @property
    def column_names_mapping(self):
        columns = {
            "mc_altitude": "true_altitude",
            "mc_azimuth": "true_azimuth",
            "mc_energy": "true_energy",
            "mc_impact_x": "true_core_x",
            "mc_impact_y": "true_core_y",
            "reco_impact_x": "reco_core_x",
            "reco_impact_y": "reco_core_y",
            "mc_particle": "true_particle"
        }
        return columns

    @property
    def unit_map(self):
        unit_map = {
            'true_core_x': u.m,
            'true_core_y': u.m,
            'reco_core_x': u.m,
            'reco_core_y': u.m,
            'true_energy': u.TeV,
            'reco_energy': u.TeV,
            'true_altitude': u.rad,
            'reco_altitude': u.rad,
            'true_azimuth': u.rad,
            'reco_azimuth': u.rad,
        }
        return unit_map

    def check_basic_columns(self):
        column_names = ['true_altitude', 'reco_altitude',
                        'true_azimuth', 'reco_azimuth',
                        'true_energy', 'reco_energy',
                        'true_particle',
                        ]
        for name in column_names:
            if name not in self.data.colnames:
                raise ValueError(f"{name} not in columns")


class LstchainDL2Reader(BaseParamsReader):
    """
    lstchain>=0.6
    """

    def __init__(self, filename):
        path = 'dl2/event/telescope/parameters/LST_LSTCam'
        super().__init__(filename, path=path)

    @property
    def column_names_mapping(self):
        columns = {
            "mc_alt": "true_altitude",
            "mc_az": "true_azimuth",
            "reco_alt": "reco_altitude",
            "reco_az": "reco_azimuth",
            "gammaness": "reco_gammaness",
            "mc_type": "true_particle",
            "mc_energy": "true_energy",
            "mc_core_x": "true_core_x",
            "mc_core_y": "true_core_y",
            "reco_core_x": "reco_core_x",
            "reco_core_y": "reco_core_y"
        }
        return columns

    @property
    def unit_map(self):
        unit_map = {
            'true_core_x': u.m,
            'true_core_y': u.m,
            'reco_core_x': u.m,
            'reco_core_y': u.m,
            'true_energy': u.TeV,
            'reco_energy': u.TeV,
            'true_altitude': u.rad,
            'reco_altitude': u.rad,
            'true_azimuth': u.rad,
            'reco_azimuth': u.rad,
        }
        return unit_map
