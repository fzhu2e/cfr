import xarray as xr
import pybaywatch as pb
import numpy as np

from . import utils
from . import obs

class IdenticalTS:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['TS', 'tas']

    def forward(self):
        if 'TS' in self.clim:
            output = self.clim['TS'].values
        elif 'tas' in self.clim:
            output = self.clim['tas'].values

        return output

class IdenticalSST:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['TEMP', 'SST', 'sst', 'tos']

    def forward(self):
        if 'TEMP' in self.clim:
            if 'z_t' in self.clim['TEMP'].dims:
                output = self.clim['TEMP'].isel(z_t=0).values
            else:
                output = self.clim['TEMP'].values

        elif 'SST' in self.clim:
            output = self.clim['SST'].values
        elif 'sst' in self.clim:
            output = self.clim['sst'].values
        elif 'tos' in self.clim:
            output = self.clim['tos'].values
        
        return output

class IdenticalSSS:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['SALT', 'SSS', 'sss', 'sos']

    def forward(self):
        if 'SALT' in self.clim:
            if 'z_t' in self.clim['SALT'].dims:
                output = self.clim['SALT'].isel(z_t=0).values
            else:
                output = self.clim['SALT'].values
        elif 'SSS' in self.clim:
            output = self.clim['SSS'].values
        elif 'sss' in self.clim:
            output = self.clim['sss'].values
        elif 'sos' in self.clim:
            output = self.clim['sos'].values

        return output

class IdenticalSSTSSS:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['TEMP', 'SALT']

    def forward(self):
        output = self.clim['TEMP'].isel(z_t=0).values+self.clim['SALT'].isel(z_t=0).values
        return output

class TEX86:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['TEMP', 'tos', 'sst']

    def forward(self, seed=2333, mode='analog', type='SST', tolerance=1):
        if 'TEMP' in self.clim:
            sst = self.clim['TEMP'].isel(z_t=0).values
        elif 'tos' in self.clim:
            sst = self.clim['tos'].values
        elif 'sst' in self.clim:
            sst = self.clim['sst'].values

        lat = self.record.data.lat
        lon = self.record.data.lon
        lon180 = utils.lon180(lon)

        # run
        self.params = {
            'lat': lat,
            'lon': lon180,
            'temp': sst,
            'seed': seed,
            'type': type,
            'mode': mode,
            'tolerance': tolerance,
        }
        res = pb.TEX_forward(**self.params)
        if res['status'] == 'FAIL':
            utils.p_warning(f'>>> Forward modeling failed for proxy: {self.meta["pid"]}')
            output = None
        else:
            output = np.median(res['values'], axis=1)

        return output

class UK37:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['TEMP', 'tos', 'sst']

    def forward(self, order=3, seed=2333):
        if 'TEMP' in self.clim:
            sst = self.clim['TEMP'].isel(z_t=0).values
        elif 'tos' in self.clim:
            sst = self.clim['tos'].values
        elif 'sst' in self.clim:
            sst = self.clim['sst'].values

        # run
        self.params = {
            'sst': sst,
            'order': order,
            'seed': seed,
        }
        res = pb.UK_forward(**self.params)
        output = np.median(res['values'], axis=1)
        return output

class MgCa:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['TEMP', 'tos', 'sst', 'SALT', 'sos', 'sss']

    def forward(self, age, omega=None, pH=None, clean=None, species=None, sw=2, H=1, seed=2333):
        if 'TEMP' in self.clim and 'SALT' in self.clim:
            sst = self.clim['TEMP'].isel(z_t=0).values
            sss = self.clim['SALT'].isel(z_t=0).values
        elif 'tos' in self.clim and 'sos' in self.clim:
            sst = self.clim['tos'].values
            sss = self.clim['sos'].values
        elif 'sst' in self.clim and 'sss' in self.clim:
            sst = self.clim['sst'].values
            sss = self.clim['sss'].values

        # get omega and pH
        lat = self.record.data.lat
        lon = self.record.data.lon
        depth = self.record.data.depth
        if omega is None and pH is None:
            lon180 = np.mod(lon + 180, 360) - 180
            omega, pH = pb.core.omgph(lat, lon180, depth)

        if clean is None: clean = self.record.data.clean
        if species is None: species = self.record.data.species

        # run
        self.params = {
            'age': age,
            'sst': sst,
            'salinity': sss,
            'pH': pH,
            'omega': omega,
            'species': species,
            'clean': clean,
            'sw': sw,
            'H': H,
            'seed': seed,
        }
        res = pb.MgCa_forward(**self.params)
        output = np.median(res['values'], axis=1)
        return output

class d18Oc:
    def __init__(self, record:obs.ProxyRecord=None):
        self.record = record

    @property
    def clim_vns(self):
        return ['TEMP', 'tos', 'sst', 'SALT', 'sos', 'sss', 'd18Osw']

    def forward(self, pH=None, species=None, pH_type=0, seed=2333):
        if 'TEMP' in self.clim and 'SALT' in self.clim:
            sst = self.clim['TEMP'].isel(z_t=0).values
            sss = self.clim['SALT'].isel(z_t=0).values
        elif 'tos' in self.clim and 'sos' in self.clim:
            sst = self.clim['tos'].values
            sss = self.clim['sos'].values
        elif 'sst' in self.clim and 'sss' in self.clim:
            sst = self.clim['sst'].values
            sss = self.clim['sss'].values

        d18Osw = self.clim['d18Osw'].values
        if pH is None: pH = self.record.data.species
        if species is None: species = self.record.data.species

        # run
        self.params = {
            'sst': sst,
            'd18Osw': d18Osw,
            'sss': sss,
            'pH': pH,
            'pH_type': pH_type,
            'species': species,
            'seed': seed,
        }
        res = pb.d18Oc_forward(**self.params)
        output = np.median(res['values'], axis=1)
        return output