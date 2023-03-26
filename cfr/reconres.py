import xarray as xr
import numpy as np
from .climate import ClimateField
from scipy.stats import pearsonr
import glob
import os
from .ts import EnsTS
from .utils import (
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
)

class ReconRes:
    ''' The class for reconstruction results

    Args:
        dirpath (str): the directory path where the reconstruction results are stored.
        load_num (int): the number of ensembles to load
        verbose (bool, optional): print verbose information. Defaults to False.
    '''

    def __init__(self, dirpath, load_num=None, verbose=False):
        try:
            recon_paths = sorted(glob.glob(os.path.join(dirpath, 'job_r*_recon.nc')))
            if load_num is not None:
                recon_paths = recon_paths[:load_num]
            self.paths = recon_paths
        except:
            raise ValueError('No ""')

        if verbose:
            p_header(f'>>> res.paths:')
            print(self.paths)

        self.recons = {}
        self.da = {}
    
    def load(self, vn_list, verbose=False):
        ''' Load reconstruction results.

        Args:
            vn_list (list): list of variable names; supported names, taking 'tas' as an example:

                * ensemble fields: 'tas'
                * ensemble timeseries: 'tas_gm', 'tas_nhm', 'tas_shm'

            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        if type(vn_list) is str:
            vn_list = [vn_list]

        for vn in vn_list:
            da_list = []
            for path in self.paths:
                with xr.open_dataset(path) as ds_tmp:
                    da_list.append(ds_tmp[vn])

            da = xr.concat(da_list, dim='ens')
            if 'ens' not in da.coords:
                da.coords['ens'] = np.arange(len(self.paths))
            da = da.transpose('time', 'ens', ...)

            self.da[vn] = da
            if 'lat' not in da.coords and 'lon' not in da.coords:
                self.recons[vn] = EnsTS(time=da.time, value=da.values, value_name=vn)
            else:
                self.recons[vn] = ClimateField(da.mean(dim='ens'))

            if verbose:
                p_success(f'>>> ReconRes.recons["{vn}"] created')
                p_success(f'>>> ReconRes.da["{vn}"] created')