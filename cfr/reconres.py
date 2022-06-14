from pathlib import Path
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
from .climate import ClimateField
import glob
import os
import copy
from .utils import (
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
)
import pens

class ReconRes:
    ''' Reconstruction Result
    '''

    def __init__(self, job_dirpath, load_num=None, verbose=False):
        try:
            recon_paths = sorted(glob.glob(os.path.join(job_dirpath, 'job_r*_recon.nc')))
            if load_num is not None:
                recon_paths = recon_paths[:load_num]
            self.paths = recon_paths
        except:
            raise ValueError('No ""')

        if verbose:
            p_header(f'>>> recon.paths:')
            print(self.paths)

        self.recons = {}
        self.da = {}
    
    def load(self, vn, verbose=False):
        ''' Load reconstruction results

        Args:
            vn (str): the variable name, supported names, taking 'tas' for example:
            - ensemble timeseries: 'tas_gm', 'tas_nhm', 'tas_shm'
            - ensemble fields: 'tas'
        '''
        da_list = []
        for path in self.paths:
            with xr.open_dataset(path) as ds_tmp:
                da_list.append(ds_tmp[vn])

        da = xr.concat(da_list, dim='ens')
        if 'ens' not in da.coords:
            da.coords['ens'] = np.arange(len(self.paths))
        da = da.transpose('year', 'ens', ...)

        self.da[vn] = da
        if 'lat' not in da.coords and 'lon' not in da.coords:
            self.recons[vn] = pens.EnsembleTS(time=da['year'], value=da.values)
        else:
            self.recons[vn] = ClimateField().from_da(da.mean(dim='ens'), time_name='year')

        if verbose:
            p_success(f'>>> ReconRes.da["{vn}"] created')
            p_success(f'>>> ReconRes.recons["{vn}"] created')
