import os
import glob
import xarray as xr
import numpy as np
from tqdm import tqdm
import datetime
from . import visual
from .climate import ClimateField
from .utils import (
    coefficient_efficiency,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
)

class CESMarchive:
    ''' The class for postprocessing the CESM archives
    
    Args:
        dirpath (str): the directory path where the reconstruction results are stored.
        load_num (int): the number of ensembles to load
        verbose (bool, optional): print verbose information. Defaults to False.
    '''

    def __init__(self, dirpath, load_num=None, include_tags=['h'], exclude_tags=['nday', 'once'], verbose=False):
        if type(include_tags) is str:
            include_tags = [include_tags]
        if type(exclude_tags) is str:
            exclude_tags = [exclude_tags]

        try:
            fpaths = glob.glob(os.path.join(dirpath, '*.nc'))
            
            self.paths = []
            for path in fpaths:
                fname = os.path.basename(path)
                include = True

                for in_tag in include_tags:
                    if in_tag not in fname:
                        include = False

                for ex_tag in exclude_tags:
                    if ex_tag in fname:
                        include = False

                if include:
                    self.paths.append(path)

            self.paths = sorted(self.paths)
            if load_num is not None:
                self.paths = self.paths[:load_num]
        except:
            raise ValueError('No CESM archive files available in `dirpath`!')

        if verbose:
            p_header(f'>>> {len(self.paths)} CESMarchive.paths:')
            print(self.paths)

        self.fd = {}

    def get_ds(self, fid=0):
        ''' Get a `xarray.Dataset` from a certain file
        '''
        with xr.open_dataset(self.paths[fid]) as ds:
            return ds

    def load(self, vn_list, time_name='time', adjust_month=False,
             save_dirpath=None, compress_params=None, verbose=False):
        ''' Load variables.

        Args:
            vn_list (list): list of variable names.
            time_name (str): the name of the time dimension.
            adjust_month (bool): the current CESM version has a bug that the output
                has a time stamp inconsistent with the filename with 1 months off, hence
                requires an adjustment.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        if type(vn_list) is str:
            vn_list = [vn_list]

        ds_list = []
        for path in tqdm(self.paths, desc='Loading files'):
            with xr.open_dataset(path) as ds_tmp:
                ds_list.append(ds_tmp)

        for vn in vn_list:
            p_header(f'Extracting {vn} ...')
            da = xr.concat([ds[vn] for ds in ds_list], dim=time_name)
            if adjust_month:
                da[time_name] = da[time_name].get_index('time') - datetime.timedelta(days=1)
            self.fd[vn] = ClimateField().from_da(da)

            if save_dirpath is not None:
                year_start = da[time_name].values[0].year
                month_start = da[time_name].values[0].month
                year_end = da[time_name].values[-1].year
                month_end = da[time_name].values[-1].month
                fname = f'{vn}.{year_start:04d}{month_start:02d}-{year_end:04d}{month_end:02d}.nc'
                save_path = os.path.join(save_dirpath, fname)
                self.fd[vn].to_nc(save_path, compress_params=compress_params)

            if verbose:
                p_success(f'>>> CESMarchive.fd["{vn}"] created')

            