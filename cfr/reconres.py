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
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .visual import CartopySettings

class ReconRes:
    ''' The class for reconstruction results '''
    def __init__(self, dirpath, load_num=None, verbose=False):
        ''' Initialize a reconstruction result object.

        Args:
            dirpath (str): the directory path where the reconstruction results are stored.
            load_num (int): the number of ensembles to load
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
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


    def valid(self, target_dict, stat=['corr'], timespan=None, verbose=False):
        ''' Validate against a target dictionary

        Args:
            target_dict (dict): a dictionary of multiple variables for validation.
            stat (list of str): the statistics to calculate. Supported quantaties:

                * 'corr': correlation coefficient
                * 'R2': coefficient of determination
                * 'CE': coefficient of efficiency

            timespan (list or tuple): the timespan over which to perform the validation.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        if type(stat) is not list: stat = [stat]
        vn_list = target_dict.keys()
        self.load(vn_list, verbose=verbose)
        valid_fd, valid_ts = {}, {}
        for vn in vn_list:
            p_header(f'>>> Validating variable: {vn} ...')
            if isinstance(self.recons[vn], ClimateField):
                for st in stat:
                    valid_fd[f'{vn}_{st}'] = self.recons[vn].compare(target_dict[vn], stat=st, timespan=timespan)
                    valid_fd[f'{vn}_{st}'].plot_kwargs.update({'cbar_orientation': 'horizontal', 'cbar_pad': 0.1})
                    if verbose: p_success(f'>>> ReconRes.valid_fd[{vn}_{st}] created')
            elif isinstance(self.recons[vn], EnsTS):
                valid_ts[vn] = self.recons[vn].compare(target_dict[vn], timespan=timespan)
                if verbose: p_success(f'>>> ReconRes.valid_ts[{vn}] created')

        self.valid_fd = valid_fd
        self.valid_ts = valid_ts

            
    def plot_valid(self, recon_name_dict=None, target_name_dict=None,
                   valid_ts_kws=None, valid_fd_kws=None):
        ''' Plot the validation result

        Args:
            recon_name_dict (dict): the dictionary for variable names in the reconstruction. For example, {'tas': 'LMR/tas', 'nino3.4': 'NINO3.4 [K]'}.
            target_name_dict (dict): the dictionary for variable names in the validation target. For example, {'tas': '20CRv3', 'nino3.4': 'BC09'}.
            valid_ts_kws (dict): the dictionary of keyword arguments for validating the timeseries.
            valid_fd_kws (dict): the dictionary of keyword arguments for validating the field.
        '''
        # print(valid_fd_kws)
        valid_fd_kws = {} if valid_fd_kws is None else valid_fd_kws
        valid_ts_kws = {} if valid_ts_kws is None else valid_ts_kws
        target_name_dict = {} if target_name_dict is None else target_name_dict
        recon_name_dict = {} if recon_name_dict is None else recon_name_dict

        if 'latlon_range' in valid_fd_kws:
            lat_min, lat_max, lon_min, lon_max = valid_fd_kws['latlon_range']
        else:
            lat_min, lat_max, lon_min, lon_max = -90, 90, 0, 360

        fig, ax = {}, {}
        for k, v in self.valid_fd.items():
            vn, st = k.split('_')
            if vn not in target_name_dict: target_name_dict[vn] = 'obs'
            fig[k], ax[k] = v.plot(
                title=f'{st}({recon_name_dict[vn]}, {target_name_dict[vn]}), mean={v.geo_mean(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max).value[0,0]:.2f}',
                **valid_fd_kws)

        for k, v in self.valid_ts.items():
            v.ref_name = target_name_dict[k]
            if v.value.shape[-1] > 1:
                fig[k], ax[k] = v.plot_qs(**valid_ts_kws)
            else:
                fig[k], ax[k] = v.plot(label='recon', **valid_ts_kws)
            ax[k].set_ylabel(recon_name_dict[k])

        return fig, ax