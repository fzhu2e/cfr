import os
import glob
import xarray as xr
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

class Case:
    ''' The class for postprocessing a GCM simulation case (e.g., CESM)
    
    Args:
        dirpath (str): the directory path where the reconstruction results are stored.
        load_num (int): the number of ensembles to load
        mode (str): if 'archive', then assume files are loaded from the archive director;
            if 'vars', then assume files are loaded from the director for the processed variables.
        verbose (bool, optional): print verbose information. Defaults to False.
    '''

    def __init__(self, dirpath=None, load_num=None, name=None, include_tags=['h'], exclude_tags=['nday', 'once'], mode='archive', verbose=False):
        self.fd = {}
        self.name = name
        if mode == 'archive':
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
                p_header(f'>>> {len(self.paths)} Case.paths:')
                print(self.paths)

        elif mode == 'vars':
            fpaths = glob.glob(os.path.join(dirpath, '*.nc'))
            for path in fpaths:
                fd_tmp = ClimateField().load_nc(path)
                vn = fd_tmp.da.name
                self.fd[vn] = fd_tmp

            if verbose:
                p_success(f'>>> Case loaded with vars: {list(self.fd.keys())}')

        else:
            raise ValueError('Wrong `mode` specified! Options: "archive" or "vars".')


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
                p_success(f'>>> Case.fd["{vn}"] created')

    def plot_atm_gm(self, figsize=[10, 6], nrow=2, ncol=2, wspace=0.3, hspace=0.2, xlim=(0, 100), lw=2,
                    xlabel='Time [yr]', ylable_dict=None, color_dict=None, ylim_dict=None):
        vars = {}
        if 'TS' in self.fd:
            gmst = self.fd['TS'].annualize().geo_mean()
            vars['GMST'] = gmst - 273.15
        elif 'TREFHT' in self.fd:
            gmst = self.fd['TREFHT'].annualize().geo_mean()
            vars['GMST'] = gmst

        if 'FSNT' in self.fd and 'FLNT' in self.fd:
            gmrestom = (self.fd['FSNT'] - self.fd['FLNT']).annualize().geo_mean()
            vars['GMRESTOM'] = gmrestom

        if 'LWCF' in self.fd:
            gmlwcf = self.fd['LWCF'].annualize().geo_mean()
            vars['GMLWCF'] = gmlwcf

        if 'SWCF' in self.fd:
            gmswcf = self.fd['SWCF'].annualize().geo_mean()
            vars['GMSWCF'] = gmswcf

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrow, ncol)
        gs.update(wspace=wspace, hspace=hspace)

        _ylim_dict = {
            'GMST': (13.5, 15.5),
            'GMRESTOM': (-1, 3),
            'GMLWCF': (24, 26),
            'GMSWCF': (-54, -44),
        }
        if ylim_dict is not None:
            _ylim_dict.update(ylim_dict)

        _ylb_dict = {
            'GMST': r'GMST [$^\circ$C]',
            'GMRESTOM': r'GMRESTOM [W/m$^2$]',
            'GMLWCF': r'GMLWCF [W/m$^2$]',
            'GMSWCF': r'GMSWCF [W/m$^2$]',
        }
        if ylable_dict is not None:
            _ylb_dict.update(ylable_dict)

        _clr_dict = {
            'GMST': 'tab:red',
            'GMRESTOM': 'tab:blue',
            'GMLWCF': 'tab:green',
            'GMSWCF': 'tab:orange',
        }
        if color_dict is not None:
            _clr_dict.update(color_dict)

        ax = {}
        i = 0
        i_row, i_col = 0, 0
        for k, v in vars.items():
            ax[k] = fig.add_subplot(gs[i_row, i_col])
            if i_row == nrow-1:
                _xlb = xlabel
            else:
                _xlb = None


            if k == 'GMRESTOM':
                ax[k].axhline(y=0, linestyle='--', color='tab:grey')
            elif k == 'GMLWCF':
                ax[k].axhline(y=25, linestyle='--', color='tab:grey')
            elif k == 'GMSWCF':
                ax[k].axhline(y=-47, linestyle='--', color='tab:grey')

            v.plot(
                ax=ax[k], xlim=xlim, ylim=_ylim_dict[k],
                linewidth=lw, xlabel=_xlb, ylabel=_ylb_dict[k],
                color=_clr_dict[k],
            )

            i += 1
            i_col += 1

            if i % 2 == 0:
                i_row += 1

            if i_col == ncol:
                i_col = 0

        if self.name is not None:
            fig.suptitle(self.name)

        return fig, ax
            
                


            