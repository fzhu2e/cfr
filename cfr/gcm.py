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

class GCMCase:
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
                p_success(f'>>> GCMCase loaded with vars: {list(self.fd.keys())}')

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
                fname = f'{vn}.nc'
                save_path = os.path.join(save_dirpath, fname)
                self.fd[vn].to_nc(save_path, compress_params=compress_params)

            if verbose:
                p_success(f'>>> GCMCase.fd["{vn}"] created')

    def calc_atm_gm(self, vns=['GMST', 'RESTOM', 'LWCF', 'SWCF'], verbose=False):
        self.vars = {}

        for vn in vns:
            if vn == 'GMST':
                v = 'TS' if 'TS' in self.fd else 'TREFHT'
                gmst = self.fd[v].annualize().geo_mean()
                self.vars[vn] = gmst - 273.15

            elif vn == 'RESTOM':
                restom = (self.fd['FSNT'] - self.fd['FLNT']).annualize().geo_mean()
                self.vars[vn] = restom
            
            else:
                self.vars[vn] = self.fd[vn].annualize().geo_mean()

            if verbose:
                p_success(f'>>> GCMCase.vars["{vn}"] created')

    def plot_atm_gm(self, figsize=[10, 6], ncol=2, wspace=0.3, hspace=0.2, xlim=(0, 100), lw=2, title=None,
                    label=None, xlabel='Time [yr]', ylable_dict=None, color_dict=None, ylim_dict=None, ax=None):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = {}

        nrow = int(np.ceil(len(self.vars)/ncol))
        gs = gridspec.GridSpec(nrow, ncol)
        gs.update(wspace=wspace, hspace=hspace)

        _ylim_dict = {
            'GMST': (13.5, 15.5),
            'RESTOM': (-1, 3),
            'LWCF': (24, 26),
            'SWCF': (-54, -44),
        }
        if ylim_dict is not None:
            _ylim_dict.update(ylim_dict)

        _ylb_dict = {
            'GMST': r'GMST [$^\circ$C]',
            'RESTOM': r'RESTOM [W/m$^2$]',
            'LWCF': r'LWCF [W/m$^2$]',
            'SWCF': r'SWCF [W/m$^2$]',
        }
        if ylable_dict is not None:
            _ylb_dict.update(ylable_dict)

        _clr_dict = {
            'GMST': 'tab:red',
            'RESTOM': 'tab:blue',
            'LWCF': 'tab:green',
            'SWCF': 'tab:orange',
        }
        if color_dict is not None:
            _clr_dict.update(color_dict)

        i = 0
        i_row, i_col = 0, 0
        for k, v in self.vars.items():
            if 'fig' in locals():
                ax[k] = fig.add_subplot(gs[i_row, i_col])

            if i_row == nrow-1:
                _xlb = xlabel
            else:
                _xlb = None


            if k == 'RESTOM':
                ax[k].axhline(y=0, linestyle='--', color='tab:grey')
            elif k == 'LWCF':
                ax[k].axhline(y=25, linestyle='--', color='tab:grey')
            elif k == 'SWCF':
                ax[k].axhline(y=-47, linestyle='--', color='tab:grey')

            v.plot(
                ax=ax[k], xlim=xlim, ylim=_ylim_dict[k],
                linewidth=lw, xlabel=_xlb, ylabel=_ylb_dict[k],
                color=_clr_dict[k], label=label,
            )

            i += 1
            i_col += 1

            if i % 2 == 0:
                i_row += 1

            if i_col == ncol:
                i_col = 0

        if title is not None:
            fig.suptitle(title)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax
            
                

class GCMCases:
    ''' The class for postprocessing multiple GCM simulation cases (e.g., CESM)
    '''
    def __init__(self, case_dict=None):
        self.case_dict = case_dict
        for k, v in self.case_dict.items():
            v.name = k

    def calc_atm_gm(self, vns=['GMST', 'RESTOM', 'LWCF', 'SWCF'], verbose=False):
        for k, v in self.case_dict.items():
            p_header(f'Processing case: {k} ...')
            v.calc_atm_gm(vns=vns, verbose=verbose)

    def plot_atm_gm(self, lgd_kws=None, lgd_idx=1):
        _clr_dict = {
            'GMST': None,
            'RESTOM': None,
            'LWCF': None,
            'SWCF': None,
        }
        for k, v in self.case_dict.items():
            if 'fig' not in locals():
                fig, ax = v.plot_atm_gm(color_dict=_clr_dict, label=v.name)
            else:
                ax = v.plot_atm_gm(ax=ax, color_dict=_clr_dict, label=v.name)

        _lgd_kws = {
            'frameon': False,
            'loc': 'upper left',
            'bbox_to_anchor': [1.1, 1],
        }
        if lgd_kws is not None:
            _lgd_kws.update(lgd_kws)

        vn = list(ax.keys())[lgd_idx]
        ax[vn].legend(**_lgd_kws)

        return fig, ax
            