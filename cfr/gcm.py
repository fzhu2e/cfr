import os
import glob
import xarray as xr
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from . import visual
from .ts import EnsTS
from .climate import ClimateField
from .utils import (
    coefficient_efficiency,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
    year_float2datetime,
)

class GCMCase:
    ''' The class for postprocessing a GCM simulation case (e.g., CESM)
    
    Args:
        dirpath (str): the directory path where the reconstruction results are stored.
        load_num (int): the number of ensembles to load
        verbose (bool, optional): print verbose information. Defaults to False.
    '''

    def __init__(self, dirpath=None, load_num=None, name=None, include_tags=[], exclude_tags=[], verbose=False):
        self.fd = {}  # ClimateField
        self.ts = {}  # EnsTS
        self.name = name

        if type(include_tags) is str:
            include_tags = [include_tags]
        if type(exclude_tags) is str:
            exclude_tags = [exclude_tags]

        if dirpath is not None:
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

        if verbose:
            p_header(f'>>> {len(self.paths)} GCMCase.paths:')
            print(self.paths)

    def get_ds(self, idx=0):
        ''' Get a `xarray.Dataset` from a certain file
        '''
        with xr.open_dataset(self.paths[idx]) as ds:
            return ds

    def load(self, vars=None, time_name='time', z_name='z_t', z_val=None,
             adjust_month=False, mode='time-slice',
             save_dirpath=None, compress_params=None, verbose=False):
        ''' Load variables.

        Args:
            vars (list): list of variable names.
            time_name (str): the name of the time dimension.
            z_name (str): the name of the z dimension (e.g., for ocean output).
            z_val (float, int, list): the value(s) of the z dimension to pick (e.g., for ocean output).
            adjust_month (bool): the current CESM version has a bug that the output
                has a time stamp inconsistent with the filename with 1 months off, hence
                requires an adjustment.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        if type(vars) is str:
            vars = [vars]

        if mode == 'timeslice':
            if vars is None:
                raise ValueError('Should specify `vars` if mode is "timeslice".')

            ds_list = []
            for path in tqdm(self.paths, desc='Loading files'):
                with xr.open_dataset(path) as ds_tmp:
                    ds_list.append(ds_tmp)

            for vn in vars:
                p_header(f'>>> Extracting {vn} ...')
                if z_val is None:
                    da = xr.concat([ds[vn] for ds in ds_list], dim=time_name)
                else:
                    da = xr.concat([ds[vn].sel({z_name: z_val}) for ds in ds_list], dim=time_name)

                if adjust_month:
                    da[time_name] = da[time_name].get_index(time_name) - datetime.timedelta(days=1)

                self.fd[vn] = ClimateField(da)

                if save_dirpath is not None:
                    fname = f'{vn}.nc'
                    save_path = os.path.join(save_dirpath, fname)
                    self.fd[vn].to_nc(save_path, compress_params=compress_params)

                if verbose:
                    p_success(f'>>> GCMCase.fd["{vn}"] created')

        elif mode == 'timeseries':
            for path in self.paths:
                fd_tmp = ClimateField().load_nc(path)
                vn = fd_tmp.da.name
                self.fd[vn] = fd_tmp

            if verbose:
                p_success(f'>>> GCMCase loaded with vars: {list(self.fd.keys())}')

        else:
            raise ValueError('Wrong `mode` specified! Options: "timeslice" or "timeseries".')

    def calc_atm_gm(self, vars=['GMST', 'GMRESTOM', 'GMLWCF', 'GMSWCF'], verbose=False):

        for vn in vars:
            if vn == 'GMST':
                v = 'TS' if 'TS' in self.fd else 'TREFHT'
                gmst = self.fd[v].annualize().geo_mean()
                self.ts[vn] = gmst - 273.15

            elif vn == 'GMRESTOM':
                restom = (self.fd['FSNT'] - self.fd['FLNT']).annualize().geo_mean()
                self.ts[vn] = restom
            
            else:
                self.ts[vn] = self.fd[vn[2:]].annualize().geo_mean()

            if verbose:
                p_success(f'>>> GCMCase.ts["{vn}"] created')

    def to_ds(self):
        ''' Convert to a `xarray.Dataset`
        '''
        da_dict = {}
        for k, v in self.fd.items():
            da_dict[k] = v.da

        for k, v in self.ts.items():
            time_name = v.time.name
            da_dict[k] = xr.DataArray(v.value[:, 0], dims=[time_name], coords={time_name: v.time}, name=k)

        ds = xr.Dataset(da_dict)
        if self.name is not None:
            ds.attrs['casename'] = self.name

        return ds

    def to_nc(self, path, verbose=True, compress_params=None):
        ''' Output the GCM case to a netCDF file.

        Args:
            path (str): the path where to save
        '''
        _comp_params = {'zlib': True, 'least_significant_digit': 2}
        encoding_dict = {}
        if compress_params is not None:
            _comp_params.update(compress_params)

        for k, v in self.fd.items():
            encoding_dict[k] = _comp_params

        for k, v in self.ts.items():
            encoding_dict[k] = _comp_params

        try:
            dirpath = os.path.dirname(path)
            os.makedirs(dirpath, exist_ok=True)
        except:
            pass

        ds = self.to_ds()

        ds.to_netcdf(path, encoding=encoding_dict)
        if verbose: p_success(f'>>> GCMCase saved to: {path}')

    def load_nc(self, path, verbose=False):
        case = GCMCase()
        ds = xr.open_dataset(path)
        if 'casename' in ds.attrs:
            case.name = ds.attrs['casename']

        for vn in ds.keys():
            if vn[:2] == 'GM':
                case.ts[vn] = EnsTS(time=ds[vn].year, value=ds[vn].values)
                if verbose:
                    p_success(f'>>> GCMCase.ts["{vn}"] created')
            else:
                case.fd[vn] = ClimateField(ds[vn])
                if verbose:
                    p_success(f'>>> GCMCase.fd["{vn}"] created')

        return case


    def plot_ts(self, vars=['GMST', 'GMRESTOM', 'GMLWCF', 'GMSWCF'], figsize=[10, 6], ncol=2, wspace=0.3, hspace=0.2, xlim=(0, 100), title=None,
                    xlabel='Time [yr]', ylable_dict=None, color_dict=None, ylim_dict=None,
                    ax=None, **plot_kws):

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = {}

        nrow = int(np.ceil(len(vars)/ncol))
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

        i = 0
        i_row, i_col = 0, 0
        for k, v in self.ts.items():
            if 'fig' in locals():
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

            _plot_kws = {
                'linewidth': 2,
            }
            if plot_kws is not None:
                _plot_kws.update(plot_kws)
            

            v.plot(
                ax=ax[k], xlim=xlim, ylim=_ylim_dict[k],
                xlabel=_xlb, ylabel=_ylb_dict[k],
                color=_clr_dict[k], **_plot_kws,
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

    def calc_atm_gm(self, vars=['GMST', 'GMRESTOM', 'GMLWCF', 'GMSWCF'], verbose=False):
        for k, v in self.case_dict.items():
            p_header(f'Processing case: {k} ...')
            v.calc_atm_gm(vars=vars, verbose=verbose)

    def plot_ts(self, lgd_kws=None, lgd_idx=1, **plot_kws):
        _clr_dict = {
            'GMST': None,
            'GMRESTOM': None,
            'GMLWCF': None,
            'GMSWCF': None,
        }
        for k, v in self.case_dict.items():
            if 'fig' not in locals():
                fig, ax = v.plot_ts(color_dict=_clr_dict, label=v.name, **plot_kws)
            else:
                ax = v.plot_ts(ax=ax, color_dict=_clr_dict, label=v.name, **plot_kws)

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
            