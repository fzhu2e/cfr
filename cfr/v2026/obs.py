from copy import deepcopy
import numpy as np
import pandas as pd
import xarray as xr
import x4c
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

from . import psm
from . import utils
class Obs:
    '''Observation database for proxy records.

    Manages a collection of proxy observations stored in a DataFrame, providing
    methods for setup, proxy system modeling, plotting, and conversion to xarray.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: pid, lat, lon, time, value, psm_name, R, type.
    '''
    def __init__(self, df:pd.DataFrame):
        self.df = df.reset_index(drop=True)   # with columns: pid, lat, lon, time, value, psm_name, R, type
        self.df['lon'] = pd.to_numeric(self.df['lon'], errors='coerce')
        self.df['lon'] = (self.df['lon'] + 360) % 360

    def setup(self):
        self.df = self.df.reset_index(drop=True)
        self.nobs = len(self.df)
        self.pids = self.df['pid'].values
        self.to_ds()
        self.records = {}
        for pid in self.pids:
            self.records[pid] = self[pid]

    @property
    def y(self):
        # return self.df['value'].values[..., np.newaxis]
        return self.ds.to_dataframe().values

    @property
    def y_locs(self):
        return self.df[['lat', 'lon']].values

    @property
    def R(self):
        return np.diag(self.df['R'].values)

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, pid:str):
        mask = self.df['pid'] == pid
        row = self.df[mask].iloc[0]
        rec = ProxyRecord(row)
        return rec
    
    def get_dist(self):
        lats = self.df['lat'].values
        lons = self.df['lon'].values
        lat1, lat2 = np.meshgrid(lats, lats)
        lon1, lon2 = np.meshgrid(lons, lons)
        self.dist = utils.gcd(lat1, lon1, lat2, lon2)
        return self.dist

    def get_clim(self, prior, depth_name='z_t', ens_name='ens', mode='each_member', **nearest_kws):
        psm_names = set(self.df['psm_name'])
        clim_vns = list({
            vn for psm_name in psm_names
            for vn in psm.__dict__[psm_name]().clim_vns
            if vn in prior.ds.data_vars
        })

        self.clim = xr.Dataset()
        if mode == 'each_member':
            ens_vals = prior.ds[ens_name].values
            for vn in clim_vns:
                da = prior.ds[vn]
                if depth_name in da.dims: da = da.isel({depth_name: 0})
                member_list = []
                for i, ens_val in enumerate(ens_vals):
                    da_ens = da.sel({ens_name: ens_val})
                    result = da_ens.x.nearest2d(
                        lat=self.df['lat'],
                        lon=self.df['lon'],
                        **nearest_kws,
                    )
                    result = result.expand_dims({ens_name: [ens_val]})  # ensure ens dimension exists
                    member_list.append(result)

                result_all = xr.concat(member_list, dim=ens_name)
                self.clim[vn] = result_all.transpose(..., ens_name)
        else:
            for vn in clim_vns:
                da = prior.ds[vn]
                if depth_name in da.dims: da = da.isel({depth_name: 0})
                self.clim[vn] = da.x.nearest2d(
                    lat=self.df['lat'],
                    lon=self.df['lon'],
                    **nearest_kws,
            ).transpose(..., ens_name)

        self.clim.coords['site'] = self.df['pid']

    def get_pseudo(self, psm_colname='psm_name', **fwd_kws):
        self.df['pseudo'] = object
        for idx, row in self.df.iterrows():
            pid = row['pid']
            psm_name = row[psm_colname]
            if psm_name in fwd_kws:
                _fwd_kws = fwd_kws[psm_name]
            else:
                _fwd_kws = {}

            self.records[pid].psm = mdl = psm.__dict__[psm_name](self.records[pid])
            mdl.clim = self.clim.sel(site=pid)
            if 'month' in mdl.clim.dims:
                mdl.clim = mdl.clim.sel(month=self.records[pid].data.seasonality).mean(dim='month')

            self.df.at[idx, 'pseudo'] = mdl.forward(**_fwd_kws)
            self.df.at[idx, 'Ym'] =self.df.at[idx, 'pseudo'].mean()

    def to_ds(self):
        ds = xr.Dataset()
        for idx, row in self.df.iterrows():
            da = xr.DataArray(
                data=row['value'],
                coords={
                    'time': row['time'],
                },
                name=row['pid'],
            )
            for col in row.index:
                if col not in ['pid', 'time', 'value']:
                    da.attrs[col] = row[col]

            ds[row['pid']] = da
        self.ds = ds
        if 'time' not in self.ds.dims: self.ds['time'] = self.ds.time.expand_dims(dim='time')
        return self.ds

    def plot(self, t_idx=None, return_im=False, da_target=None, reset_target=None, val_colname='value', **kws):
        # useful for pseudoproxy experiments
        if t_idx is None:
            if len(da_target.dims) > 1 and 'time' in da_target.dims:
                da_target = da_target.mean('time')
        else:
            da_target = da_target.isel(time=t_idx)

        if reset_target is not None: da_target = da_target.where(da_target.isnull(), other=reset_target)

        fig_ax_im = da_target.x.plot(return_im=True, **kws)
        if len(fig_ax_im) == 3:
            fig, ax, im = fig_ax_im
        elif len(fig_ax_im) == 2:
            ax, im = fig_ax_im
        else:
            raise ValueError('Unexpected number of return values from x4c plot.')

        lats = self.df['lat'].values
        lons = self.df['lon'].values
        vals = np.array([v for v in self.df[val_colname].values])
        if t_idx is None:
            if len(vals.shape) > 1 and len(vals) > 1:
                vals = vals.mean(axis=-1)
        else:
            vals = vals[:, t_idx]

        _scatter_kws = {
            's': 100,
            'cmap': im.cmap,
            'norm': im.norm,
            'transform': ccrs.PlateCarree(),
            'edgecolor': 'k',
            'zorder': 99,
        }
        ax.scatter(lons, lats, c=vals, **_scatter_kws)

        if 'fig' in locals():
            return (fig, ax, im) if return_im else (fig, ax)
        else:
            return (ax, im) if return_im else ax

    def plotly(self, **kws):
        ''' Plot the database on an interactive map utilizing Plotly
        '''
        _kws = {
            'lat': 'lat',
            'lon': 'lon',
            'color': 'type',
            'hover_name': 'pid',
            'hover_data': self.df.columns.drop(['lat', 'lon', 'type', 'pid', 'time', 'value'], errors='ignore'),
            'projection': 'natural earth',
        }
        _kws.update(kws)
        return px.scatter_geo(self.df, **_kws)


class ProxyRecord:
    '''A single proxy record extracted from the observation database.

    Parameters
    ----------
    data : pd.Series
        A row from the Obs DataFrame containing proxy metadata and values.
    '''
    def __init__(self, data:pd.Series):
        self.data = data.copy()
        if 'time' in data: self.data['time'] = np.array(data['time'])
        if 'value' in data: self.data['value'] = np.array(data['value']) 

        if 'seasonality' in data:
            if isinstance(data['seasonality'], str):
                self.data['seasonality'] = utils.str2list(data['seasonality'])
            elif isinstance(data['seasonality'], list):
                self.data['seasonality'] = data['seasonality']
            else:
                raise ValueError('Wrong seasonality type; should be a string or a list.')


#     def get_clim(self, clim_ds, vns:list=None, verbose=False):
#         if vns is None:
#             vns = clim_ds.data_vars
#         else:
#             vns = [vn for vn in vns if vn in clim_ds.data_vars]

#         self.clim = xr.Dataset()
#         for vn in vns:
#             self.clim[vn] = clim_ds[vn].x.nearest2d(
#             # filled_da = clim_ds[vn].ffill(dim='lon').bfill(dim='lon').ffill(dim='lat').bfill(dim='lat')
#             # self.clim[vn] = filled_da.sel(
#                 lat=self.data.lat,
#                 lon=self.data.lon,
#                 method='nearest',
#             ).sel(month=self.data.seasonality).mean(dim='month')
#             if verbose: utils.p_success(f'>>> ProxyRecord.clim["{vn}"] created')

#         self.clim.attrs['seasonality'] = self.data.seasonality

