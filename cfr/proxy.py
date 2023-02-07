from dataclasses import replace
import glob
from operator import le
import os
from .climate import ClimateField
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
import copy
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from multiprocessing import Pool, cpu_count
from functools import partial
import seaborn as sns
from . import utils, visual
from .utils import (
    p_warning,
    p_header,
    p_success,
    p_fail,
)

def get_ptype(archive_type, proxy_type):
    '''Get proxy type string based on archive and proxy strings

    If not predefined, it will return in the format `archive_type.proxy_type` with blanks replaced with underscores.

    Args:
        archive_type (str): archive string
        proxy_type (str): proxy string

    Returns:
        str: the proxy type string, e.g., coral.d18O

    '''
    ptype_dict = {
        ('tree', 'delta Density'): 'tree.MXD',
        ('tree', 'MXD'): 'tree.MXD',
        ('tree', 'TRW'): 'tree.TRW',
        ('tree', 'ENSO'): 'tree.ENSO',
        ('coral', 'Sr/Ca'): 'coral.SrCa',
        ('coral', 'SrCa_annual'): 'coral.SrCa',
        ('coral', 'Coral Sr/Ca'): 'coral.SrCa',
        ('coral', 'd18O'): 'coral.d18O',
        ('coral', 'd18O_annual'): 'coral.d18O',
        ('coral', 'd18O_sw'): 'coral.d18Osw',
        ('coral', 'd18O_sw_annual'): 'coral.d18Osw',
        ('coral', 'calcification'): 'coral.calc',
        ('coral', 'calcification rate'): 'coral.calc',
        ('sclerosponge', 'd18O'): 'coral.d18O',
        ('sclerosponge', 'Sr/Ca'): 'coral.SrCa',
        ('sclerosponge', 'SrCa'): 'coral.SrCa',
        ('glacier ice', 'melt'): 'ice.melt',
        ('glacier ice', 'd18O'): 'ice.d18O',
        ('glacier ice', 'dD'): 'ice.dD',
        ('glacier ice', 'deterium excess'): 'ice.d-excess',
        ('glacier ice', 'isotope diffusion'): 'ice.isotope_diffusion',
        ('glacier ice', 'hybrid-ice'): 'ice.hybrid',
        ('glacier ice', '15N/40Ar fractionation'): 'ice.15N40Ar',
        ('speleothem', 'd18O'): 'speleothem.d18O',
        ('speleothem', 'dD'): 'speleothem.dD',
        ('marine sediment', 'TEX86'): 'marine.TEX86',
        ('marine sediment', 'Mg/Ca'): 'marine.MgCa',
        ('marine sediment', 'foram Mg/Ca'): 'marine.MgCa',
        ('marine sediment', 'd18O'): 'marine.d18O',
        ('marine sediment', 'dynocist MAT'): 'marine.MAT',
        ('marine sediment', 'alkenone'): 'marine.alkenone',
        ('marine sediment', 'planktonic foraminifera'): 'marine.foram',
        ('marine sediment', 'foraminifera'): 'marine.foram',
        ('marine sediment', 'foram d18O'): 'marine.d18O',
        ('marine sediment', 'diatom'): 'marine.diatom',
        ('marine sediment', 'dinocyst'): 'marine.dinocyst',
        ('marine sediment', 'radiolaria'): 'marine.radiolaria',
        ('marine sediment', 'GDGT'): 'marine.GDGT',
        ('lake sediment', 'varve thickness'): 'lake.varve_thickness',
        ('lake sediment', 'varve property'): 'lake.varve_property',
        ('lake sediment', 'sed accumulation'): 'lake.accumulation',
        ('lake sediment', 'chironomid'): 'lake.chironomid',
        ('lake sediment', 'midge'): 'lake.midge',
        ('lake sediment', 'TEX86'): 'lake.TEX86',
        ('lake sediment', 'BSi'): 'lake.BSi',
        ('lake sediment', 'chrysophyte'): 'lake.chrysophyte',
        ('lake sediment', 'reflectance'): 'lake.reflectance',
        ('lake sediment', 'pollen'): 'lake.pollen',
        ('lake sediment', 'alkenone'): 'lake.alkenone',
        ('lake sediment', 'diatom'): 'lake.diatom',
        ('lake sediment', 'd18O'): 'lake.d18O',
        ('borehole', 'borehole'): 'borehole',
        ('hybrid', 'hybrid'): 'hybrid',
        ('bivalve', 'd18O'): 'bivalve.d18O',
        ('documents', 'Documentary'): 'documents',
        ('documents', 'historic'): 'documents',
        ('peat', 'pollen'): 'peat.pollen',
    }

    ptype_dict_fuzzy = {}
    for k, v in ptype_dict.items():
        archive_str, proxy_str = k
        ptype_dict_fuzzy[(
            archive_str.lower().replace(' ', '').replace('/', '').replace('_', ''),
            proxy_str.lower().replace(' ', '').replace('/', '').replace('_', ''),
        )] = v

    archive_type_fuzzy = archive_type.lower().replace(' ', '').replace('/', '').replace('_', '')
    proxy_type_fuzzy = proxy_type.lower().replace(' ', '').replace('/', '').replace('_', '')

    input_pair_fuzzy = (archive_type_fuzzy, proxy_type_fuzzy)

    if input_pair_fuzzy in ptype_dict_fuzzy:
        ptype = ptype_dict_fuzzy[(archive_type_fuzzy, proxy_type_fuzzy)]
    else:
        ptype = f"{archive_type_fuzzy}.{proxy_type_fuzzy}"

    return ptype

class ProxyRecord:
    ''' The class for a proxy record.

    Args:
        pid (str): the unique proxy ID
        lat (float): latitude
        lon (float): longitude
        time (numpy.array): time axis in unit of year CE 
        value (numpy.array): proxy value axis
        ptype (str): the label of proxy type according to archive and proxy information;
            some examples:

            * 'tree.trw' : TRW
            * 'tree.mxd' : MXD
            * 'coral.d18O' : Coral d18O isotopes
            * 'coral.SrCa' : Coral Sr/Ca ratios
            * 'ice.d18O' : Ice d18O isotopes
        tags (a set of str):
            the tags for the record, to enable tag filtering
    '''
    def __init__(self, pid=None, time=None, value=None, lat=None, lon=None, ptype=None, tags=None,
        value_name=None, value_unit=None, time_name=None, time_unit=None, seasonality=None):
        self.pid = pid
        self.time = time
        self.value = value
        self.lat = lat
        self.lon = np.mod(lon, 360) if lon is not None else None
        self.ptype = ptype
        self.tags = set() if tags is None else tags

        self.dt = np.median(np.diff(time)) if time is not None else None
        self.value_name = 'Proxy Value' if value_name is None else value_name
        self.value_unit = value_unit
        self.time_name = 'Time' if time_name is None else time_name
        self.time_unit = 'yr' if time_unit is None else time_unit
        self.seasonality = seasonality

    def copy(self):
        ''' Make a deepcopy of the object. '''
        return copy.deepcopy(self)

    def center(self, ref_period):
        ''' Centering the proxy timeseries regarding a reference period.

        Args:
            ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
        '''
        new = self.copy()
        ref = self.slice(ref_period)
        new.value -= np.mean(ref.value)
        return new

    def slice(self, timespan):
        ''' Slicing the timeseries with a timespan (tuple or list)

        Args:
            timespan (tuple or list):
                The list of time points for slicing, whose length must be even.
                When there are n time points, the output Series includes n/2 segments.
                For example, if timespan = [a, b], then the sliced output includes one segment [a, b];
                if timespan = [a, b, c, d], then the sliced output includes segment [a, b] and segment [c, d].

        Returns:
            ProxyRecord: The sliced Series object.

        '''
        new = self.copy()
        n_elements = len(timespan)
        if n_elements % 2 == 1:
            raise ValueError('The number of elements in timespan must be even!')

        n_segments = int(n_elements / 2)
        mask = [False for i in range(np.size(self.time))]
        for i in range(n_segments):
            mask |= (self.time >= timespan[i*2]) & (self.time <= timespan[i*2+1])

        new.time = self.time[mask]
        new.value = self.value[mask]
        new.dt = np.median(np.diff(new.time))
        return new

    def concat(self, rec_list):
        new = self.copy()
        ts_list = [pd.Series(index=self.time, data=self.value)]
        for rec in rec_list:
            ts_list.append(pd.Series(index=rec.time, data=rec.value))

        ts_concat = pd.concat(ts_list)
        ts_concat = ts_concat.sort_index()
        new.time = ts_concat.index.to_numpy()
        new.value = ts_concat.values
        new.dt = np.median(np.diff(new.time))
        return new

    def to_nc(self, path, verbose=True, **kwargs):
        ''' Convert the record to a netCDF file.

        Args:
            path (str): the path to save the file.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        da = self.to_da()
        da.to_netcdf(path=path, **kwargs)
        if verbose: utils.p_success(f'ProxyRecord saved to: {path}')

    def load_nc(self, path, **kwargs):
        da = xr.open_dataarray(path, **kwargs)
        new = self.from_da(da)
        return new

    def to_da(self):
        ''' Convert to Xarray.DataArray for computation purposes
        '''
        dates = utils.year_float2datetime(self.time)
        da = xr.DataArray(
            self.value, dims=['time'], coords={'time': dates}, name=self.pid,
            attrs={
                'lat': self.lat,
                'lon': self.lon,
                'ptype': self.ptype,
                'dt': self.dt,
                'time_name': self.time_name,
                'time_unit': self.time_unit,
                'value_name': self.value_name,
                'value_unit': self.value_unit,
            }
        )
        return da

    def from_da(self, da):
        ''' Get the time and value axis from the given Xarray.DataArray
        '''
        new = ProxyRecord()
        if 'time' in da.dims:
            new.time = utils.datetime2year_float(da.time.values)
        elif 'year' in da.dims:
            new.time = da.year.values

        new.value = da.values
        new.time, new.value = utils.clean_ts(new.time, new.value)

        new.pid = da.name
        new.lat = da.attrs['lat'] if 'lat' in da.attrs else None
        new.lon = da.attrs['lon'] if 'lon' in da.attrs else None
        new.ptype = da.attrs['ptype'] if 'ptype' in da.attrs else None

        new.dt = np.median(np.diff(new.time))
        new.value_name = da.attrs['value_name'] if 'value_name' in da.attrs else None
        new.value_unit = da.attrs['value_unit'] if 'value_name' in da.attrs else None
        new.time_name = da.attrs['time_name'] if 'time_name' in da.attrs else None
        new.time_unit = da.attrs['time_unit'] if 'time_name' in da.attrs else None
        return new

    def annualize(self, months=list(range(1, 13)), verbose=False):
        new = self.copy()
        try:
            new.time, new.value = utils.annualize(self.time, self.value, months=months)
        except:
            new.time, new.value = utils.annualize(self.time, self.value, months=list(range(1, 13)))
            if verbose: p_warning(f'Record {self.pid} cannot be annualized with months {months}. Use calendar year instead.')

        new.time, new.value = utils.clean_ts(new.time, new.value)
        return new
            

    def standardize(self):
        new = self.copy()
        if self.value.std() == 0:
            new.value = np.zeros(np.size(self.value))
        else:
            new.value = (self.value - self.value.mean()) / self.value.std()
        return new

    def __getitem__(self, key):
        ''' This makes the object subscriptable. '''
        new = self.copy()
        if type(key) is int or type(key) is list:
            new.value = new.value[key]
            new.time = new.time[key]
        elif type(key) is slice: 
            if type(key.start) is int:
                new.value = new.value[key]
                new.time = new.time[key]
            elif type(key.start) is str:
                time_mask = (new.time>=int(key.start)) & (new.time<=int(key.stop))
                new.value = new.value[time_mask]
                new.time = new.time[time_mask]
                if key.step is not None:
                    new.value = new.value[::int(key.step)]
                    new.time = new.time[::int(key.step)]

        new.dt = np.median(np.diff(new.time))
        return new

    def __add__(self, records):
        ''' Add a list of records into a database
        '''
        new = ProxyDatabase()
        new.records[self.pid] = self.copy()
        if isinstance(records, ProxyRecord):
            # if only one record
            records = [records]

        if isinstance(records, ProxyDatabase):
            # if a database
            records = [records.records[pid] for pid in records.records.keys()]

        for record in records:
            new.records[record.pid] = record

        new.refresh()
        return new

    def __sub__(self, ref):
        ''' Substract the reference record
        '''
        new = self.copy()
        new.value = self.value - ref.value
        return new

    def del_clim(self, verbose=False):
        if hasattr(self, 'clim'): del self.clim
        if verbose: utils.p_success(f'ProxyRecord.clim deleted for {self.pid}.')

    def get_clim(self, fields, tag=None, verbose=False, search_dist=5, load=True, **kwargs):
        ''' Get the nearest climate from cliamte fields

        Args:
            fields (list of cfr.climate.ClimateField): the climate fields
            search_dist (float): the farest distance to search for climate data in degree
        '''
        if isinstance(fields, ClimateField):
            fields = [fields]

        _kwargs = {'method': 'nearest'}
        _kwargs.update(kwargs)
        
        for field in fields:
            name = field.da.name
            if tag is not None:
                name = f'{tag}.{name}'

            nda = field.da.sel(lat=self.lat, lon=self.lon, **_kwargs)
            if np.all(np.isnan(nda.values)):
                for i in range(1, search_dist+1):
                    p_header(f'{self.pid} >>> Nearest climate is NaN. Searching around within distance of {i} deg ...')
                    da_cond = field.da.where(np.abs(field.da.lat - self.lat)<= i).where(
                        np.abs(field.da.lon - self.lon) <= i
                    )
                    nda = utils.geo_mean(da_cond)
                    nda.coords['lat'] = self.lat
                    nda.coords['lon'] = self.lon
                    if not np.all(np.isnan(nda.values)):
                        p_success(f'{self.pid} >>> Found nearest climate within distance of {i} deg.')
                        break

            if not hasattr(self, 'clim'):
                self.clim = {}

            if 'time' in field.da.dims:
                time_name = 'time'
            elif 'year' in field.da.dims:
                time_name = 'year'
            else:
                raise ValueError(f'Incorrect name for the time dimension in {tag}.')

            self.clim[name] = ClimateField().from_da(nda, time_name=time_name)
            self.clim[name].time = field.time
            if load: self.clim[name].da.load()
            if verbose: utils.p_success(f'{self.pid} >>> ProxyRecord.clim["{name}"] created.')

    def del_pseudo(self, verbose=False):
        if hasattr(self, 'pseudo'): del self.pseudo
        if verbose: utils.p_success(f'ProxyRecord.pseudo deleted for {self.pid}.')

    def get_pseudo(self, psm, model_vars=None,
                   add_noise=False, noise='white', SNR=10, seed=None,
                   match_mean=False, match_var=False, verbose=False,
                   calib_kws=None, forward_kws=None, colored_noise_kws=None):
        calib_kws = {} if calib_kws is None else calib_kws
        forward_kws = {} if forward_kws is None else forward_kws

        if not hasattr(self, 'clim'):
            for var in model_vars:
                self.get_clim(var, tag='model', verbose=verbose)

        mdl = psm(self)
        if hasattr(mdl, 'calibrate'):
            mdl.calibrate(**calib_kws)

        self.pseudo = mdl.forward(**forward_kws)
        if verbose: utils.p_success(f'>>> ProxyRecord.pseudo created.')

        if add_noise:
            sigma = np.nanstd(self.pseudo.value) / SNR
            if noise == 'white':
                rng = np.random.default_rng(seed)
                noise = rng.normal(0, sigma, np.size(self.pseudo.value))
            elif noise == 'colored':
                colored_noise_kws = {} if colored_noise_kws is None else colored_noise_kws
                _colored_noise_kws = {'seed': seed, 'alpha': 1, 't': self.pseudo.time}
                _colored_noise_kws.update(colored_noise_kws)
                noise = utils.colored_noise(**_colored_noise_kws)

            self.pseudo.value += noise / np.std(noise) * sigma
            if verbose: utils.p_success(f'>>> ProxyRecord.pseudo added with {noise} noise (SNR={SNR}).')

        if match_var or match_mean:
            proxy_time_min = np.min(self.time)
            proxy_time_max = np.max(self.time)
            pseudo_time_min = np.min(self.pseudo.time)
            pseudo_time_max = np.max(self.pseudo.time)
            time_min = np.max([proxy_time_min, pseudo_time_min])
            time_max = np.min([proxy_time_max, pseudo_time_max])
            mask_proxy = (self.time>=time_min)&(self.time<=time_max)
            mask_pseudo = (self.pseudo.time>=time_min)&(self.pseudo.time<=time_max)

        value = self.pseudo.value

        if match_var:
            value = value / np.nanstd(value[mask_pseudo]) * np.nanstd(self.value[mask_proxy])
            if verbose: utils.p_success(f'>>> Variance matched.')

        if match_mean:
            value = value - np.nanmean(value[mask_pseudo]) + np.nanmean(self.value[mask_proxy])
            if verbose: utils.p_success(f'>>> Mean matched.')

        self.pseudo.value = value



    def plotly(self, **kwargs):
        time_lb = visual.make_lb(self.time_name, self.time_unit)
        value_lb = visual.make_lb(self.value_name, self.value_unit)

        _kwargs = {'markers': 'o', 'template': 'seaborn'}
        _kwargs.update(kwargs)
        fig = px.line(
            x=self.time, y=self.value,
            labels={'x': time_lb, 'y': value_lb},
            **_kwargs,
        )

        return fig

    def plot(self, figsize=[12, 4], legend=False, ms=200, stock_img=True, edge_clr='w',
        wspace=0.1, hspace=0.1, plot_map=True, **kwargs):
        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = visual.STYLE.colors_dict[self.ptype]

        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(1, 3)
        gs.update(wspace=wspace, hspace=hspace)
        ax = {}

        # plot timeseries
        ax['ts'] = plt.subplot(gs[0, :2])

        # _kwargs = {'marker': 'o'}
        _kwargs = {}
        _kwargs.update(kwargs)
        ax['ts'].plot(self.time, self.value, **_kwargs)

        time_lb = visual.make_lb(self.time_name, self.time_unit)
        value_lb = visual.make_lb(self.value_name, self.value_unit)
        ax['ts'].set_xlabel(time_lb)
        ax['ts'].set_ylabel(value_lb)

        title = f'{self.pid} ({self.ptype}) @ (lat:{self.lat:.2f}, lon:{self.lon:.2f})'
        if self.seasonality is not None:
            title += f'\nSeasonality: {self.seasonality}'
        ax['ts'].set_title(title)

        if legend:
            ax['ts'].legend()

        # plot map
        if plot_map:
            ax['map'] = plt.subplot(gs[0, 2], projection=ccrs.Orthographic(central_longitude=self.lon, central_latitude=self.lat))
            ax['map'].set_global()
            if stock_img:
                ax['map'].stock_img()

            transform=ccrs.PlateCarree()
            ax['map'].scatter(
                self.lon, self.lat, marker=visual.STYLE.markers_dict[self.ptype],
                s=ms, c=kwargs['color'], edgecolor=edge_clr, transform=transform,
            )

        return fig, ax

    def dashboard(self, figsize=[10, 8], ms=200, stock_img=True, edge_clr='w',
        wspace=0.1, hspace=0.3, spec_method='wwz', pseudo_clr=None, **kwargs):
        ''' Plot a dashboard of the proxy/pseudoproxy.
        '''

        if not hasattr(self, 'pseudo'):
            raise ValueError('Need to get the pseudoproxy data.')

        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = visual.STYLE.colors_dict[self.ptype]

        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(2, 3)
        gs.update(wspace=wspace, hspace=hspace)
        ax = {}

        # plot proxy/pseudoproxy timeseries
        ax['ts'] = plt.subplot(gs[0, :2])

        _kwargs = {'label': 'real', 'zorder': 3, 'alpha': 0.7}
        _kwargs.update(kwargs)
        ax['ts'].plot(self.time, self.value, **_kwargs)

        time_lb = visual.make_lb(self.time_name, self.time_unit)
        value_lb = visual.make_lb(self.value_name, self.value_unit)

        ax['ts'].set_ylabel(value_lb)
        ax['ts'].set_xlabel(time_lb)
        ax['ts'].plot(
            self.pseudo.time, self.pseudo.value,
            color=pseudo_clr, label='pseudo')
        ax['ts'].legend(loc='upper left', ncol=2)
        title = f'{self.pid} ({self.ptype})'
        if self.seasonality is not None:
            title += f'\nSeasonality: {self.seasonality}'
        ax['ts'].set_title(title)

        # plot probability density
        ax['pd'] = plt.subplot(gs[0, 2], sharey=ax['ts'])
        df_real = pd.DataFrame()
        df_real['value'] = self.value
        df_real['year'] = [int(t) for t in self.time]
        df_real['dataset'] = 'real'

        df_pseudo = pd.DataFrame()
        df_pseudo['value'] = self.pseudo.value
        df_pseudo['year'] = [int(t) for t in self.pseudo.time]
        df_pseudo['dataset'] = 'pseudo'

        df = pd.concat([df_real, df_pseudo])
        df['all'] = ''
        sns.violinplot(
            data=df, x='all', y='value', ax=ax['pd'],
            split=True, hue='dataset', inner='quart',
            palette={'real': ax['ts'].lines[0].get_color(), 'pseudo': ax['ts'].lines[1].get_color()},
        )
        ax['pd'].set_ylabel('')
        ax['pd'].set_xlabel('')
        ax['pd'].tick_params(axis='y', colors='none')
        ax['pd'].yaxis.set_ticks_position('none') 
        ax['pd'].tick_params(axis='x', colors='none')
        ax['pd'].xaxis.set_ticks_position('none') 
        ax['pd'].set_title('Probability Distribution')
        ax['pd'].legend_ = None
            
        # plot map
        ax['map'] = plt.subplot(gs[1, 2], projection=ccrs.Orthographic(central_longitude=self.lon, central_latitude=self.lat))
        ax['map'].set_global()
        if stock_img:
            ax['map'].stock_img()

        transform=ccrs.PlateCarree()
        ax['map'].scatter(
            self.lon, self.lat, marker=visual.STYLE.markers_dict[self.ptype],
            s=ms, c=visual.STYLE.colors_dict[self.ptype], edgecolor=edge_clr, transform=transform,
        )
        ax['map'].set_title(f'lat: {self.lat:.2f}, lon: {self.lon:.2f}')

        # plot spectral analysis
        try:
            import pyleoclim as pyleo
        except:
            raise ImportError('Need to install pyleoclim: `pip install pyleoclim`.')

        ax['psd'] = plt.subplot(gs[1, :2])

        ts, psd = {}, {}
        ts['real'] = pyleo.Series(time=self.time, value=self.value)
        psd['real'] = ts['real'].spectral(method=spec_method)
        psd['real'].plot(ax=ax['psd'], **_kwargs)

        ts['pseudo'] = pyleo.Series(time=self.pseudo.time, value=self.pseudo.value)
        psd['pseudo'] = ts['pseudo'].slice([self.time.min(), self.time.max()]).spectral(method=spec_method)
        psd['pseudo'].plot(ax=ax['psd'], color=pseudo_clr, label='pseudo')

        ax['psd'].legend_ = None
        ax['psd'].text(
            x=0.95, y=0.97, s=f'Timespan: {int(self.time.min())}-{int(self.time.max())}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax['psd'].transAxes,
        )
        ax['psd'].set_xlabel('Period [yrs]')

        return fig, ax

    def dashboard_clim(self, clim_units=None, clim_colors=None, figsize=[14, 8], scaled_pr=False, ms=200, stock_img=True, edge_clr='w',
        wspace=0.3, hspace=0.5, spec_method='wwz', **kwargs):
        ''' Plot a dashboard of the proxy/pseudoproxy along with the climate signal.

        Args:
            clim_units (dict, optional): the dictionary of units for climate signals. Defaults to None.
            clim_colors (dict, optional): the dictionary of colors for climate signals. Defaults to None.
        '''

        if not hasattr(self, 'clim'):
            raise ValueError('Need to get the nearest climate data.')

        if not hasattr(self, 'pseudo'):
            raise ValueError('Need to get the pseudoproxy data.')

        if 'color' not in kwargs and 'c' not in kwargs:
            kwargs['color'] = visual.STYLE.colors_dict[self.ptype]

        fig = plt.figure(figsize=figsize)

        nclim = len(clim_units)
        gs = gridspec.GridSpec(2*nclim, 3)
        gs.update(wspace=wspace, hspace=hspace)
        ax = {}

        # plot proxy/pseudoproxy timeseries
        ax['ts'] = plt.subplot(gs[:nclim, :2])

        _kwargs = {'label': 'real', 'zorder': 3}
        _kwargs.update(kwargs)
        ax['ts'].plot(self.time, self.value, **_kwargs)

        time_lb = visual.make_lb(self.time_name, self.time_unit)
        value_lb = visual.make_lb(self.value_name, self.value_unit)

        ax['ts'].set_ylabel(value_lb)
        ax['ts'].plot(self.pseudo.time, self.pseudo.value, label='pseudo')
        ax['ts'].legend(loc='upper left', ncol=2, bbox_to_anchor=(0, 1.0))
        title = f'{self.pid} ({self.ptype}) @ (lat:{self.lat:.2f}, lon:{self.lon:.2f})'
        if self.seasonality is not None:
            title += f'\nSeasonality: {self.seasonality}'
        ax['ts'].set_title(title)
            
        # plot climate signals
        i = 0
        for k, v in self.clim.items():
            if k in clim_units:
                ax[k] = plt.subplot(gs[nclim+i, :2], sharex=ax['ts'])
                ax[k].plot(v.time, v.da.values, color=clim_colors[k], label=k)
                # ax[k].set_title(k)
                vn = k.split('_')[-1] if '_' in k else k
                ylb = f'{vn} [{clim_units[k]}]'
                if len(ylb)>=10:
                    ylb = f'{vn}\n[{clim_units[k]}]'
                    
                ax[k].set_ylabel(ylb)
                if vn == 'pr' and scaled_pr:
                    ax[k].set_title(r'1e-5', loc='left')

                ax[k].legend(loc='upper left', bbox_to_anchor=(0, 1.2))
                ax[k].set_xlim([self.time.min(), self.time.max()])
                i += 1
        ax[k].set_xlabel(time_lb)


        # plot map
        ax['map'] = plt.subplot(gs[:nclim, 2], projection=ccrs.Orthographic(central_longitude=self.lon, central_latitude=self.lat))
        ax['map'].set_global()
        if stock_img:
            ax['map'].stock_img()

        transform=ccrs.PlateCarree()
        ax['map'].scatter(
            self.lon, self.lat, marker=visual.STYLE.markers_dict[self.ptype],
            s=ms, c=kwargs['color'], edgecolor=edge_clr, transform=transform,
        )

        # plot spectral analysis
        try:
            import pyleoclim as pyleo
        except:
            raise ImportError('Need to install pyleoclim: `pip install pyleoclim`.')

        ax['psd'] = plt.subplot(gs[nclim:, 2])

        ts, psd = {}, {}
        ts['real'] = pyleo.Series(time=self.time, value=self.value)
        psd['real'] = ts['real'].spectral(method=spec_method)
        psd['real'].plot(ax=ax['psd'], **_kwargs)

        ts['pseudo'] = pyleo.Series(time=self.pseudo.time, value=self.pseudo.value)
        psd['pseudo'] = ts['pseudo'].slice([self.time.min(), self.time.max()]).spectral(method=spec_method)
        psd['pseudo'].plot(ax=ax['psd'], label='pseudo')

        for k, v in self.clim.items():
            if k in clim_units:
                ts[k] = pyleo.Series(time=v.time, value=v.da.values)
                psd[k] = ts[k].slice([self.time.min(), self.time.max()]).spectral(method=spec_method)
                psd[k].plot(ax=ax['psd'], label=k, color=clim_colors[k])

        # ax['psd'].legend(loc='upper left', ncol=2, bbox_to_anchor=(0, 1.2))
        ax['psd'].legend_ = None
        ax['psd'].set_title(f'Timespan: {int(self.time.min())}-{int(self.time.max())}')

        return fig, ax

    def plot_dups(self, figsize=[12, 4], legend=False, ms=200, stock_img=True, edge_clr='w',
        wspace=0.1, hspace=0.1, plot_map=True, lgd_kws=None, **kwargs):
        lgd_kws = {} if lgd_kws is None else lgd_kws

        fig, ax = self.plot(
            figsize=figsize,
            legend=legend,
            ms=ms,
            stock_img=stock_img,
            edge_clr=edge_clr,
            wspace=wspace,
            hspace=hspace,
            plot_map=plot_map,
            label=f'{self.pid} ({self.ptype})',
            **kwargs
        )
        for pobj in self.dups:
            ax['ts'].plot(pobj.time, pobj.value, label=f'{pobj.pid} ({pobj.ptype})')
            transform=ccrs.PlateCarree()
            ax['map'].scatter(
                pobj.lon, pobj.lat, marker=visual.STYLE.markers_dict[pobj.ptype],
                s=ms, edgecolor=edge_clr, transform=transform,
            )

        ax['ts'].set_ylabel('Proxy values')
        _lgd_kws = {'ncol': 2}
        _lgd_kws.update(lgd_kws)
        ax['ts'].legend(**_lgd_kws)

        return fig, ax


    def plot_compare(self, ref, label=None, title=None, ref_label=None, ref_color=None, ref_zorder=2,
                    figsize=[12, 4], legend=False, ms=200, stock_img=True, edge_clr='w',
                    wspace=0.1, hspace=0.1, plot_map=True, lgd_kws=None, **kwargs):
        lgd_kws = {} if lgd_kws is None else lgd_kws

        fig, ax = self.plot(
            figsize=figsize,
            legend=legend,
            ms=ms,
            stock_img=stock_img,
            edge_clr=edge_clr,
            wspace=wspace,
            hspace=hspace,
            plot_map=plot_map,
            label=label,
            **kwargs
        )

        if title is not None:
            ax['ts'].set_title(title)

        ax['ts'].plot(ref.time, ref.value, color=ref_color, label=ref_label, zorder=ref_zorder)
        # transform=ccrs.PlateCarree()
        # ax['map'].scatter(
        #     ref.lon, ref.lat, marker=visual.STYLE.markers_dict[ref.ptype],
        #     s=ms, edgecolor=edge_clr, transform=transform,
        #     zorder=ref_zorder, 
        # )

        if label is not None or ref_label is not None:
            _lgd_kws = {'ncol': 2}
            _lgd_kws.update(lgd_kws)
            ax['ts'].legend(**_lgd_kws)

        return fig, ax


class ProxyDatabase:
    ''' The class for a proxy database.

    Args:
        records (dict): a dict of the :py:mod:`cfr.proxy.ProxyRecord` objects with proxy ID as keys
        source (str): a path to the original source file

    '''
    def __init__(self, records=None, source=None):
        self.records = {} if records is None else records
        self.source = source
        if records is not None:
            self.refresh()

    def __getitem__(self, key):
        ''' This makes the object subscriptable. '''
        new = self.copy()
        if type(key) is str:
            new = new.records[key]
        else:
            key = new.pids[key]
            new = new.filter(by='pid', keys=key, mode='exact')
            if len(new.records) == 1:
                pid = new.pids[0]
                new = new.records[pid]

        return new

    def copy(self):
        ''' Make a deepcopy of the object. '''
        return copy.deepcopy(self)

    def center(self, ref_period):
        ''' Center the proxy timeseries against a reference time period.

        Args:
            ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
        '''
        new = self.copy()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Centering each of the ProxyRecord'):
            ref = pobj.slice(ref_period)
            if np.size(ref.time) == 0:
                new -= pobj
            else:
                new.records[pid].value -= np.mean(ref.value)

        return new

    def refresh(self):
        ''' Refresh a bunch of attributes. '''
        self.nrec = len(self.records)
        self.pids = [pobj.pid for pid, pobj in self.records.items()]
        self.lats = [pobj.lat for pid, pobj in self.records.items()]
        self.lons = [pobj.lon for pid, pobj in self.records.items()]
        self.type_list = [pobj.ptype for pid, pobj in self.records.items()]
        self.type_dict = {}
        for t in self.type_list:
            if t not in self.type_dict:
                self.type_dict[t] = 1
            else:
                self.type_dict[t] += 1

    def from_df(self, df, pid_column='paleoData_pages2kID', lat_column='geo_meanLat', lon_column='geo_meanLon',
                time_column='year', value_column='paleoData_values', proxy_type_column='paleoData_proxy', archive_type_column='archiveType',
                value_name_column='paleoData_variableName', value_unit_column='paleoData_units',
                verbose=False):
        ''' Load database from a Pandas DataFrame

        Args:
            df (pandas.DataFrame): a Pandas DataFrame include at least lat, lon, time, value, proxy_type
            ptype_psm (dict): a mapping from ptype to psm
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        new = self.copy()
        if not isinstance(df, pd.DataFrame):
            err_msg = 'the input df should be a Pandas DataFrame.'
            if verbose:
                utils.p_fail(f'ProxyDatabase.from_df() >>> {err_msg}')
            raise TypeError(err_msg)

        records = OrderedDict()

        for idx, row in df.iterrows():
            proxy_type = row[proxy_type_column]
            archive_type = row[archive_type_column]
            ptype = get_ptype(archive_type, proxy_type)
            pid = row[pid_column]
            lat = row[lat_column]
            lon = np.mod(row[lon_column], 360)
            time = np.array(row[time_column])
            value = np.array(row[value_column])
            time, value = utils.clean_ts(time, value)
            value_name=row[value_name_column] if value_name_column in row else None
            value_unit=row[value_unit_column] if value_name_column in row else None

            record = ProxyRecord(
                pid=pid, lat=lat, lon=lon,
                time=time, value=value, ptype=ptype,
                value_name=value_name, value_unit=value_unit,
            )
            records[pid] = record

        # update the attributes
        new.records = records
        new.refresh()
        return new

    def __add__(self, records):
        ''' Add a list of records into the database.

        Args:
            records (list): a list of :py:mod:`cfr.proxy.ProxyRecord`
                can also be a single :py:mod:`cfr.proxy.ProxyRecord` or :py:mod:`cfr.proxy.ProxyDatabase`

        '''
        new = self.copy()
        if isinstance(records, ProxyRecord):
            # if only one record
            records = [records]

        if isinstance(records, ProxyDatabase):
            # if a database
            pdb = records
            records = [pdb.records[pid] for pid in pdb.records.keys()]

        for record in records:
            new.records[record.pid] = record

        new.refresh()
        return new

    def __sub__(self, records):
        ''' Subtract a list of records from a database.

        Args:
            records (list): a list of :py:mod:`cfr.proxy.ProxyRecord`
                can also be a single :py:mod:`cfr.proxy.ProxyRecord` or :py:mod:`cfr.proxy.ProxyDatabase`

        '''
        new = self.copy()
        if isinstance(records, ProxyRecord):
            # if only one record
            records = [records]

        if isinstance(records, ProxyDatabase):
            # if a database
            pdb = records
            records = [pdb.records[pid] for pid in pdb.records.keys()]

        for record in records:
            try:
                del new.records[record.pid]
            except:
                utils.p_warning(f'>>> Subtracting {record.pid} failed.')

        new.refresh()
        return new

    def filter(self, by, keys, mode='fuzzy'):
        ''' Filter the proxy database according to given ptype list.

        Args:
            by (str): filter by a keyword {'ptype', 'pid', 'lat', 'lon', 'loc', 'tag'}
            keys (set): a set of keywords

                * For `by = 'ptype' or 'pid'`, keys take a fuzzy match
                * For `by = 'lat' or 'lon'`, keys = (lat_min, lat_max) or (lon_min, lon_max)
                * For `by = 'loc-squre'`, keys = (lat_min, lat_max, lon_min, lon_max)
                * For `by = 'loc-circle'`, keys = (center_lat, center_lon, distance)

            mode (str): 'fuzzy' or 'exact' search when `by = 'ptype' or 'pid'`

        '''
        if isinstance(keys, str): keys = [keys]

        new_db = ProxyDatabase()
        pobjs = []
        for pid, pobj in self.records.items():
            target = {
                'ptype': pobj.ptype,
                'pid': pobj.pid,
                'lat': pobj.lat,
                'lon': pobj.lon,
                'loc-square': (pobj.lat, pobj.lon),
                'loc-circle': (pobj.lat, pobj.lon),
                'tag': pobj.tags,
            }
            if by in ['ptype', 'pid']:
                for key in keys:
                    if mode == 'fuzzy':
                        if key in target[by]:
                            pobjs.append(pobj)
                    elif mode == 'exact':
                        if key == target[by]:
                            pobjs.append(pobj)
            elif by in ['lat', 'lon']:
                if target[by] >= keys[0] and target[by] <= keys[-1]:
                    pobjs.append(pobj)
            elif by == 'loc-square':
                plat, plon = target[by]
                if plat >= keys[0] and plat <= keys[1] and plon >= keys[2] and plon <= keys[3]:
                    pobjs.append(pobj)
            elif by == 'loc-circle':
                plat, plon = target[by]
                d = utils.gcd(plat, plon, keys[0], keys[1])
                if d <= keys[2]:
                    pobjs.append(pobj)
            elif by == 'tag':
                if set(keys) <= target[by]:
                    pobjs.append(pobj)
            
        new_db += pobjs
        new_db.refresh()
        return new_db

    def nrec_tags(self, keys):
        ''' Check the number of tagged records.

        Args:
            keys (list): list of tag strings

        '''
        nrec = 0
        if isinstance(keys, str): keys = [keys]
        keys = set(keys)
        for pid, pobj in self.records.items():
            if keys <= pobj.tags:
                nrec += 1

        return nrec

    def find_duplicates(self, r_thresh=0.9, time_period=[0, 2000]):
        df_proxy = pd.DataFrame(index=np.arange(time_period[0], time_period[1]+1))
        for pid, pobj in self.records.items():
            series = pd.Series(index=pobj.time, data=pobj.value, name=pid)
            df_proxy = pd.concat([df_proxy, series], axis=1)

        mask = (df_proxy.index>=time_period[0]) & (df_proxy.index<=time_period[-1])
        df_proxy = df_proxy[mask]
        pid_list = df_proxy.columns.values
        R = np.triu(np.corrcoef(df_proxy.values.T), k=1) 
        R[R==0] = np.nan
        di, dj = np.where(R >= r_thresh)
        dup_pids = []
        for i, j in zip(di, dj):
            pid_i = pid_list[i]
            pid_j = pid_list[j]
            if not hasattr(self.records[pid_i], 'dups'):
                self.records[pid_i].dups = [self.records[pid_j]]
                self.records[pid_i].dup_pids = [pid_j]
            else:
                self.records[pid_i].dups.append(self.records[pid_j])
                self.records[pid_i].dup_pids.append(pid_j)

            if not hasattr(self.records[pid_j], 'dups'):
                self.records[pid_j].dups = [self.records[pid_i]]
                self.records[pid_j].dup_pids = [pid_i]
            else:
                self.records[pid_j].dups.append(self.records[pid_i])
                self.records[pid_j].dup_pids.append(pid_i)

            dup_pids.append(pid_i)
            dup_pids.append(pid_j)

        dup_pids = set(dup_pids)
        pdb_dups = self.filter(by='pid', keys=dup_pids, mode='exact')
        pdb_dups.groups = []
        for pid, pobj in pdb_dups.records.items():
            s_tmp = set([pid, *pobj.dup_pids])
            flag = True
            for g in pdb_dups.groups:
                if len(s_tmp & set(g)) > 0:  # found in an existing group
                    g |= s_tmp  # merge into that group
                    flag = False  # will not create a new group
                    break

            if flag:
                pdb_dups.groups.append(s_tmp)

        pdb_dups.dup_args = {'r_thresh': r_thresh, 'time_period': time_period}

        p_header('>>> Groups of duplicates:')
        for i, g in enumerate(pdb_dups.groups):
            print(i+1, g)

        p_header('>>> Hint for the next step:')
        p_header('Use the method `ProxyDatabase.squeeze_dups(pids_to_keep=pid_list)` to keep only one record from each group.')

        return pdb_dups

    def squeeze_dups(self, pids_to_keep=None):
        if pids_to_keep is None:
            p_warning('>>> Note: since `pids_to_keep` is not specified, the first of each group of the duplicates is picked.')
            pids_to_keep = []
            for g in self.groups:
                pids_to_keep.append(list(g)[0])
        
        pids_to_keep = set(pids_to_keep)
        p_header(f'>>> pids to keep (n={len(pids_to_keep)}):')
        print(pids_to_keep)
        pdb_to_keep = self.filter(by='pid', keys=pids_to_keep)
        for pid, pobj in pdb_to_keep.records.items():
            if hasattr(pobj, 'dups'): del(pobj.dups)
            if hasattr(pobj, 'dup_pids'):del(pobj.dup_pids)
        return pdb_to_keep
        

    def plot(self, **kws):
        '''Visualize the proxy database.

        See :py:func:`cfr.visual.plot_proxies()` for more information.
        '''

        time_list = []
        for pid, pobj in self.records.items():
            time_list.append(pobj.time)

        df = pd.DataFrame({'lat': self.lats, 'lon': self.lons, 'type': self.type_list, 'time': time_list, 'pid': self.pids})

        if 'return_gs' in kws and kws['return_gs'] is True:
            fig, ax, gs = visual.plot_proxies(df, **kws)
            return fig, ax, gs
        else:
            fig, ax = visual.plot_proxies(df, **kws)
            return fig, ax


    def plotly(self, **kwargs):
        ''' Plot the database on an interactive map utilizing Plotly
        '''
        df = self.to_df()
        fig = px.scatter_geo(
            df, lat='lat', lon='lon',
            color='ptype',
            hover_name='pid',
            projection='natural earth',
            **kwargs,
        )

        return fig

    def make_composite(self, obs=None, obs_nc_path=None, vn='tas', lat_name=None, lon_name=None, bin_width=10, n_bootstraps=1000, qs=(0.025, 0.975), stat_func=np.nanmean, anom_period=[1951, 1980]):
        ''' Make composites of the records in the proxy database.'''
        if obs is None and obs_nc_path is not None:
            obs = ClimateField().load_nc(obs_nc_path, vn=vn, lat_name=lat_name, lon_name=lon_name)

        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Analyzing ProxyRecord'):
            pobj_stdd = pobj.standardize()
            proxy_time, proxy_value, _ = utils.smooth_ts(pobj_stdd.time, pobj_stdd.value, bin_width=bin_width)

            ts_proxy = pd.Series(index=proxy_time, data=proxy_value, name=pid)
            if 'df_proxy' not in locals():
                df_proxy = ts_proxy.to_frame()
            else:
                df_proxy = pd.merge(df_proxy, ts_proxy, left_index=True, right_index=True, how='outer')

            if obs is not None:
                pobj.get_clim(obs, tag='obs')
                pobj.clim[f'obs.{vn}'].center(ref_period=anom_period)
                obs_time, obs_value, _ = utils.smooth_ts(pobj.clim[f'obs.{vn}'].time, pobj.clim[f'obs.{vn}'].da.values, bin_width=bin_width)
                ts_obs = pd.Series(index=obs_time, data=obs_value, name=pid)

                if 'df_obs' not in locals():
                    df_obs = ts_obs.to_frame()
                else:
                    df_obs = pd.merge(df_obs, ts_obs, left_index=True, right_index=True, how='outer')
            else:
                df_obs = None

        proxy_comp = df_proxy.apply(stat_func, axis=1)

        if obs is not None:
            obs_comp = df_obs.apply(stat_func, axis=1)
            ols_model = utils.ols_ts(proxy_comp.index, proxy_comp.values, obs_comp.index, obs_comp.values)
            results = ols_model.fit()
            intercept = results.params[0]
            slope = results.params[1]
            r2 = results.rsquared
        else:
            obs_comp = None
            r2 = None
            slope = 1 / np.nanstd(proxy_comp.values)
            intercept = - np.nanmean(proxy_comp.values) * slope

        proxy_comp_scaled = proxy_comp.values*slope + intercept

        proxy_bin_vector = utils.make_bin_vector(proxy_comp.index, bin_width=bin_width)
        proxy_comp_time, proxy_comp_value = utils.bin_ts(proxy_comp.index, proxy_comp_scaled, bin_vector=proxy_bin_vector, smoothed=True)

        if obs is not None:
            obs_bin_vector = utils.make_bin_vector(obs_comp.index, bin_width=bin_width)
            obs_comp_time, obs_comp_value = utils.bin_ts(obs_comp.index, obs_comp.values, bin_vector=obs_bin_vector, smoothed=True)
        else:
            obs_comp_time = None
            obs_comp_value = None

        proxy_sq_low = np.empty_like(proxy_comp.index)
        proxy_sq_high = np.empty_like(proxy_comp.index)
        proxy_num = np.empty_like(proxy_comp.index)
        idx = 0
        for _, row in tqdm(df_proxy.iterrows(), total=len(df_proxy), desc='Bootstrapping'):
            samples = np.array(row.to_list())*slope + intercept
            proxy_num[idx] = np.size([s for s in samples if not np.isnan(s)])
            bootstrap_samples = utils.bootstrap(samples, n_bootstraps=n_bootstraps, stat_func=stat_func)
            proxy_sq_low[idx] = np.quantile(bootstrap_samples, qs[0])
            proxy_sq_high[idx] = np.quantile(bootstrap_samples, qs[1])
            idx += 1

        proxy_sq_low_time, proxy_sq_low_value = utils.bin_ts(proxy_comp.index, proxy_sq_low, bin_vector=proxy_bin_vector, smoothed=True)
        proxy_sq_high_time, proxy_sq_high_value = utils.bin_ts(proxy_comp.index, proxy_sq_high, bin_vector=proxy_bin_vector, smoothed=True)
        proxy_num_time, proxy_num_value = utils.bin_ts(proxy_comp.index, proxy_num, bin_vector=proxy_bin_vector, smoothed=True)

        res_dict = {
            'df_proxy': df_proxy,
            'df_obs': df_obs,
            'proxy_comp': proxy_comp,
            'proxy_comp_time': proxy_comp_time,
            'proxy_comp_value': proxy_comp_value,
            'proxy_sq_low_time': proxy_sq_low_time,
            'proxy_sq_high_time': proxy_sq_high_time,
            'proxy_sq_low_value': proxy_sq_low_value,
            'proxy_sq_high_value': proxy_sq_high_value,
            'proxy_num': proxy_num,
            'proxy_num_time': proxy_num_time,
            'proxy_num_value': proxy_num_value,
            'obs_comp': obs_comp,
            'obs_comp_time': obs_comp_time,
            'obs_comp_value': obs_comp_value,
            'bin_width': bin_width,
            'intercept': intercept,
            'slope': slope,
            'r2': r2,
        }

        self.composite = res_dict


    def plot_composite(self, figsize=[10, 4], clr_proxy=None, clr_count='tab:gray', clr_obs='tab:red',
                       left_ylim=[-2, 2], right_ylim=None, ylim_num=5, xlim=[0, 2000], base_n=60,
                       ax=None, bin_width=10):
        ''' Plot the composites of the records in the proxy database.'''
        if clr_proxy is None:
            type_dict = sorted(self.type_dict.items(), key=lambda item: item[1])
            majority = next(iter(type_dict))[0]
            try:
                archive = majority.split('.')[0]
            except:
                archive = majority

            if archive in visual.PAGES2k.archive_types:
                clr_proxy = visual.PAGES2k.colors_dict[archive]
            else:
                clr_proxy = 'tab:blue'

        if ax is None:
            fig = plt.figure(figsize=figsize,facecolor='white')
            ax = {}

        # title_font = {
        #     'fontname': 'Arial',
        #     'size': '24',
        #     'color': 'black',
        #     'weight': 'normal',
        #     'verticalalignment': 'bottom',
        # }
        if self.composite['df_obs'] is not None:
            lb_proxy = fr'proxy, conversion factor = {np.abs(self.composite["slope"]):.3f}, $R^2$ = {self.composite["r2"]:.3f}'
        else:
            lb_proxy = f'proxy, conversion factor = {np.abs(self.composite["slope"]):.3f}'

        ax['var'] = fig.add_subplot()
        ax['var'].plot(self.composite['proxy_comp_time'], self.composite['proxy_comp_value'], color=clr_proxy, lw=1, label=lb_proxy)
        if self.composite['df_obs'] is not None:
            ax['var'].plot(self.composite['obs_comp_time'], self.composite['obs_comp_value'], color=clr_obs, lw=1, label='instrumental')
        ax['var'].fill_between(
            self.composite['proxy_sq_low_time'],
            self.composite['proxy_sq_low_value'],
            self.composite['proxy_sq_high_value'],
            alpha=0.2, color=clr_proxy,
        )
        ax['var'].set_xlim(xlim)
        ax['var'].set_ylim(left_ylim)
        ax['var'].set_yticks(np.linspace(np.min(left_ylim), np.max(left_ylim), ylim_num))
        ax['var'].set_xticks(np.linspace(np.min(xlim), np.max(xlim), 5))
        ax['var'].set_yticks(np.linspace(left_ylim[0], left_ylim[1], ylim_num))
        ax['var'].set_xlabel('Year (CE)')
        ax['var'].set_ylabel('Composite', color=clr_proxy)
        ax['var'].tick_params('y', colors=clr_proxy)
        ax['var'].spines['left'].set_color(clr_proxy)
        ax['var'].spines['bottom'].set_color('k')
        ax['var'].spines['bottom'].set_alpha(0.5)
        ax['var'].spines['top'].set_visible(False)
        ax['var'].yaxis.grid(True, color=clr_proxy, alpha=0.5, ls='-')
        ax['var'].xaxis.grid(False)
        ax['var'].set_title(f'{archive}, {self.nrec} records, bin_width={self.composite["bin_width"]}')
        ax['var'].legend(loc='upper left', frameon=False)

        ax['count'] = ax['var'].twinx()
        ax['count'].set_ylabel('# records', color=clr_count)
        ax['count'].bar(
            self.composite['proxy_comp'].index,
            self.composite['proxy_num'],
            self.composite['bin_width']*0.9, color=clr_count, alpha=0.2)

        if right_ylim is None:
            count_max = int((np.max(self.composite['proxy_num']) // base_n + 1) * base_n)
            right_ylim = [0, count_max]

        ax['count'].set_ylim(right_ylim)
        ax['count'].set_yticks(np.linspace(np.min(right_ylim), np.max(right_ylim), ylim_num))
        ax['count'].grid(False)
        ax['count'].spines['bottom'].set_visible(False)
        ax['count'].spines['right'].set_visible(True)
        ax['count'].spines['left'].set_visible(False)
        ax['count'].spines['top'].set_visible(False)
        ax['count'].tick_params(axis='y', colors=clr_count)
        ax['count'].spines['right'].set_color(clr_count)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax

    def annualize(self, months=list(range(1, 13)), verbose=False):
        ''' Annualize the records in the proxy database.'''
        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Annualizing ProxyRecord'):
            spobj = pobj.annualize(months=months, verbose=verbose)
            if spobj is not None:
                new += spobj

        new.refresh()
        return new

    def slice(self, timespan):
        ''' Slice the records in the proxy database.'''
        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Slicing ProxyRecord'):
            spobj = pobj.slice(timespan=timespan)
            new += spobj

        new.refresh()
        return new


    def del_clim(self, verbose=False):
        ''' Delete the nearest climate data for the records in the proxy database.'''
        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Deleting the nearest climate for ProxyRecord'):
            pobj.del_clim(verbose=verbose)
            new += pobj

        new.refresh()
        return new

    def get_clim(self, field, tag=None, verbose=False, load=True, **kwargs):
        ''' Get the nearest climate data for the records in the proxy database.'''

        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Getting the nearest climate for ProxyRecord'):
            pobj.get_clim(field, tag=tag, verbose=verbose, load=load, **kwargs)
            new += pobj

        new.refresh()

        return new

    def to_df(self):
        ''' Convert the proxy database to a `pandas.DataFrame`.'''
        df = pd.DataFrame(columns=['pid', 'lat', 'lon', 'ptype', 'time', 'value'])
        # df['time'] = df['time'].astype(object)  # not necessary after pandas 1.5.2
        # df['value'] = df['value'].astype(object) # not necessary after pandas 1.5.2

        i = 0
        for pid, pobj in self.records.items():
            df.loc[i, 'pid'] = pobj.pid
            df.loc[i, 'lat'] = pobj.lat
            df.loc[i, 'lon'] = pobj.lon
            df.loc[i, 'ptype'] = pobj.ptype
            df.loc[i, 'time'] = pobj.time
            df.loc[i, 'value'] = pobj.value
            i += 1
            
        return df

    def to_ds(self, annualize=False, months=None, verbose=False):
        ''' Convert the proxy database to a `xarray.Dataset`

        Args:
            annualize (bool): annualize the proxy records with `months`
            months (list): months for annulization
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        da_dict = {}
        pid_truncated = []
        for pobj in tqdm(self, total=self.nrec):
            if np.min(pobj.time) <= 0:
                pid_truncated.append(pobj.pid)
                pobj = pobj.slice([1, np.max(pobj.time)])

            if annualize:
                pobj = pobj.annualize(months=months)

            da = pobj.to_da()

            # remove potential duplicated indexes
            _, index = np.unique(da['time'], return_index=True)
            da = da.isel(time=index)

            if annualize:
                da = da.rename({'time': 'year'})
                da['year'] = np.array([int(t) for t in pobj.time])

            da_dict[pobj.pid] = da

        if verbose:
            utils.p_warning(f'>>> Data before 1 CE is dropped for records: {pid_truncated}.')

        ds = xr.Dataset(da_dict)
        if annualize:
            ds['year'] = np.array([int(t) for t in ds['year']])

        return ds

    def from_ds(self, ds):
        new = self.copy()
        for vn in ds.var():
            da = ds[vn]
            new.records[vn] = ProxyRecord().from_da(da)

        new.refresh()
        return new
        

    def to_nc(self, path, annualize=False, months=None, verbose=True, compress_params={'zlib': True, 'least_significant_digit': 2}):
        ''' Convert the database to a netCDF file.

        Args:
            path (str): the path to save the file.
            annualize (bool): annualize the proxy records with `months`
            months (list): months for annulization
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        encoding_dict = {}
        for k in self.records.keys():
            encoding_dict[k] = compress_params

        ds = self.to_ds(annualize=annualize, months=months)
        ds.to_netcdf(path=path, encoding=encoding_dict)
        if verbose: utils.p_success(f'ProxyDatabase saved to: {path}')

    def load_nc(self, path):
        ds = xr.open_dataset(path)
        pdb = ProxyDatabase().from_ds(ds)
        return pdb

    def to_multi_nc(self, dirpath, verbose=True, compress_params={'zlib': True, 'least_significant_digit': 2}):
        ''' Convert the proxy database to multiple netCDF files. One for each record.

        Args:
            dirpath (str): the directory path of the multiple .nc files
        '''
        os.makedirs(dirpath, exist_ok=True)
        pid_truncated = []
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Saving ProxyDatabase to .nc files'):
            if np.min(pobj.time) <= 0:
                pid_truncated.append(pid)
                pobj = pobj.slice([1, np.max(pobj.time)])
            da = pobj.to_da()
            path = os.path.join(dirpath, f'{pid}.nc')
            da.to_netcdf(path=path, encoding={da.name: compress_params})

        if verbose:
            utils.p_warning(f'>>> Data before 1 CE is dropped for records: {pid_truncated}.')
            utils.p_success(f'>>> ProxyDatabase saved to: {dirpath}')

    def load_multi_nc(self, dirpath, nproc=None):
        ''' Load from multiple netCDF files.

        Args:
            dirpath (str): the directory path of the multiple .nc files
            nproc (int): the number of processors for loading,
                the default is by multiprocessing.cpu_count()

        '''
        paths = sorted(glob.glob(os.path.join(dirpath, '*.nc')))
        new = ProxyDatabase()

        if nproc is None: nproc = cpu_count()
        with Pool(nproc) as pool:
            das = list(
                tqdm(
                    pool.imap(partial(xr.open_dataarray, use_cftime=True), paths),
                    total=len(paths),
                    desc='Loading .nc files',
                )
            )

        for da in das:
            pobj = ProxyRecord().from_da(da)
            new += pobj

        return new