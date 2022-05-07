import glob
import os
from .climate import ClimateField, ClimateDataset
import xarray as xr
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
from multiprocessing import Pool, cpu_count
from functools import partial
from . import utils
from . import visual

def get_ptype(archive_type, proxy_type):
    ptype_dict = {
        ('tree', 'delta Density'): 'tree.MXD',
        ('tree', 'MXD'): 'tree.MXD',
        ('tree', 'TRW'): 'tree.TRW',
        ('tree', 'ENSO'): 'tree.ENSO',
        ('coral', 'Sr/Ca'): 'coral.SrCa',
        ('coral', 'Coral Sr/Ca'): 'coral.SrCa',
        ('coral', 'd18O'): 'coral.d18O',
        ('coral', 'calcification'): 'coral.calc',
        ('coral', 'calcification rate'): 'coral.calc',
        ('sclerosponge', 'd18O'): 'coral.d18O',
        ('sclerosponge', 'Sr/Ca'): 'coral.SrCa',
        ('glacier ice', 'melt'): 'ice.melt',
        ('glacier ice', 'd18O'): 'ice.d18O',
        ('glacier ice', 'dD'): 'ice.dD',
        ('speleothem', 'd18O'): 'speleothem.d18O',
        ('marine sediment', 'TEX86'): 'marine.TEX86',
        ('marine sediment', 'foram Mg/Ca'): 'marine.MgCa',
        ('marine sediment', 'd18O'): 'marine.d18O',
        ('marine sediment', 'dynocist MAT'): 'marine.MAT',
        ('marine sediment', 'alkenone'): 'marine.alkenone',
        ('marine sediment', 'planktonic foraminifera'): 'marine.foram',
        ('marine sediment', 'foraminifera'): 'marine.foram',
        ('marine sediment', 'foram d18O'): 'marine.foram',
        ('marine sediment', 'diatom'): 'marine.diatom',
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
        ('borehole', 'borehole'): 'borehole',
        ('hybrid', 'hybrid'): 'hybrid',
        ('bivalve', 'd18O'): 'bivalve.d18O',
        ('documents', 'Documentary'): 'documents',
        ('documents', 'historic'): 'documents',
    }

    return ptype_dict[(archive_type, proxy_type)]

class PAGES2k:
    ''' A bunch of PAGES2k style settings
    '''
    archive_types = [
        'bivalve',
        'borehole',
        'coral',
        'documents',
        'ice',
        'hybrid',
        'lake',
        'marine',
        'sclerosponge',
        'speleothem',
        'tree',
    ]
    markers = ['p', 'p', 'o', 'v', 'd', '*', 's', 's', '8', 'D', '^']
    markers_dict = dict(zip(archive_types, markers))
    colors = [np.array([ 1.        ,  0.83984375,  0.        ]),
              np.array([ 0.73828125,  0.71484375,  0.41796875]),
              np.array([ 1.        ,  0.546875  ,  0.        ]),
              np.array([ 0.41015625,  0.41015625,  0.41015625]),
              np.array([ 0.52734375,  0.8046875 ,  0.97916667]),
              np.array([ 0.        ,  0.74609375,  1.        ]),
              np.array([ 0.25390625,  0.41015625,  0.87890625]),
              np.array([ 0.54296875,  0.26953125,  0.07421875]),
              np.array([ 1         ,           0,           0]),
              np.array([ 1.        ,  0.078125  ,  0.57421875]),
              np.array([ 0.1953125 ,  0.80078125,  0.1953125 ])]
    colors_dict = dict(zip(archive_types, colors))

class ProxyRecord:
    def __init__(self, pid=None, time=None, value=None, lat=None, lon=None, ptype=None, tags=None,
        value_name=None, value_unit=None, time_name=None, time_unit=None, seasonality=None):
        '''
        Parameters
        ----------
        pid : str
            the unique proxy ID

        lat : float
            latitude

        lon : float
            longitude

        time : np.array
            time axis in unit of year CE 

        value : np.array
            proxy value axis

        ptype : str
            the label of proxy type according to archive and proxy information;
            some examples:
            - 'tree.trw' : TRW
            - 'tree.mxd' : MXD
            - 'coral.d18O' : Coral d18O isotopes
            - 'coral.SrCa' : Coral Sr/Ca ratios
            - 'ice.d18O' : Ice d18O isotopes

        tags : a set of str
            the tags for the record, to enable tag filtering
        '''
        self.pid = pid
        self.time = time
        self.value = value
        self.lat = lat
        self.lon = lon
        self.ptype = ptype
        self.tags = set() if tags is None else tags

        self.dt = np.median(np.diff(time)) if time is not None else None
        self.value_name = 'Proxy Value' if value_name is None else value_name
        self.value_unit = value_unit
        self.time_name = 'Time' if time_name is None else time_name
        self.time_unit = 'yr' if time_unit is None else time_unit
        self.seasonality = seasonality

    def copy(self):
        return copy.deepcopy(self)

    def slice(self, timespan):
        ''' Slicing the timeseries with a timespan (tuple or list)

        Parameters
        ----------

        timespan : tuple or list
            The list of time points for slicing, whose length must be even.
            When there are n time points, the output Series includes n/2 segments.
            For example, if timespan = [a, b], then the sliced output includes one segment [a, b];
            if timespan = [a, b, c, d], then the sliced output includes segment [a, b] and segment [c, d].

        Returns
        -------

        new : Series
            The sliced Series object.

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

    def to_nc(self, path, verbose=True, **kwargs):
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
        new.time = utils.datetime2year_float(da.time.values)
        new.value = da.values
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
            return new
        except:
            if verbose:
                print(f'Record {self.pid} cannot be annualized with months {months}. None returned.')
            return None
            

    def standardize(self):
        new = self.copy()
        if self.value.std() == 0:
            new.value = np.zeros(np.size(self.value))
        else:
            new.value = (self.value - self.value.mean()) / self.value.std()
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

    def del_clim(self, verbose=False):
        if hasattr(self, 'clim'): del self.clim
        if verbose: utils.p_success(f'ProxyRecord.clim deleted for {self.pid}.')

    def get_clim(self, fields, tag=None, verbose=False, load=True, **kwargs):
        ''' Get the nearest climate from cliamte fields

        Parameters
        ----------
        fields : cfr.climate.ClimateField or cfr.climate.ClimateDataset
            the climate field
        '''
        if isinstance(fields, ClimateDataset):
            fields = list(fields.fields.values())
        elif isinstance(fields, ClimateField):
            fields = [fields]

        _kwargs = {'method': 'nearest'}
        _kwargs.update(kwargs)
        
        for field in fields:
            name = field.da.name
            if tag is not None:
                name = f'{tag}_{name}'

            nda = field.da.sel(lat=self.lat, lon=self.lon, **_kwargs)
            if not hasattr(self, 'clim'):
                self.clim = {}

            self.clim[name] = ClimateField().from_da(nda)
            if load: self.clim[name].da.load()
            if verbose: utils.p_success(f'ProxyRecord.clim["{name}"] created.')


    def plotly(self, **kwargs):
        import plotly.express as px
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
            kwargs['color'] = visual.PAGES2k.colors_dict[self.ptype]

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

        title = f'{self.pid} @ (lat:{self.lat}, lon:{self.lon}) | Type: {self.ptype}'
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
                self.lon, self.lat, marker=visual.PAGES2k.markers_dict[self.ptype],
                s=ms, c=kwargs['color'], edgecolor=edge_clr, transform=transform,
            )

        return fig, ax



class ProxyDatabase:
    def __init__(self, records=None, source=None):
        '''
        Parameters
        ----------
        records : dict
            a dict of the ProxyRecord objects with proxy ID as keys

        source : str
            a path to the original source file

        '''
        self.records = {} if records is None else records
        self.source = source
        if records is not None:
            self.refresh()

    def copy(self):
        return copy.deepcopy(self)

    def refresh(self):
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

        Parameters
        ----------
        df : Pandas DataFrame
            a Pandas DataFrame include at least lat, lon, time, value, proxy_type
        
        ptype_psm : dict
            a mapping from ptype to psm
        '''
        new = self.copy()
        if not isinstance(df, pd.DataFrame):
            err_msg = 'the input df should be a Pandas DataFrame.'
            if verbose:
                utils.p_fail(f'ProxyDatabase.load_df() >>> {err_msg}')
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
        ''' Add a list of records into the database
        '''
        new = self.copy()
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

    def __sub__(self, records):
        ''' Subtract a list of records from a database

        Args:
            records (list): a list of cfr.proxy.ProxyRecord
                can also be a single cfr.proxy.ProxyRecord or cfr.proxy.ProxyDatabase

        '''
        new = self.copy()
        if isinstance(records, ProxyRecord):
            # if only one record
            records = [records]

        if isinstance(records, ProxyDatabase):
            # if a database
            records = [records.records[pid] for pid in records.records.keys()]

        for record in records:
            try:
                del new.records[record.pid]
            except:
                utils.p_warning(f'>>> Subtracting {record.pid} failed.')

        new.refresh()
        return new

    def filter(self, by, keys):
        ''' Filter the proxy database according to given ptype list

        Parameters
        ----------
        by : str
            filter by a keyword {'ptype', 'pid', 'lat', 'lon', 'loc', 'tag'}

        keys : set
            | a set of keywords
            | For by = 'ptype' or 'pid', keys take a fuzzy match
            | For by = 'lat' or 'lon', keys = (lat_min, lat_max) or (lon_min, lon_max)
            | For by = 'loc-squre', keys = (lat_min, lat_max, lon_min, lon_max)
            | For by = 'loc-circle', keys = (center_lat, center_lon, distance)

        '''
        if isinstance(keys, str): keys = [keys]
        keys = set(keys)

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
                    if key in target[by]:
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
                if keys <= target[by]:
                    pobjs.append(pobj)
            
        new_db += pobjs
        new_db.refresh()
        return new_db

    def nrec_tags(self, keys):
        nrec = 0
        if isinstance(keys, str): keys = [keys]
        keys = set(keys)
        for pid, pobj in self.records.items():
            if keys <= pobj.tags:
                nrec += 1

        return nrec

    def plot(self, **kws):

        time_list = []
        for pid, pobj in self.records.items():
            time_list.append(pobj.time)

        df = pd.DataFrame({'lat': self.lats, 'lon': self.lons, 'type': self.type_list, 'time': time_list})
        fig, ax = visual.plot_proxies(df, **kws)

        return fig, ax

    def make_composite(self, obs_nc_path, vn='tas', lat_name=None, lon_name=None, bin_width=10, n_bootstraps=1000, qs=(0.025, 0.975), stat_func=np.nanmean, anom_period=[1951, 1980]):
        obs = ClimateField().load_nc(obs_nc_path, vn=vn, lat_name=lat_name, lon_name=lon_name)

        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Analyzing ProxyRecord'):
            pobj.get_clim(obs, tag='obs')
            pobj_stdd = pobj.standardize()

            proxy_time, proxy_value, _ = utils.smooth_ts(pobj_stdd.time, pobj_stdd.value, bin_width=bin_width)

            ts_proxy = pd.Series(index=proxy_time, data=proxy_value, name=pid)
            if 'df_proxy' not in locals():
                df_proxy = ts_proxy.to_frame()
            else:
                df_proxy = pd.merge(df_proxy, ts_proxy, left_index=True, right_index=True, how='outer')

            pobj.clim[f'obs_{vn}'].center(ref_period=anom_period)
            obs_time, obs_value, _ = utils.smooth_ts(pobj.clim[f'obs_{vn}'].time, pobj.clim[f'obs_{vn}'].da.values, bin_width=bin_width)
            ts_obs = pd.Series(index=obs_time, data=obs_value, name=pid)
            if 'df_obs' not in locals():
                df_obs = ts_obs.to_frame()
            else:
                df_obs = pd.merge(df_obs, ts_obs, left_index=True, right_index=True, how='outer')

        proxy_comp = df_proxy.apply(stat_func, axis=1)
        obs_comp = df_obs.apply(stat_func, axis=1)
        ols_model = utils.ols_ts(proxy_comp.index, proxy_comp.values, obs_comp.index, obs_comp.values)
        results = ols_model.fit()
        intercept = results.params[0]
        slope = results.params[1]
        r2 = results.rsquared
        proxy_comp_scaled = proxy_comp.values*slope + intercept

        proxy_bin_vector = utils.make_bin_vector(proxy_comp.index, bin_width=bin_width)
        obs_bin_vector = utils.make_bin_vector(obs_comp.index, bin_width=bin_width)
        proxy_comp_time, proxy_comp_value = utils.bin_ts(proxy_comp.index, proxy_comp_scaled, bin_vector=proxy_bin_vector, smoothed=True)
        obs_comp_time, obs_comp_value = utils.bin_ts(obs_comp.index, obs_comp.values, bin_vector=obs_bin_vector, smoothed=True)

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
            'intercept': intercept,
            'slope': slope,
            'r2': r2,
        }

        self.composite = res_dict


    def plot_composite(self, figsize=[10, 4], clr_proxy=None, clr_count='tab:gray', clr_obs='tab:red',
                       left_ylim=[-2, 2], right_ylim=None, xlim=[0, 2000], base_n=60,
                       ax=None, bin_width=10):
        if clr_proxy is None:
            type_dict = sorted(self.type_dict.items(), key=lambda item: item[1])
            majority = next(iter(type_dict))[0]
            try:
                archive = majority.split('.')[0]
            except:
                archive = majority

            if archive in PAGES2k().archive_types:
                clr_proxy = PAGES2k().colors_dict[archive]
            else:
                clr_proxy = 'tab:blue'

        if ax is None:
            fig = plt.figure(figsize=figsize,facecolor='white')
            ax = {}

        title_font = {
            'fontname': 'Arial',
            'size': '24',
            'color': 'black',
            'weight': 'normal',
            'verticalalignment': 'bottom',
        }
        lb_proxy = fr'proxy, conversion factor = {np.abs(self.composite["slope"]):.3f}, $R^2$ = {self.composite["r2"]:.3f}'
        ax['var'] = fig.add_subplot()
        ax['var'].plot(self.composite['proxy_comp_time'], self.composite['proxy_comp_value'], color=clr_proxy, lw=1, label=lb_proxy)
        ax['var'].plot(self.composite['obs_comp_time'], self.composite['obs_comp_value'], color=clr_obs, lw=1, label='instrumental')
        ax['var'].fill_between(
            self.composite['proxy_sq_low_time'],
            self.composite['proxy_sq_low_value'],
            self.composite['proxy_sq_high_value'],
            alpha=0.2, color=clr_proxy,
        )
        ax['var'].set_xlim(xlim)
        ax['var'].set_ylim(left_ylim)
        ax['var'].set_yticks(np.linspace(np.min(left_ylim), np.max(left_ylim), 5))
        ax['var'].set_xticks(np.linspace(np.min(xlim), np.max(xlim), 5))
        ax['var'].set_yticks(np.linspace(-2, 2, 5))
        ax['var'].set_xlabel('Year (CE)')
        ax['var'].set_ylabel('proxy', color=clr_proxy)
        ax['var'].tick_params('y', colors=clr_proxy)
        ax['var'].spines['left'].set_color(clr_proxy)
        ax['var'].spines['bottom'].set_color(clr_proxy)
        ax['var'].spines['bottom'].set_alpha(0.5)
        ax['var'].spines['top'].set_visible(False)
        ax['var'].yaxis.grid(True, color=clr_proxy, alpha=0.5, ls='-')
        ax['var'].xaxis.grid(False)
        ax['var'].set_title(f'{archive}, {self.nrec} records, bin_width={bin_width}')
        ax['var'].legend(loc='upper left', frameon=False)

        ax['count'] = ax['var'].twinx()
        ax['count'].set_ylabel('# records', color=clr_count)
        ax['count'].bar(
            self.composite['proxy_comp'].index,
            self.composite['proxy_num'],
            bin_width*0.9, color=clr_count, alpha=0.2)

        if right_ylim is None:
            count_max = int((np.max(self.composite['proxy_num']) // base_n + 1) * base_n)
            right_ylim = [0, count_max]

        ax['count'].set_ylim(right_ylim)
        ax['count'].set_yticks(np.linspace(np.min(right_ylim), np.max(right_ylim), 5))
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
        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Annualizing ProxyRecord'):
            spobj = pobj.annualize(months=months, verbose=verbose)
            if spobj is not None:
                new += spobj

        new.refresh()
        return new

    def slice(self, timespan):
        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Slicing ProxyRecord'):
            spobj = pobj.slice(timespan=timespan)
            new += spobj

        new.refresh()
        return new


    def del_clim(self, verbose=False):
        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Deleting the nearest climate for ProxyRecord'):
            pobj.del_clim(verbose=verbose)
            new += pobj

        new.refresh()
        return new

    def get_clim(self, field, tag=None, verbose=False, load=True, **kwargs):

        new = ProxyDatabase()
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Getting the nearest climate for ProxyRecord'):
            pobj.get_clim(field, tag=tag, verbose=verbose, load=load, **kwargs)
            new += pobj

        new.refresh()

        return new

    def to_nc(self, dirpath, verbose=True, **kwargs):
        os.makedirs(dirpath, exist_ok=True)
        pid_truncated = []
        for pid, pobj in tqdm(self.records.items(), total=self.nrec, desc='Saving ProxyDatabase to .nc files'):
            if np.min(pobj.time) <= 0:
                pid_truncated.append(pid)
                pobj = pobj.slice([1, np.max(pobj.time)])
            da = pobj.to_da()
            path = os.path.join(dirpath, f'{pid}.nc')
            da.to_netcdf(path=path, **kwargs)

        if verbose:
            utils.p_warning(f'>>> Data before 1 CE is dropped for records: {pid_truncated}.')
            utils.p_success(f'>>> ProxyDatabase saved to: {dirpath}')

    def load_nc(self, dirpath, nproc=None, **kwargs):
        ''' Load from multiple .nc files

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