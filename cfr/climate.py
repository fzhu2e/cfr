from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np
import copy
import plotly.express as px
from tqdm import tqdm
from . import visual
from . import utils
import eofs


class ClimateField:
    ''' The class for the gridded climate field data.
    
    Args:
        da (xarray.DataArray): the gridded data array.
        time_name (str): the name of the time dimension.
        lat_name (str): the name of the latitude dimension.
        lon_name (str): the name of the longitude dimension.
    
    '''
    def __init__(self, da=None, time_name=None, lat_name=None, lon_name=None):
        self.da = da
        if self.da is not None:
            self.refresh(time_name=time_name, lat_name=lat_name, lon_name=lon_name)

    def __getitem__(self, key):
        ''' This makes the object subscriptable. '''
        new = self.copy()
        new.da = new.da[key]
        if type(key) is tuple:
            new.time = new.time[key[0]]
        else:
            new.time = new.time[key]
        return new

    def refresh(self, time_name=None, lat_name=None, lon_name=None):
        ''' Refresh a bunch of attributes.
        '''
        time_name = 'time' if time_name is None else time_name
        lat_name = 'lat' if lat_name is None else lat_name
        lon_name = 'lon' if lon_name is None else lon_name

        self.lat = self.da[lat_name].values
        self.lon = self.da[lon_name].values
        if time_name == 'year':
            self.time = self.da[time_name].values
        elif time_name == 'time':
            self.time = utils.datetime2year_float(self.da[time_name].values)
        else:
            raise ValueError('Wrong time_name; should be either "time" or "year".')

        self.vn = self.da.name
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.time_name = time_name
        try:
            self.unit = self.da.attrs['units']
        except:
            self.unit = None

    def wrap_lon(self, mode='360', time_name='time', lon_name='lon'):
        ''' Convert longitude values from the range (-180, 180) to (0, 360).
        '''
        if mode == '360':
            tmp_da = self.da.assign_coords({lon_name: np.mod(self.da[lon_name], 360)})
        elif mode == '180':
            tmp_da = self.da.assign_coords({lon_name: ((self.da[lon_name]+180) % 360)-180})
        else:
            raise ValueError('Wrong mode. Should be either "360" or "180".')

        tmp_da = tmp_da.sortby(tmp_da[self.lon_name])
        new = ClimateField().from_da(tmp_da, time_name=time_name)
        return new

    def from_da(self, da, time_name='time', lat_name='lat', lon_name='lon'):
        ''' Load data from a `xarray.DataArray`.
        '''
        new = self.copy()
        new.da = da
        new.refresh(time_name=time_name, lat_name=lat_name, lon_name=lon_name)
        return new

    def from_np(self, time, lat, lon, value, time_name='time', lat_name='lat', lon_name='lon', value_name='tas'):
        ''' Load data from a `numpy.ndarray`.
        '''
        new = self.copy()
        lat_da = xr.DataArray(lat, dims=[lat_name], coords={lat_name: lat})
        lon_da = xr.DataArray(lon, dims=[lon_name], coords={lon_name: lon})
        time_da = utils.year_float2datetime(time)
        da = xr.DataArray(
            value, dims=[time_name, lat_name, lon_name],
            coords={time_name: time_da, lat_name: lat_da, lon_name: lon_da},
            name=value_name,
        )
        new.da = da
        new.refresh(time_name=time_name, lat_name=lat_name, lon_name=lon_name)
        return new

    def get_anom(self, ref_period=[1951, 1980]):
        ''' Get the anomaly against a reference time period.

        Args:
            ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
        '''
        new = self.copy()

        if ref_period is not None:
            if ref_period[0] > np.max(self.time) or ref_period[-1] < np.min(self.time):
                utils.p_warning(f'>>> The time axis does not overlap with the reference period {ref_period}; use its own time period as reference [{np.min(self.time):.2f}, {np.max(self.time):.2f}].')
                var_ref = self.da
            else:
                var_ref = self.da.loc[str(ref_period[0]):str(ref_period[-1])]

            clim = var_ref.groupby('time.month').mean('time')
            new.da = self.da.groupby('time.month') - clim

        return new

    def pt(self, time):
        ''' Pick a time point or a time slice.

        Args:
            time (float, int, list): a time point or a time range
        '''
        new = self.copy()
        if np.size(np.array(time)) == 1:
            time = [time]

        # time_mask = (self.da[self.time_name]>=time[0]) & (self.da[self.time_name]<=time[-1])
        time_mask = (self.time>=time[0]) & (self.time<=time[-1])
        new.da = self.da.sel({self.time_name: self.da[self.time_name][time_mask]})
        new.refresh(time_name=self.time_name)
        return new

    def center(self, ref_period=[1951, 1980], time_name='time'):
        ''' Center the climate field against a reference time period.

        Args:
            ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
            time_name (str): name of the time dimention
        '''
        new = self.copy()

        if ref_period is not None:
            if ref_period[0] > np.max(self.time) or ref_period[-1] < np.min(self.time):
                utils.p_warning(f'>>> The time axis does not overlap with the reference period {ref_period}; use its own time period as reference [{np.min(self.time):.2f}, {np.max(self.time):.2f}].')
                var_ref = self.da
            else:
                var_ref = self.da.loc[str(ref_period[0]):str(ref_period[-1])]

            clim = var_ref.mean(time_name)
            new.da = self.da - clim

        return new

    def load_nc(self, path, vn=None, time_name='time', lat_name='lat', lon_name='lon', load=False, **kwargs):
        ''' Load the climate field from a netCDF file.

        Args:
            path (str): the path where to load data from
        '''
        if vn is None: 
            da = xr.open_dataarray(path, **kwargs)
        else:
            ds = xr.open_dataset(path, **kwargs)
            da = ds[vn]

        new = ClimateField(da=da, time_name=time_name, lat_name=lat_name, lon_name=lon_name)
        if lat_name != 'lat' or lon_name != 'lon':
            new = new.rename({lat_name: 'lat', lon_name: 'lon'}, modify_vn=False)  # rename dimension names only

        if load: new.da.load()
        return new

    def to_nc(self, path, verbose=True, **kwargs):
        ''' Convert the climate field to a netCDF file.

        Args:
            path (str): the path where to save
        '''
        self.da.to_netcdf(path, **kwargs)
        if verbose: utils.p_success(f'ClimateField saved to: {path}')

    def copy(self):
        ''' Make a deepcopy of the object. '''
        return copy.deepcopy(self)

    def rename(self, new_vn, modify_vn=True):
        ''' Rename the variable name of the climate field.'''
        new = self.copy()
        new.da = self.da.rename(new_vn)
        if modify_vn:
            new.vn = new_vn
        return new

    # def __add__(self, fields):
    #     ''' Add a list of fields into the dataset
    #     '''
    #     new = ClimateDataset()
    #     new.fields[self.vn] = self.copy()
    #     if isinstance(fields, ClimateField):
    #         fields = [fields]

    #     if isinstance(fields, ClimateDataset):
    #         fields = [fields.fields[vn] for vn in fields.fields.keys()]

    #     for field in fields:
    #         new.fields[field.vn] = field

    #     new.refresh()
    #     return new

    def __add__(self, ref):
        ''' Add the reference field.
        '''
        new = self.copy()
        if isinstance(ref, ClimateField):
            new.da = self.da + ref.da
        elif isinstance(ref, float):
            new.da = self.da + ref
        elif isinstance(ref, int):
            new.da = self.da + ref
        else:
            raise ValueError('`ref` should be a `ClimateField` object or a float like value.')

        return new

    def __sub__(self, ref):
        ''' Substract the reference field.
        '''
        new = self.copy()
        if isinstance(ref, ClimateField):
            new.da = self.da - ref.da
        elif isinstance(ref, float):
            new.da = self.da - ref
        elif isinstance(ref, int):
            new.da = self.da - ref
        else:
            raise ValueError('`ref` should be a `ClimateField` object or a float like value.')

        return new

    def validate(self, ref, stat='corr', interp_direction='to-ref', time_name='year', valid_period=(1880, 2000)):
        ''' Validate against a reference field.

        Args:
            ref (cfr.climate.ClimateField): the reference to compare against, assuming the first dimension to be time
            valid_period (tuple, optional): the time period for validation. Defaults to None.
            interp_direction (str, optional): the direction to interpolate the fields:
            
                * 'to-ref': interpolate from `self` to `ref`
                * 'from-ref': interpolate from `ref` to `self`
            stat (str): the statistics to calculate. Supported quantaties:

                * 'corr': correlation coefficient
                * 'R2': coefficient of determination
                * 'CE': coefficient of efficiency
        '''
        fd_slice = self.da.sel({time_name: slice(valid_period[0], valid_period[-1])})
        ref_slice = ref.da.sel({time_name: slice(valid_period[0], valid_period[-1])})

        if interp_direction == 'to-ref':
            fd_slice = fd_slice.interp({'lat': ref_slice.lat, 'lon': ref_slice.lon})
        elif interp_direction == 'from-ref':
            ref_slice = ref_slice.interp({'lat': fd_slice.lat, 'lon': fd_slice.lon})

        if 'time' in ref_slice.dims:
            time_name = 'time'
        elif 'year' in ref_slice.dims:
            time_name = 'year'

        fd_slice = xr.DataArray(
            fd_slice.values,
            coords={time_name: ref_slice[time_name], 'lat': fd_slice.lat, 'lon': fd_slice.lon}
        )

        if stat == 'corr':
            stat_da = xr.corr(fd_slice, ref_slice, dim=time_name)
            stat_da = stat_da.expand_dims({'year': 1})
            stat_fd = ClimateField().from_da(da=stat_da, time_name='year')
            stat_fd.vn = stat
            stat_fd.plot_kwargs = {
                'cmap': 'RdBu_r',
                'extend': 'neither',
                'levels': np.linspace(-1, 1, 21),
                'cbar_labels': np.linspace(-1, 1, 11),
                'cbar_title': r'$r$',
            }
        elif stat == 'R2':
            stat_da = xr.corr(fd_slice, ref_slice, dim=time_name)
            stat_da = stat_da.expand_dims({'year': 1})
            stat_fd = ClimateField().from_da(da=stat_da**2, time_name='year')
            stat_fd.vn = stat
            stat_fd.plot_kwargs = {
                'cmap': 'Reds',
                'extend': 'neither',
                'levels': np.linspace(0, 1, 21),
                'cbar_labels': np.linspace(0, 1, 11),
                'cbar_title': r'$R^2$',
            }
        elif stat == 'CE':
            ce = utils.coefficient_efficiency(ref_slice.values, fd_slice.values)
            stat_da = xr.DataArray(ce[np.newaxis], coords={'year': [1], 'lat': fd_slice.lat, 'lon': fd_slice.lon})
            stat_fd = ClimateField().from_da(da=stat_da, time_name='year')
            stat_fd.vn = stat
            stat_fd.plot_kwargs = {
                'cmap': 'RdBu_r',
                'extend': 'both',
                'levels': np.linspace(-1, 1, 21),
                'cbar_labels': np.linspace(-1, 1, 11),
                'cbar_title': r'$CE$',
            }
        else:
            raise ValueError('Wrong `stat`; should be one of `corr`, `R2`, and `CE`.' )

        return stat_fd
            
    def get_eof(self, n=1, time_period=None, verbose=False, flip=False):
        if time_period is None:
            da = self.da
        else:
            da = self.da.loc[f'{time_period[0]}':f'{time_period[1]}']

        if flip:
            da = -da

        coslat = np.cos(np.deg2rad(self.da['lat'].values))
        wgts = np.sqrt(coslat)[..., np.newaxis]
        solver = eofs.standard.Eof(da.values, weights=wgts)
        modes = solver.eofsAsCorrelation(neofs=n)
        pcs = solver.pcs(npcs=n, pcscaling=1)
        fracs = solver.varianceFraction(neigs=n)

        self.eof_res = {
            'modes': modes,
            'pcs': pcs,
            'fracs': fracs,
            'time_period': time_period,
        }

        if verbose: utils.p_success(f'ClimateField.eof_res created with {n} mode(s).')

    def plot_eof(self, n=1, eof_title=None, pc_title=None):
        eof = self.eof_res['modes']
        pc = self.eof_res['pcs']
        fracs = self.eof_res['fracs']
        time_period = self.eof_res['time_period']
        if eof_title is None: eof_title = f'EOF{n} ({fracs[n-1]*100:.2f}%)'
        if pc_title is None: pc_title = f'PC{n} ({fracs[n-1]*100:.2f}%)'

        if time_period is None:
            time = self.time
        else:
            time_mask = (self.time>=time_period[0]) & (self.time<=time_period[1])
            time = self.time[time_mask]

        fig, ax = visual.plot_eof(
            eof[n-1], pc[:, n-1], self.lat, self.lon, time,
            eof_title, pc_title)

        return fig, ax

    def plot(self, it=0, **kwargs):
        ''' Plot the climate field.'''
        if self.da[self.time_name].values.ndim == 0:
            t = self.da[self.time_name].values
        else:
            t = self.da[self.time_name].values[it]

        try:
            yyyy = str(t)[:4]
            mm = str(t)[5:7]
            t = datetime(year=int(yyyy), month=int(mm), day=1)
        except:
            pass

        if isinstance(t, np.datetime64):
            # convert to cftime.datetime
            t = utils.datetime2year_float([t])
            t = utils.year_float2datetime(t)[0]

        cbar_title = visual.make_lb(self.vn, self.unit)
        cmap_dict = {
            'tas': 'RdBu_r',
            'tos': 'RdBu_r',
            'pr': 'BrBG',
        }
        cmap = cmap_dict[self.vn] if self.vn in cmap_dict else 'viridis'
        if 'title' not in kwargs:
            if self.time_name == 'time':
                kwargs['title'] = f'{self.vn}, {t.year}-{t.month}' 
            elif self.time_name == 'year':
                kwargs['title'] = f'{self.vn}, {t}' 

        _kwargs = {
            'cbar_title': cbar_title,
            'cmap': cmap,
        }
        _kwargs.update(kwargs)
        if len(self.da.dims) == 3:
            fig, ax =  visual.plot_field_map(self.da.values[it], self.lat, self.lon, **_kwargs)
        elif len(self.da.dims) == 2:
            fig, ax =  visual.plot_field_map(self.da.values, self.lat, self.lon, **_kwargs)

        return fig, ax

    def plotly_grid(self, site_lats=None, site_lons=None, **kwargs):
        ''' Plot the grid on an interactive map utilizing Plotly
        '''
        nlat, nlon = np.size(self.lat), np.size(self.lon)
        df = pd.DataFrame()
        n = 0
        for i in range(nlat):
            for j in range(nlon):
                lat = self.lat[i]
                lon = self.lon[j]
                if not np.isnan(self.da[-1, i, j]):
                    avail = 'Data'
                else:
                    avail = 'NaN'
                df.loc[n, 'lat'] = lat
                df.loc[n, 'lon'] = lon
                df.loc[n, 'Type'] = avail
                n += 1

        if site_lats is not None:
            if type(site_lats) is not list:
                site_lats = [site_lats]

        if site_lons is not None:
            if type(site_lons) is not list:
                site_lons = [site_lons]

        for i, site_lat in enumerate(site_lats):
            for j, site_lon in enumerate(site_lons):
                df.loc[n, 'lat'] = site_lat
                df.loc[n, 'lon'] = site_lon
                df.loc[n, 'Type'] = 'Site'
                n += 1

        fig = px.scatter_geo(
            df, lat='lat', lon='lon',
            color='Type',
            projection='natural earth',
            **kwargs,
        )

        return fig

    def annualize(self, months=list(range(1, 13))):
        ''' Annualize/seasonalize the climate field based on a list of months.

        Args:
            months (list): the months based on which for annualization; e.g., [6, 7, 8] means JJA annualization
        '''
        new = self.copy()
        new.da = utils.annualize(da=self.da, months=months)
        time = np.floor(utils.datetime2year_float(new.da.time.values))
        new.time = np.array([int(t) for t in time])
        new.time_name = 'year'
        if 'time' in new.da.dims:
            new.da = new.da.rename({'time': 'year'})
            new.da['year'] = new.time

        return new

    def regrid(self, lats, lons, periodic_lon=False):
        ''' Regrid the climate field.'''

        new = self.copy()
        lat_da = xr.DataArray(lats, dims=[self.lat_name], coords={self.lat_name: lats})
        lon_da = xr.DataArray(lons, dims=[self.lon_name], coords={self.lon_name: lons})
        dai = self.da.interp(coords={self.lon_name: lon_da, self.lat_name: lat_da})
        if periodic_lon:
            dai[..., -1] = dai[..., 0]

        new.da = dai
        new.lat = dai[self.lat_name]
        new.lon = dai[self.lon_name]

        return new

    def crop(self, time_min=None, time_max=None, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        ''' Crop the climate field.'''
        new = self.copy()
        if time_min is not None and time_max is not None:
            mask_time = (self.da[self.time_name] >= time_min) & (self.da[self.time_name] <= time_max)
        else:
            mask_time = None

        if lat_min is not None and lat_max is not None:
            mask_lat = (self.da[self.lat_name] >= lat_min) & (self.da[self.lat_name] <= lat_max)
        else:
            mask_lat = None

        if lon_min is not None and lon_max is not None:
            mask_lon = (self.da[self.lon_name] >= lon_min) & (self.da[self.lon_name] <= lon_max)
        else:
            mask_lon = None

        crop_dict = {}

        if mask_time is not None:
            crop_dict[self.time_name] = self.da[self.time_name][mask_time]

        if mask_lat is not None:
            crop_dict[self.lat_name] = self.da[self.lat_name][mask_lat]

        if mask_lon is not None:
            crop_dict[self.lon_name] = self.da[self.lon_name][mask_lon]

        dac = self.da.sel(crop_dict)
        new.da = dac
        new.refresh(time_name=self.time_name, lon_name=self.lon_name, lat_name=self.lat_name)

        return  new

    def geo_mean(self, lat_min=-90, lat_max=90, lon_min=0, lon_max=360, lat_name='lat', lon_name='lon'):
        ''' Calculate the geographical mean value of the climate field. '''
        m = utils.geo_mean(self.da, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
                           lat_name=lat_name, lon_name=lon_name)
        return m


# Not very useful at this moment.
# class ClimateDataset:
#     ''' The class for the gridded climate field dataset
    
#     Args:
#         fields (dict): the dictionary of :py:mod:`cfr.climate.ClimateField`
    
#     '''
#     def __init__(self, fields=None):
#         self.fields = {} if fields is None else fields
#         if fields is not None:
#             self.refresh()

#     def refresh(self):
#         ''' Refresh a bunch of attributes. '''
#         self.nv = len(list(self.fields.keys()))

#     def copy(self):
#         ''' Make a deepcopy of the object. '''
#         return copy.deepcopy(self)

#     def load_nc(self, path, time_name='time', lat_name='lat', lon_name='lon', vn=None, load=False, **kwargs):
#         new = self.copy()
#         fields = {}
#         ds = xr.open_dataset(path, **kwargs)
#         if load: ds.load()
#         if vn is None:
#             for k in ds.variables.keys():
#                 if k not in [time_name, lat_name, lon_name]:
#                     fields[k] = ClimateField(da=ds[k], time_name=time_name, lat_name=lat_name, lon_name=lon_name)
#         else:
#             fields[vn] = ClimateField(da=ds[vn], time_name=time_name, lat_name=lat_name, lon_name=lon_name)

#         new.fields = fields
#         new.nv = len(list(fields.keys()))
#         return new

#     def __add__(self, fields):
#         ''' Add a list of fields into the dataset
#         '''
#         new = self.copy()
#         if isinstance(fields, ClimateField):
#             fields = [fields]

#         if isinstance(fields, ClimateDataset):
#             fields = [fields.fields[vn] for vn in fields.fields.keys()]

#         for field in fields:
#             new.fields[field.vn] = field

#         new.refresh()
#         return new

#     def __sub__(self, fields):
#         ''' Add a list of fields into the dataset
#         '''
#         new = self.copy()
#         if isinstance(fields, ClimateField):
#             fields = [fields]

#         if isinstance(fields, ClimateDataset):
#             fields = [fields.fields[vn] for vn in fields.fields.keys()]

#         for field in fields:
#             try:
#                 del new.fields[field.vn]
#             except:
#                 utils.p_warning(f'>>> Subtracting {field.vn} failed.')

#         new.refresh()
#         return new

#     def annualize(self, months=list(range(1, 13))):
#         ''' Annualize/seasonalize the climate dataset based on a list of months.

#         Args:
#             months (list): the months based on which for annualization; e.g., [6, 7, 8] means JJA annualization
#         '''
#         new = ClimateDataset()
#         for vn, fd in tqdm(self.fields.items(), total=self.nv, desc='Annualizing ClimateField'):
#             sfd = fd.annualize(months=months)
#             new += sfd

#         new.refresh()
#         return new

#     def get_anom(self, ref_period=[1951, 1980]):
#         ''' Get anomaly of the climate dataset against a reference time period.

#         Args:
#             ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
#         '''
#         new = ClimateDataset()
#         for vn, fd in tqdm(self.fields.items(), total=self.nv, desc='Getting anomaly from ClimateField'):
#             sfd = fd.get_anom(ref_period=ref_period)
#             new += sfd
#         new.refresh()
#         return new

#     def center(self, ref_period=[1951, 1980]):
#         ''' Center the climate dataset against a reference time period.

#         Args:
#             ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
#         '''
#         new = ClimateDataset()
#         for vn, fd in tqdm(self.fields.items(), total=self.nv, desc='Getting anomaly from ClimateField'):
#             sfd = fd.center(ref_period=ref_period)
#             new += sfd
#         new.refresh()
#         return new

#     def from_ds(self, ds, time_name='time', lat_name='lat', lon_name='lon'):
#         ''' Get data from the Xarray.Dataset object
#         '''
#         new = self.copy()
#         fields = {}
#         for k in ds.variables.keys():
#             if k not in ['time', 'lat', 'lon']:
#                 fields[k] = ClimateField(da=ds[k], time_name=time_name, lat_name=lat_name, lon_name=lon_name)

#         new.fields = fields
#         return new

#     def regrid(self, lat, lon):
#         ''' Regrid the climate dataset

#         Args:
#             lat (numpy.array): the latitudes of the target grid.
#             lon (numpy.array): the longitudes of the target grid.

#         '''
#         lat_da = xr.DataArray(lat, dims=['lat'], coords={'lat': lat})
#         lon_da = xr.DataArray(lon, dims=['lon'], coords={'lon': lon})
#         dsi = self.ds.interp(lon=lon_da, lat=lat_da)
#         new = self.from_ds(dsi)

#         return new
