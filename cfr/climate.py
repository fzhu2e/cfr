import os
from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np
import copy
import plotly.express as px
from tqdm import tqdm
from . import visual
from . import utils
from .ts import EnsTS
import eofs


class ClimateField:
    ''' The class for the gridded climate field data.
    
    Args:
        da (xarray.DataArray): the gridded data array.
    '''
    def __init__(self, da=None):
        self.da = da

    def __getitem__(self, key):
        ''' This makes the object subscriptable. '''
        if type(key) is int or type(key) is list:
            da = self.da[key]
        else:
            if type(key) is str: 
                key = slice(key, key, None)
            elif type(key) is not slice:
                raise TypeError('Wrong type for key!')

            try:
                da = self.da.sel({'time': slice(float(key.start), float(key.stop), key.step)})
            except:
                da = self.da.loc[key]

        fd = ClimateField(da)

        return fd

    def __len__(self):
        return len(self.time)

    def wrap_lon(self, mode='360'):
        ''' Convert longitude values from the range (-180, 180) to (0, 360).
        '''
        if mode == '360':
            da = self.da.assign_coords({'lon': np.mod(self.da['lon'], 360)})
        elif mode == '180':
            da = self.da.assign_coords({'lon': ((self.da['lon']+180) % 360)-180})
        else:
            raise ValueError('Wrong mode. Should be either "360" or "180".')

        da = da.sortby(da['lon'])
        fd = ClimateField(da)
        return fd

    def from_np(self, time, lat, lon, value):
        ''' Load data from a `numpy.ndarray`.
        '''
        lat_da = xr.DataArray(lat, dims=['lat'], coords={'lat': lat})
        lon_da = xr.DataArray(lon, dims=['lon'], coords={'lon': lon})
        time_da = xr.DataArray(time, dims=['time'], coords={'time': time})
        da = xr.DataArray(
            value, dims=['time', 'lat', 'lon'],
            coords={'time': time_da, 'lat': lat_da, 'lon': lon_da},
        )
        fd = ClimateField(da)
        return fd

    def get_anom(self, ref_period=[1951, 1980]):
        ''' Get the anomaly against a reference time period.

        Args:
            ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
        '''
        try:
            var_ref = self.da.loc[str(ref_period[0]):str(ref_period[-1])]
        except:
            utils.p_warning(f'>>> The time axis does not overlap with the reference period {ref_period}; use its own time period as reference [{np.min(self.da.time):.2f}, {np.max(self.da.time):.2f}].')
            var_ref = self.da

        clim = var_ref.groupby('time.month').mean('time')
        da = self.da.groupby('time.month') - clim
        fd = ClimateField(da)

        return fd

    def center(self, ref_period=[1951, 1980]):
        ''' Center the climate field against a reference time period.

        Args:
            ref_period (tuple or list): the reference time period in the form or (start_yr, end_yr)
            time_name (str): name of the time dimention
        '''
        try:
            var_ref = self.da.loc[str(ref_period[0]):str(ref_period[-1])]
        except:
            utils.p_warning(f'>>> The time axis does not overlap with the reference period {ref_period}; use its own time period as reference [{np.min(self.da.time):.2f}, {np.max(self.da.time):.2f}].')
            var_ref = self.da

        clim = var_ref.mean('time')
        da = self.da - clim
        fd = ClimateField(da)

        return fd

    def load_nc(self, path, vn=None, time_name='time', lat_name='lat', lon_name='lon', load=False, return_ds=False, use_cftime=True, **kwargs):
        ''' Load the climate field from a netCDF file.

        Args:
            path (str): the path where to load data from
        '''
        ds = xr.open_dataset(path, use_cftime=use_cftime, **kwargs)
        if return_ds:
            return ds
        else:
            if len(ds.keys()) == 1:
                vn = list(ds.keys())[0]
            else:
                if vn is None:
                    raise ValueError('Variable name should be specified with `vn`.')

            da = ds[vn]
            fd = ClimateField(da)
            if time_name != 'time':
                fd.da = fd.da.rename({time_name: 'time'})
            if lat_name != 'lat':
                fd.da = fd.da.rename({lat_name: 'lat'})
            if lon_name != 'lon':
                fd.da = fd.da.rename({lon_name: 'lon'})

            if np.min(fd.da.lon) < 0:
                fd = fd.wrap_lon()

            if load: fd.da.load()
        
        return fd

    def to_nc(self, path, verbose=True, compress_params=None):
        ''' Convert the climate field to a netCDF file.

        Args:
            path (str): the path where to save
        '''
        # _comp_params = {'zlib': True, 'least_significant_digit': 2}
        _comp_params = {'zlib': True}

        encoding_dict = {}
        if compress_params is not None:
            _comp_params.update(compress_params)

        encoding_dict[self.da.name] = _comp_params

        try:
            dirpath = os.path.dirname(path)
            os.makedirs(dirpath, exist_ok=True)
        except:
            pass

        if os.path.exists(path):
            os.remove(path)

        self.da.to_netcdf(path, encoding=encoding_dict)
        if verbose: utils.p_success(f'>>> ClimateField.da["{self.da.name}"] saved to: {path}')

    def copy(self):
        ''' Make a deepcopy of the object. '''
        return copy.deepcopy(self)

    def rename(self, new_vn):
        ''' Rename the variable name of the climate field.'''
        da = self.da.rename(new_vn)
        fd = ClimateField(da)
        return fd

    def __add__(self, ref):
        ''' Add a reference.
        '''
        if isinstance(ref, ClimateField):
            da = self.da + ref.da
        elif isinstance(ref, float):
            da = self.da + ref
        elif isinstance(ref, int):
            da = self.da + ref
        else:
            raise ValueError('`ref` should be a `ClimateField` object or a float like value.')

        fd = ClimateField(da)
        return fd

    def __sub__(self, ref):
        ''' Substract a reference.
        '''
        if isinstance(ref, ClimateField):
            da = self.da - ref.da
        elif isinstance(ref, float):
            da = self.da - ref
        elif isinstance(ref, int):
            da = self.da - ref
        else:
            raise ValueError('`ref` should be a `ClimateField` object or a float like value.')

        fd = ClimateField(da)
        return fd

    def __mul__(self, ref):
        ''' Multiply a scaler.
        '''
        if isinstance(ref, ClimateField):
            da = self.da * ref.da
        elif isinstance(ref, float):
            da = self.da * ref
        elif isinstance(ref, int):
            da = self.da * ref
        else:
            raise ValueError('`ref` should be a `ClimateField` object or a float like value.')

        fd = ClimateField(da)
        return fd

    def __truediv__(self, ref):
        ''' Divide a scaler.
        '''
        if isinstance(ref, ClimateField):
            da = self.da / ref.da
        elif isinstance(ref, float):
            da = self.da / ref
        elif isinstance(ref, int):
            da = self.da / ref
        else:
            raise ValueError('`ref` should be a `ClimateField` object or a float like value.')

        fd = ClimateField(da)
        return fd
    
    def __floordiv__(self, ref):
        ''' Floor-divide a scaler.
        '''
        if isinstance(ref, ClimateField):
            da = self.da // ref.da
        elif isinstance(ref, float):
            da = self.da // ref
        elif isinstance(ref, int):
            da = self.da // ref
        else:
            raise ValueError('`ref` should be a `ClimateField` object or a float like value.')

        fd = ClimateField(da)
        return fd

    def compare(self, ref, timespan=None, stat='corr', interp_target='ref', interp=True):
        ''' Compare against a reference field.

        Args:
            ref (cfr.climate.ClimateField): the reference to compare against, assuming the first dimension to be time
            interp_target (str, optional): the direction to interpolate the fields:
            
                * 'ref': interpolate from `self` to `ref`
                * 'self': interpolate from `ref` to `self`

            stat (str): the statistics to calculate. Supported quantaties:

                * 'corr': correlation coefficient
                * 'R2': coefficient of determination
                * 'CE': coefficient of efficiency
        '''
        if interp:
            if interp_target == 'ref':
                fd_rg = self.regrid(ref.da.lat, ref.da.lon)
                ref_rg = ref.copy()
            elif interp_target == 'self':
                fd_rg = self.copy()
                ref_rg = ref.regrid(self.da.lat, self.da.lon)
        else:
            fd_rg = self.copy()
            ref_rg = self.copy()

        if timespan is not None:
            fd_rg = fd_rg[str(timespan[0]):str(timespan[-1])]
            ref_rg = ref_rg[str(timespan[0]):str(timespan[-1])]

        if len(fd_rg.da.lat.shape) == 1:
            fd_rg.da = xr.DataArray(
                fd_rg.da.values,
                coords={'time': ref_rg.da.time, 'lat': fd_rg.da.lat, 'lon': fd_rg.da.lon}
            )

        if stat == 'corr':
            stat_da = xr.corr(fd_rg.da, ref_rg.da, dim='time')
            stat_da = stat_da.expand_dims({'time': [1]})
            stat_da.name = stat
            stat_fd = ClimateField(stat_da)
            stat_fd.plot_kwargs = {
                'cmap': 'RdBu_r',
                'extend': 'neither',
                'levels': np.linspace(-1, 1, 21),
                'cbar_labels': np.linspace(-1, 1, 11),
                'cbar_title': r'$r$',
                'cbar_title_y': 1,
                'title': 'Correlation',
            }
        elif stat == 'R2':
            stat_da = xr.corr(fd_rg.da, ref_rg.da, dim='time')
            stat_da = stat_da.expand_dims({'time': [1]})
            stat_da.name = stat
            stat_fd = ClimateField(stat_da**2)
            stat_fd.plot_kwargs = {
                'cmap': 'Reds',
                'extend': 'neither',
                'levels': np.linspace(0, 1, 21),
                'cbar_labels': np.linspace(0, 1, 11),
                'cbar_title': r'$R^2$',
                'cbar_title_y': 1,
                'title': 'Coefficient of Determination',
            }
        elif stat == 'CE':
            ce = utils.coefficient_efficiency(ref_rg.da.values, fd_rg.da.values)
            stat_da = xr.DataArray(
                ce[np.newaxis],
                name=stat,
                coords={
                    'time': [1],
                    'lat': fd_rg.da.lat,
                    'lon': fd_rg.da.lon,
                })
            stat_fd = ClimateField(stat_da)
            stat_fd.plot_kwargs = {
                'cmap': 'RdBu_r',
                'extend': 'min',
                'levels': np.linspace(-1, 1, 21),
                'cbar_labels': np.linspace(-1, 1, 11),
                'cbar_title': r'$CE$',
                'cbar_title_y': 1,
                'title': 'Coefficient of Efficiency',
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

    def plot(self, **kwargs):
        ''' Plot a climate field at a time point.'''
        cbar_title = visual.make_lb(
            self.da.name,
            self.da.attrs['unit'] if 'unit' in self.da.attrs else None)

        cmap_dict = {
            'tas': 'RdBu_r',
            'tos': 'RdBu_r',
            'sst': 'RdBu_r',
            'pr': 'BrBG',
        }
        cmap = cmap_dict[self.da.name] if self.da.name in cmap_dict else 'viridis'

        _kwargs = {
            'cbar_title': cbar_title,
            'cmap': cmap,
        }

        if hasattr(self, 'plot_kwargs'):
            _kwargs.update(self.plot_kwargs)

        if 'title' not in kwargs and 'title' not in _kwargs:
            if len(self.da.dims) == 3:
                t_value = self.da.time.values[0]
            elif len(self.da.dims) == 2:
                t_value = self.da.time.values

            try:
                date_str = '-'.join(str(t_value).split('-')[:2])
            except:
                date_str = str(t_value)

            kwargs['title'] = f'{self.da.name}, {date_str}' 
            
        _kwargs.update(kwargs)
        if len(self.da.dims) == 3:
            vals = self.da.values[0]
        elif len(self.da.dims) == 2:
            vals = self.da.values

        fig, ax = visual.plot_field_map(vals, self.da.lat, self.da.lon, **_kwargs)

        return fig, ax

    def plotly_grid(self, site_lats=None, site_lons=None, **kwargs):
        ''' Plot the grid on an interactive map utilizing Plotly
        '''
        nlat, nlon = np.size(self.da.lat), np.size(self.da.lon)
        df = pd.DataFrame()
        n = 0
        for i in range(nlat):
            for j in range(nlon):
                lat = self.da.lat[i]
                lon = self.da.lon[j]
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
        if 'annualized' in self.da.attrs and self.da.attrs['annualized']==1:
            da = self.da
        else:
            da = utils.annualize(da=self.da, months=months)
            yrs = np.floor(utils.datetime2year_float(da.time.values))
            da = da.assign_coords({'time': [int(y) for y in yrs]})
            da.attrs['annualized'] = 1
        fd = ClimateField(da)
        return fd

    def regrid(self, lats, lons, periodic_lon=False):
        ''' Regrid the climate field.'''

        if len(self.da.lat.values.shape) == 1:
            lat_da = xr.DataArray(lats, dims=['lat'], coords={'lat': lats})
            lon_da = xr.DataArray(lons, dims=['lon'], coords={'lon': lons})
            da = self.da.interp(coords={'lat': lat_da, 'lon': lon_da})

        elif len(self.da.lon.values.shape) == 2:
            rgd_data, lats, lons = utils.regrid_field_curv_rect(
                self.da.values, self.da.lat.values, (self.da.lon.values+180)%360-180,
                lats=lats, lons=lons,
            )
            da = xr.DataArray(
                rgd_data, dims=['time', 'lat', 'lon'],
                coords={
                    'time': self.da.time,
                    'lat': lats,
                    'lon': lons,
                },
                name=self.da.name,
            )

        if periodic_lon:
            da[..., -1] = da[..., 0]

        fd = ClimateField(da)
        return fd

    def crop(self, lat_min=-90, lat_max=90, lon_min=0, lon_max=360):
        ''' Crop the climate field.'''

        mask_lat = (self.da.lat >= lat_min) & (self.da.lat <= lat_max)
        mask_lon = (self.da.lon >= lon_min) & (self.da.lon <= lon_max)

        da = self.da.sel({
                'lat': self.da.lat[mask_lat],
                'lon': self.da.lon[mask_lon],
            })
        fd = ClimateField(da)

        return  fd

    def geo_mean(self, lat_min=-90, lat_max=90, lon_min=0, lon_max=360):
        ''' Calculate the geographical mean value of the climate field. '''
        m = utils.geo_mean(self.da, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        ts = EnsTS(time=m['time'], value=m.values)
        return ts