import os
from datetime import datetime
import cftime
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
    '''
    def __init__(self, da=None):
        ''' Initialize a ClimateField object with a `xarray.DataArray`.

        Args:
            da (xarray.DataArray): the gridded data array.
        '''
        self.da = da

    def __getitem__(self, key):
        ''' This makes the object subscriptable. '''
        if type(key) is int:
            # one index
            da = self.da[key]
        elif type(key) is str:
            # one timestamp
            # key = slice(key, key, None)
            # da = self.da.sel({'time': key})
            # if len(da.values) == 0:
            #     mask = (self.da['time']>=float(key.start)) & (self.da['time']<=float(key.stop))
            #     da = self.da.sel({'time': mask})
            try:
                mask = (utils.datetime2year_float(self.da['time'].values)>=float(key)) & (utils.datetime2year_float(self.da['time'].values)<float(key)+1)
            except:
                mask = (self.da['time'].values>=float(key)) & (self.da['time'].values<float(key)+1)

            da = self.da.sel({'time': mask})

        elif type(key) is list:
            # multiple discrete items
            if type(key[0]) is int:
                # index
                da = self.da[key]
            elif type(key[0]) is str:
                # timestamp
                da_list = []
                for k in key:
                    da_list.append(self[k].da)
                da = xr.concat(da_list, dim='time')
                
        elif type(key) is slice:
            if key.start is not None:
                dtype = type(key.start)
            else:
                key.start = self.da['time'][0]

            if key.stop is not None:
                dtype = type(key.stop)
            else:
                key.stop = self.da['time'][-1]

            if dtype is int:
                # index
                da = self.da[key]
            elif dtype is str:
                # timestamp
                try:
                    mask = (utils.datetime2year_float(self.da['time'].values)>=float(key.start)) & (utils.datetime2year_float(self.da['time'].values)<float(key.stop))
                except:
                    mask = (self.da['time'].values>=float(key.start)) & (self.da['time'].values<float(key.stop))

                da = self.da.sel({'time': mask})
                if key.step is not None:
                    da = da[::int(key.step)]

            else:
                raise TypeError('Wrong type for key!')
        else:
            raise TypeError('Wrong type for key!')

        fd = ClimateField(da)

        return fd

    def __len__(self):
        return len(self.time)

    def wrap_lon(self, mode='360'):
        ''' Convert the longitude values

        Args:
            mode (str): if '360', convert the longitude values from the range (-180, 180) to (0, 360);
                if '180', convert the longitude values from the range (0, 360) to (-180, 180);
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

    def fetch(self, name=None, **load_nc_kws):
        ''' Fetch a gridded climate field from cloud

        Args:
            name (str): a predefined name, or an URL starting with "http", or a local file path.
                If not set, the method will return hints of available predefined names.
            load_nc_kws (dict): the dictionary of keyword arguments for loading a netCDF file.
        '''
        url_dict = utils.climfd_url_dict

        if name is None:
            utils.p_warning(f'>>> Choose one from the supported entries:')
            for k in url_dict.keys():
                utils.p_warning(f'- {k}')
            return None

        if name in url_dict:
            url = url_dict[name]
        else:
            url = name

        if url[:4] == 'http':
            # cloud
            os.makedirs('./data', exist_ok=True)
            fpath = f'./data/{os.path.basename(url)}'
            if os.path.exists(fpath):
                utils.p_hint(f'>>> The target file seems existed at: {fpath} . Loading from it instead of downloading ...')
            else:
                utils.download(url, fpath)
                utils.p_success(f'>>> Downloaded file saved at: {fpath}')
        else:
            # local
            fpath = url

        fd = self.load_nc(fpath, **load_nc_kws)

        return fd

    def from_np(self, time, lat, lon, value):
        ''' Load data from a `numpy.ndarray`.

        Args:
            time (array-like): the array of the time axis.
            lat (array-like): the array of the lat axis.
            lon (array-like): the array of the lon axis.
            value (array-like): the array of the values.
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
            path (str): the path where to load data from.
            vn (str): the variable name to load.
            time_name (str): the name for the time axis. Defaults to 'time'.
            lat_name (str): the name for the lat axis. Defaults to 'lat'.
            lon_name (str): the name for the lon axis. Defaults to 'lon'.
            load (bool): if True, the netCDF file will be loaded into the memory; if False, will take the advantage of lazy loading. Defaults to `False`.
            return_ds (bool): if True, will return a `xarray.Dataset` object instead of a `ClimateField` object. Defaults to `False`.
            use_cftime (bool): if True, use the cftime convention. Defaults to `True`.
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
            verbose (bool, optional): print verbose information. Defaults to False.
            compress_params (dict): the paramters for compression when storing the reconstruction results to netCDF files.
        '''
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
        ''' Rename the variable name of the climate field.

        Args:
            new_vn (str): the new variable name.
        '''
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
            timespan (tuple or list): the timespan over which to compare two ClimateField objects.
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
            ref_rg = ref.copy()

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
        ''' Get the EOF analysis result of the ClimateField object

        Args:
            n (int): perform EOF analysis and return the first `n` modes.
            time_period (tuple or list): the timespan over which to perfom the EOF analysis.
            verbose (bool, optional): print verbose information. Defaults to False.
            flip (bool, optional): flip the sign of the field values. Defaults to False.
        '''
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
        ''' Plot the EOF analysis result

        Args:
            n (int): plot the `n`-th mode.
            eof_title (str): the subplot title for the mode field.
            pc_title (str): the subplot title for the PC time series.
        '''
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
        ''' Plot a climate field at a time point.

        See also:
            cfr.visual.plot_field_map : Visualize a field on a map.
        '''
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
            
        # if len(set(np.diff(self.da.lon))) > 1:
        #     # the case when longitudes cross 0 degree
        #     fd_plot = self.wrap_lon(mode='180')
        #     kwargs['central_longitude'] = 0
        # else:
            # fd_plot = self

        fd_plot = self

        _kwargs.update(kwargs)
            
        if len(fd_plot.da.dims) == 3:
            vals = fd_plot.da.values[0]
        elif len(fd_plot.da.dims) == 2:
            vals = fd_plot.da.values

        fig, ax = visual.plot_field_map(vals, fd_plot.da.lat, fd_plot.da.lon, **_kwargs)

        return fig, ax

    def plotly_grid(self, site_lats=None, site_lons=None, **kwargs):
        ''' Plot the grid on an interactive map utilizing Plotly

        Args:
            site_lats (list): a list of the latitudes of the sites to plot
            site_lons (list): a list of the longitudes of the sites to plot
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
        ''' Crop the climate field based on the range of latitude and longitude.

        Note that in cases when the crop range is crossing the 0 degree of longitude, `lon_min` should be less than 0.

        Args:
            lat_min (float): the lower bound of latitude to crop.
            lat_max (float): the upper bound of latitude to crop.
            lon_min (float): the lower bound of longitude to crop.
            lon_max (float): the upper bound of longitude to crop.
        '''

        mask_lat = (self.da.lat >= lat_min) & (self.da.lat <= lat_max)
        if lon_min >= 0:
            mask_lon = (self.da.lon >= lon_min) & (self.da.lon <= lon_max)
        else:
            # the case when longitudes in mode [-180, 180]
            lon_min = np.mod(lon_min, 360)
            lon_max = np.mod(lon_max, 360)
            mask_lon = (self.da.lon >= lon_min) | (self.da.lon <= lon_max)

        da = self.da.sel({
                'lat': self.da.lat[mask_lat],
                'lon': self.da.lon[mask_lon],
            })
        fd = ClimateField(da)

        return  fd

    def geo_mean(self, lat_min=-90, lat_max=90, lon_min=0, lon_max=360):
        ''' Calculate the geographical mean value of the climate field.

        Args:
            lat_min (float): the lower bound of latitude for the calculation.
            lat_max (float): the upper bound of latitude for the calculation.
            lon_min (float): the lower bound of longitude for the calculation.
            lon_max (float): the upper bound of longitude for the calculation.
        '''
        fdc = self.crop(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        wgts = np.cos(np.deg2rad(fdc.da['lat']))
        m = fdc.da.weighted(wgts).mean(('lon', 'lat'))
        ts = EnsTS(time=m['time'].values, value=m.values)
        return ts

    def index(self, name):
        ''' Calculate the predefined indices.

        Args:
            name (str): the predefined index name; supports the below:

                * 'nino3.4'
                * 'nino1+2'
                * 'nino3'
                * 'nino4'
                * 'tpi'
                * 'wp'
                * 'dmi'
                * 'iobw'
        '''
        if name == 'gm': 
            return self.geo_mean()
        elif name == 'nhm': 
            return self.geo_mean(lat_min=0)
        elif name == 'shm':
            return self.geo_mean(lat_max=0)
        elif name == 'nino3.4':
            return self.geo_mean(lat_min=-5, lat_max=5, lon_min=np.mod(-170, 360), lon_max=np.mod(-120, 360))
        elif name == 'nino1+2':
            return self.geo_mean(lat_min=-10, lat_max=10, lon_min=np.mod(-90, 360), lon_max=np.mod(-80, 360))
        elif name == 'nino3':
            return self.geo_mean(lat_min=-5, lat_max=5, lon_min=np.mod(-150, 360), lon_max=np.mod(-90, 360))
        elif name == 'nino4':
            return self.geo_mean(lat_min=-5, lat_max=5, lon_min=np.mod(160, 360), lon_max=np.mod(-150, 360))
        elif name == 'wpi':
            # Western Pacific Index
            return self.geo_mean(lat_min=-10, lat_max=10, lon_min=np.mod(120, 360), lon_max=np.mod(150, 360))
        elif name == 'tpi':
            # Tri-Pole Index
            v1 = self.geo_mean(lat_min=25, lat_max=45, lon_min=np.mod(140, 360), lon_max=np.mod(-145, 360))
            v2 = self.geo_mean(lat_min=-10, lat_max=10, lon_min=np.mod(170, 360), lon_max=np.mod(-90, 360))
            v3 = self.geo_mean(lat_min=-50, lat_max=-15, lon_min=np.mod(150, 360), lon_max=np.mod(-160, 360))
            return v2 - (v1 + v3)/2
        elif name == 'dmi':
            # Indian Ocean Dipole Mode
            dmiw = self.geo_mean(lat_min=-10, lat_max=10, lon_min=50, lon_max=70)
            dmie = self.geo_mean(lat_min=-10, lat_max=0, lon_min=90, lon_max=110)
            return dmiw - dmie
        elif name == 'iobw':
            # Indian Ocean Basin Wide
            return self.geo_mean(lat_min=-20, lat_max=20, lon_min=40 ,lon_max=100)
        else:
            raise ValueError('Wrong index name.')