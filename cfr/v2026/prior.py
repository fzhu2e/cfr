import numpy as np
import xarray as xr
import xesmf as xe
from tqdm import tqdm
from scipy.stats import norm
from copy import deepcopy

from . import psm
from . import utils

class PriorMember:
    '''A single prior member from which ensemble samples can be generated.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Climate model output dataset to serve as a prior member.
    '''
    def __init__(self, ds):
        if isinstance(ds, xr.DataArray): ds = ds.to_dataset()
        self.ds = ds
        self.vns = list(ds.data_vars)

    def gen_samples_Gaussian(self, local_sigma:dict, global_sigma:dict, nens:int=100, seed:int=0):
        ''' Generate samples following Gaussian

        Args:
            sigma (dict): Dictionary with standard deviation (sigma) for each variable.
            nens (int): Number of ensemble members to generate.
            seed (int): Seed for reproducibility.
        '''
        rng = np.random.default_rng(seed)
        self.samples = xr.Dataset()
        for vn in self.vns:
            mean = self.ds[vn].values
            samples_shape = (*mean.shape, nens)
            global_perturbation = norm.rvs(loc=0, scale=global_sigma[vn], size=nens, random_state=rng)
            local_perturbation = norm.rvs(loc=0, scale=local_sigma[vn], size=samples_shape, random_state=rng)
            samples = mean[..., np.newaxis] + global_perturbation + local_perturbation
            samples_da = xr.DataArray(samples, dims=(*self.ds[vn].dims, 'ens'), coords=self.ds[vn].coords)
            samples_da.attrs = self.ds[vn].attrs
            self.samples[vn] = samples_da

    def gen_samples_bootstrap(self, nens:int=30, clim_yrs:int=50, seed:int=0, dim='time'):
        ''' Generate samples from the prior pool

        Args:
        '''
        nt = len(self.ds[dim])
        pool_idx = list(range(nt))
        sample_list = []
        for i in range(nens):
            seed += 1
            rng = np.random.default_rng(seed)
            sample_idx = rng.choice(pool_idx, size=clim_yrs, replace=False)
            sample = self.ds.isel({dim: sample_idx}).mean(dim)
            sample_list.append(sample)

        samples = xr.concat(sample_list, dim=dim)
        self.samples = xr.Dataset(samples).rename({dim: 'ens'}).transpose(..., 'ens')


class Prior:
    '''Prior ensemble constructed from one or more PriorMember instances.

    Concatenates ensemble samples from prior members and provides methods
    for regridding, annualization, inflation, and state vector extraction.

    Parameters
    ----------
    members : PriorMember or list of PriorMember
        One or more prior members whose samples form the ensemble.
    '''
    def __init__(self, members):
        if not isinstance(members, list): members = [members]
        ds_list = []
        for m in members:
            if hasattr(m, 'samples'):
                ds_list.append(m.samples)
            else:
                if 'ens' not in m.ds.dims:
                    ds_list.append(m.ds.expand_dims({'ens': 1}))
                else:
                    ds_list.append(m.ds)

        self.ds = xr.concat(ds_list, dim='ens').transpose(..., 'ens')
        self.nens = len(self.ds.ens)
        self.nvar = len(self.ds.data_vars)

    def regrid(self, ds_template=None, dlat=1, dlon=1, verbose=False, **kws):
        if ds_template is None:
            self.ds_rgd = xr.Dataset()
            for vn in tqdm(self.ds.data_vars, desc=f'Regridding variables to {dlat}x{dlon}'):
                self.ds_rgd[vn] = self.ds.x[vn].x.regrid(dlat=dlat, dlon=dlon)
                if verbose: utils.p_success(f'>>> Prior.ds_rgd["{vn}"] created')
        else:
            _kws = {
                'periodic': True,
                'reuse_weights': True,
            }
            _kws.update(kws)
            regridder = xe.Regridder(self.ds, ds_template, method='bilinear', **kws)
            self.ds_rgd = regridder(self.ds)
            for vn in self.ds_rgd.data_vars:
                self.ds_rgd[vn].attrs = ds_template[vn].attrs.copy()
                if verbose: utils.p_success(f'>>> Prior.ds_rgd["{vn}"] created')

        self.ds_rgd = self.ds_rgd.transpose(..., 'ens')


    def annualize(self, months=list(range(1, 13))):
        return self.ds.sel(month=months).mean('month')

    def inflate(self, factor=2):
        self.ds_raw = self.ds.copy()
        ens_mean = self.ds.mean('ens')
        ens_pert = self.ds - self.ds.mean('ens')
        inflated_pert = ens_pert * factor
        self.ds = ens_mean + inflated_pert

    def copy(self):
        return deepcopy(self)


    @property
    def X(self):
        res = []
        for vn in self.ds.data_vars:
            res.append(self.ds[vn].values.reshape(-1, self.nens))

        res = np.concatenate(res, axis=0)
        return res

    # def get_Y(self, obs, **fwd_kws):
    #     self.obs_assim = obs.copy()
    #     lats = obs.df['lat'].values
    #     lons = obs.df['lon'].values
    #     pids = obs.df['pid'].values
    #     depths = obs.df['depth'].values
    #     if 'clean' in obs.df.columns: cleans = obs.df['clean'].values
    #     if 'species' in obs.df.columns: specs = obs.df['species'].values

    #     psms = obs.df['psm'].values
    #     psm_names = list(set(psms))

    #     pseudo_obs = np.empty((len(obs.df), self.nens))

    #     # Loop over PSM types (psm_names)
    #     for psm_name in psm_names:
    #         mask = psms == psm_name
    #         idx = np.where(mask)[0]
    #         if np.any(mask):
    #             lat_lon_pairs = xr.Dataset({
    #                 'lat': (('obs',), lats[mask]),
    #                 'lon': (('obs',), lons[mask]),
    #             })
    #             self.clim_proxy_locs = xr.Dataset()
    #             for vn in self.ds_rgd.data_vars:
    #                 filled_da = self.ds_rgd[vn].ffill(dim='lat').bfill(dim='lat').ffill(dim='lon').bfill(dim='lon')
    #                 self.clim_proxy_locs[vn] = filled_da.sel(
    #                     lat=lat_lon_pairs['lat'],
    #                     lon=lat_lon_pairs['lon'],
    #                     method='nearest',
    #                 ).transpose(..., 'ens')

    #             for i in tqdm(range(len(idx)), desc=f'>>> Looping over sites w/ PSM - {psm_name}'):
    #                 pid = pids[idx[i]]
    #                 lat = lats[idx[i]]
    #                 lon = lons[idx[i]]
    #                 depth = depths[idx[i]]
    #                 if np.isnan(depth): depth = 0

    #                 obs_meta = {
    #                     'pid': pid,
    #                     'lat': lat,
    #                     'lon': lon,
    #                     'depth': depth,
    #                 }
    #                 if 'clean' in obs.df.columns:
    #                     clean = cleans[idx[i]]
    #                     if np.isnan(clean): clean = 0
    #                     obs_meta['clean'] = clean

    #                 if 'species' in obs.df.columns:
    #                     species = specs[idx[i]]
    #                     if not isinstance(species, str): species = 'all'
    #                     obs_meta['species'] = species

    #                 mdl = psm.__dict__[psm_name](obs_meta, self.clim_proxy_locs.isel({'obs': i}))
    #                 _fwd_kws = {}
    #                 _fwd_kws[psm_name] = {}
    #                 if psm_name in fwd_kws:
    #                     _fwd_kws[psm_name].update(fwd_kws[psm_name])

    #                 res = mdl.forward(**_fwd_kws[psm_name])
    #                 if res is None:
    #                     utils.p_warning(f'>>> Dropping proxy: {pid}')
    #                     self.obs_assim.df = obs.df.drop(obs.df[obs.df['pid'] == pid].index)
    #                     pseudo_obs[idx[i]] = np.nan
    #                 else:
    #                     pseudo_obs[idx[i]] = res

    #     self.obs_assim.nobs = len(self.obs_assim.df)
    #     pseudo_obs = pseudo_obs[~np.isnan(pseudo_obs).any(axis=1)]
    #     self.Y = pseudo_obs
    #     self.obs_assim.df['Ym'] = self.Y.mean(axis=1)

    # def get_Y(self, obs, depth_name='z_t', nearest_kws={}, **fwd_kws):
    #     self.obs_assim = obs.copy()
    #     psm_names = set(obs.df['psm_name'])
    #     clim_vns = list({
    #         vn for psm_name in psm_names
    #         for vn in psm.__dict__[psm_name]().clim_vns
    #         if vn in self.ds.data_vars
    #     })

    #     self.ds_proxy_locs = xr.Dataset()
    #     _nearest_kws = {}
    #     _nearest_kws.update(nearest_kws)
    #     for vn in clim_vns:
    #         da = self.ds[vn].isel({depth_name: 0}) if depth_name in self.ds[vn].dims else self.ds[vn]
    #         self.ds_proxy_locs[vn] = da.x.nearest2d(
    #             lat=obs.df['lat'],
    #             lon=obs.df['lon'],
    #             **_nearest_kws
    #         ).transpose(..., 'ens')

    #     pseudo_obs = np.empty((len(obs.df), self.nens))
    #     for i, (pid, rec) in tqdm(enumerate(obs.records.items()), total=obs.nobs, desc='Looping over records'):
    #         self.obs_assim.records[pid].psm = mdl = psm.__dict__[rec.data.psm_name](rec)
    #         if 'month' in self.ds_proxy_locs.dims:
    #             mdl.clim = self.ds_proxy_locs.isel({'site': i}).sel(month=rec.data.seasonality).mean(dim='month')
    #         else:
    #             mdl.clim = self.ds_proxy_locs.isel({'site': i})

    #         for vn in clim_vns:
    #             if vn in mdl.clim and mdl.clim[vn].isnull().any():
    #                 raise ValueError(f'NaN values detected in input climate for forward modeling of: {pid}')

    #         _fwd_kws = {}
    #         _fwd_kws[rec.data.psm_name] = {}
    #         if rec.data.psm_name in fwd_kws:
    #             _fwd_kws[rec.data.psm_name].update(fwd_kws[rec.data.psm_name])
    #         mdl.forward(**_fwd_kws[rec.data.psm_name])
    #         if mdl.output is None:
    #             utils.p_warning(f'>>> Dropping proxy: {pid}')
    #             self.obs_assim.df = obs.df.drop(obs.df[obs.df['pid'] == pid].index)
    #             pseudo_obs[i] = np.nan
    #         else:
    #             pseudo_obs[i] = mdl.output

    #     self.obs_assim.nobs = len(self.obs_assim.df)
    #     pseudo_obs = pseudo_obs[~np.isnan(pseudo_obs).any(axis=1)]
    #     self.Y = pseudo_obs
    #     self.obs_assim.df['Ym'] = self.Y.mean(axis=1)
    def get_Y(self, obs):
        self.Y = np.empty((len(obs.df), self.nens))
        for idx, row in obs.df.iterrows():
            self.Y[idx, :] = row['pseudo']

    # def get_Y(self, obs, nearest_valid_radius=5, **fwd_kws):
    #     self.obs_assim = obs.copy()
    #     pseudo_obs = np.empty((len(obs.df), self.nens))

    #     psm_names = set(obs.df['psm_name'])
    #     clim_vns = list({
    #         vn for psm_name in psm_names
    #         for vn in psm.__dict__[psm_name]().clim_vns
    #         if vn in self.ds_rgd.data_vars
    #     })

    #     lat_lon_pairs = xr.Dataset({
    #         'lat': (('sites',), obs.df['lat'].values),
    #         'lon': (('sites',), obs.df['lon'].values),
    #     })
    #     self.ds_proxy_locs = xr.Dataset()
    #     for vn in clim_vns:
    #         # filled_da = self.ds_rgd[vn].ffill(dim='lon').bfill(dim='lon').ffill(dim='lat').bfill(dim='lat')
    #         # ds_proxy_locs[vn] = filled_da.sel(
    #         #     lat=lat_lon_pairs['lat'],
    #         #     lon=lat_lon_pairs['lon'],
    #         #     method='nearest',
    #         # ).transpose(..., 'ens')

    #         self.ds_proxy_locs[vn] = self.ds_rgd[vn].x.nearest2d(
    #             lat=lat_lon_pairs['lat'],
    #             lon=lat_lon_pairs['lon'],
    #             r=nearest_valid_radius,
    #             extra_dim='ens',
    #         ).transpose(..., 'ens')

    #     if 'sites' not in self.ds_proxy_locs.dims:
    #         self.ds_proxy_locs = self.ds_proxy_locs.expand_dims({'sites': [0]})

    #         # if ds_proxy_locs[vn].isnull().any():
    #         #     for idx in obs.df.index:
    #         #         if ds_proxy_locs[vn].sel(sites=idx).isnull().any():
    #         #             utils.p_warning(f"NaN detected for {vn}: {obs.df.iloc[idx][['pid', 'lat', 'lon']].values}")
    #         #             print(ds_proxy_locs[vn].sel(sites=idx).dims)
    #         #             print(ds_proxy_locs[vn].sel(sites=idx).values)
    #         #             utils.p_warning('------------------------------------')
    #         #     raise ValueError('Some of the nearest gridcell values are NaN.')

    #     # nearest_lats, nearest_lons = [], []
    #     for i, (pid, rec) in tqdm(enumerate(obs.records.items()), total=obs.nobs, desc='Looping over records'):
    #         # nearest_clim = self.ds_proxy_locs.isel({'sites': i}).sel(month=rec.data.seasonality).mean(dim='month')
    #         # nearest_lat = nearest_clim.lat.values.mean()
    #         # nearest_lon = nearest_clim.lon.values.mean()
    #         # nearest_lats.append(nearest_lat)
    #         # nearest_lons.append(nearest_lon)
    #         # rec.data.lat = nearest_lat
    #         # rec.data.lon = nearest_lon

    #         mdl = psm.__dict__[rec.data.psm_name](rec)
    #         if 'month' in self.ds_proxy_locs.dims:
    #             mdl.record.clim = self.ds_proxy_locs.isel({'sites': i}).sel(month=rec.data.seasonality).mean(dim='month')
    #         else:
    #             mdl.record.clim = self.ds_proxy_locs.isel({'sites': i})

    #         for vn in clim_vns:
    #             if mdl.record.clim[vn].isnull().any():
    #                 # print(i, ds_proxy_locs[vn].isel({'sites': i}))
    #                 # print(ds_proxy_locs[vn].isel({'sites': i, 'ens': 6}))
    #                 # print(ds_proxy_locs.sel(month=rec.data.seasonality).mean(dim='month')[vn].values[i])
    #                 # print(vn, rec.data.pid, rec.data.lat, rec.data.lon, rec.data.seasonality)
    #                 # print(mdl.record.clim[vn].values)
    #                 raise ValueError(f'NaN values detected in input climate for forward modeling of: {pid}')

    #         _fwd_kws = {}
    #         _fwd_kws[rec.data.psm_name] = {}
    #         if rec.data.psm_name in fwd_kws:
    #             _fwd_kws[rec.data.psm_name].update(fwd_kws[rec.data.psm_name])
    #         mdl.forward(**_fwd_kws[rec.data.psm_name])
    #         if mdl.output is None:
    #             utils.p_warning(f'>>> Dropping proxy: {pid}')
    #             self.obs_assim.df = obs.df.drop(obs.df[obs.df['pid'] == pid].index)
    #             pseudo_obs[i] = np.nan
    #         else:
    #             pseudo_obs[i] = mdl.output

    #         self.obs_assim.records[pid].psm = mdl  # for debugging purposes 

    #     self.obs_assim.nobs = len(self.obs_assim.df)
    #     # self.obs_assim.df['lat'] = nearest_lats
    #     # self.obs_assim.df['lon'] = nearest_lons
    #     pseudo_obs = pseudo_obs[~np.isnan(pseudo_obs).any(axis=1)]
    #     self.Y = pseudo_obs
    #     self.obs_assim.df['Ym'] = self.Y.mean(axis=1)

    # def get_dist(self, obs, s=1):
    #     # Extract grid latitudes and longitudes as 2D arrays
    #     lat_grid = self.ds[self.lat_name].values  # shape: (nlat, nlon)
    #     lon_grid = self.ds[self.lon_name].values  # shape: (nlat, nlon)

    #     if lat_grid.ndim == 1 and lon_grid.ndim == 1:
    #         # If lat and lon are 1D, create a meshgrid
    #         lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    #     # Flatten the grid arrays to 1D
    #     lat_grid_flat = lat_grid.ravel()  # shape: (nlat * nlon,)
    #     lon_grid_flat = lon_grid.ravel()  # shape: (nlat * nlon,)

    #     # Get the observation lat/lon as a 2D array
    #     lats2 = obs.df['lat'].values  # shape: (nobs,)
    #     lons2 = obs.df['lon'].values  # shape: (nobs,)

    #     # Broadcast the grid cells to all observation points
    #     lats1 = np.repeat(lat_grid_flat, obs.nobs)  # shape: (nlat * nlon * nobs,)
    #     lons1 = np.repeat(lon_grid_flat, obs.nobs)  # shape: (nlat * nlon * nobs,)

    #     # Repeat observation points for every grid point
    #     lats2 = np.tile(lats2, len(lat_grid_flat))  # shape: (nlat * nlon * nobs,)
    #     lons2 = np.tile(lons2, len(lon_grid_flat))  # shape: (nlat * nlon * nobs,)

    #     dist0 = utils.gcd(lats1, lons1, lats2, lons2).reshape((-1, obs.nobs))

    #     if hasattr(self, 'nz'):
    #         # 3D localization
    #         s = (np.ones(self.nz)*s).reshape(-1, 1, 1)
    #         dist1 = (dist0[None, :, :] * s).reshape((-1, obs.nobs))
    #         self.dist = dist1[np.newaxis, :].repeat(self.nvar, axis=0).reshape(-1, obs.nobs)
    #     else:
    #         self.dist = dist0[None, :, :].repeat(self.nvar, axis=0).reshape(-1, obs.nobs)
    def set_hcoords(self, hcoords:dict):
        self.hcoords = hcoords

    def get_dist(self, obs, vertical_dims=['z_t', 'lev'], s=1):
        all_dists = []
        for vn in tqdm(self.ds.data_vars, desc='Processing variables'):
            lat_name, lon_name = self.hcoords[vn] if hasattr(self, 'hcoords') else ('lat', 'lon')
            lats, lons = self.ds[vn].coords[lat_name].values, self.ds[vn].coords[lon_name].values
            if lat_name not in self.ds[vn].dims:
                if len(lats.shape) == 2:
                    # POP-like curvilinear grid
                    lats = lats.ravel()
                    lons = lons.ravel()
            else:
                # regular lat-lon grid
                lons, lats = np.meshgrid(lons, lats)
                lats = lats.ravel()
                lons = lons.ravel()

            lats1 = np.repeat(lats, obs.nobs)
            lons1 = np.repeat(lons, obs.nobs)
            lats2 = np.tile(obs.df['lat'].values, len(lats))
            lons2 = np.tile(obs.df['lon'].values, len(lons))

            dist = utils.gcd(lats1, lons1, lats2, lons2).reshape((-1, obs.nobs))

            vdims = list(set(self.ds[vn].dims) & set(vertical_dims))
            if len(vdims) == 1:
                nz = len(self.ds[vn][vdims[0]])
                s_arr = np.reshape(np.ones(nz) * s, (-1, 1, 1))
                dist = (dist[None, :, :] * s_arr).reshape((-1, obs.nobs))

            all_dists.append(dist)
        
        self.dist = np.concatenate(all_dists, axis=0)

