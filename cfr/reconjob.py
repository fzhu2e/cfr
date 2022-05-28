import os
import copy
import time
from shutil import ReadError
import numpy as np
import yaml
from tqdm import tqdm
import pandas as pd
import random
from .climate import ClimateField
from .proxy import ProxyDatabase, ProxyRecord
try:
    from . import psm
except:
    pass
import xarray as xr
from . import utils, da
from .utils import (
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
)

import pprint
pp = pprint.PrettyPrinter()

class ReconJob:
    ''' Reconstruction Job
    '''
    def __init__(self, configs=None, verbose=False):
        ''' Initialize a reconstruction job

        Args:
            configs (dict, optional): a dictionary of configurations. Defaults to None.
        '''
        self.configs = {} if configs is None else configs
        if verbose:
            p_header(f'>>> job.configs:')
            pp.pprint(self.configs)

    def io_cfg(self, k, v, default=None, verbose=False):
        ''' Add-to or read-from configurations
        '''
        if v is not None:
            self.configs.update({k: v})
            if verbose: p_header(f'>>> job.configs["{k}"] = {v}')
        elif k in self.configs:
            v = self.configs[k]
        elif default is not None:
            v = default
            self.configs.update({k: v})
            if verbose: p_header(f'>>> job.configs["{k}"] = {v}')
        else:
            raise ValueError(f'{k} not properly set.')

        return v

    def write_cfg(self, k, v, verbose=False):
        self.configs.update({k: v})
        if verbose: p_header(f'>>> job.configs["{k}"] = {v}')

    def mark_pids(self, verbose=False):
        self.write_cfg('pids', self.proxydb.pids, verbose=verbose)

    def erase_cfg(self, keys, verbose=False):
        for k in keys:
            self.configs.pop(k)
            if verbose: p_success(f'>>> job.configs["{k}"] dropped')

    def save_cfg(self, save_dirpath=None, verbose=False):
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = os.path.join(save_dirpath, 'configs.yml') 
        with open(save_path, 'w') as f:
            yaml.dump(self.configs, f)

        if verbose: p_success(f'>>> job.configs saved to: {save_path}')


    def copy(self):
        return copy.deepcopy(self)

    def load_proxydb(self, path=None, verbose=False, **kwargs):
        path = self.io_cfg('proxydb_path', path, verbose=verbose)

        _, ext =  os.path.splitext(path)
        if ext.lower() == '.pkl':
            df = pd.read_pickle(path)
        else:
            raise ReadError(f'The extention of the file [{ext}] is not supported. Support list: [.pkl, ] .')

        self.proxydb = ProxyDatabase().from_df(df, **kwargs)
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records loaded')
            p_success(f'>>> job.proxydb created')

    def filter_proxydb(self, *args, inplace=True, verbose=False, **kwargs):
        if inplace:
            self.proxydb = self.proxydb.filter(*args, **kwargs)

            if verbose:
                p_success(f'>>> {self.proxydb.nrec} records remaining')
                p_success(f'>>> job.proxydb updated')

        else:
            pdb = self.proxydb.filter(*args, **kwargs)
            return pdb

    def annualize_proxydb(self, months=None, ptypes=None, inplace=True, verbose=False, **kwargs):
        months = self.io_cfg('annualize_proxydb_months', months, default=list(range(1, 13)), verbose=verbose)
        ptypes = self.io_cfg('annualize_proxydb_ptypes', ptypes, verbose=verbose)

        if ptypes is None:
            if inplace:
                self.proxydb = self.proxydb.annualize(months=months, **kwargs)
            else:
                pdb = self.proxydb.annualize(months=months, **kwargs)
                return pdb
        else:
            pdb_filtered = self.proxydb.filter(by='ptype', keys=ptypes)
            pdb_ann = pdb_filtered.annualize(months=months, **kwargs)
            pdb_left = self.proxydb - pdb_filtered

            if inplace:
                self.proxydb = pdb_ann + pdb_left
            else:
                pdb = pdb_ann + pdb_left
                return pdb
                
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records remaining')
            p_success(f'>>> job.proxydb updated')

    def clear_proxydb_tags(self, verbose=False):
        for pid, pobj in self.proxydb.records.items():
            pobj.tags = []

        if verbose:
            p_success(f'>>> job.proxydb updated with tags cleared')

    def split_proxydb(self, tag='calibrated', assim_frac=None, seed=0, verbose=False):
        assim_frac = self.io_cfg('proxy_assim_frac', assim_frac, default=0.75, verbose=verbose)
        target_pdb = self.proxydb.filter(by='tag', keys=[tag])

        nrec_assim = int(target_pdb.nrec * assim_frac)
        random.seed(seed)
        idx_assim = random.sample(range(target_pdb.nrec), nrec_assim)
        idx_eval = list(set(range(target_pdb.nrec)) - set(idx_assim))

        idx = 0
        for pid, pobj in target_pdb.records.items():
            if idx in idx_assim:
                pobj.tags.add('assim')
            elif idx in idx_eval:
                pobj.tags.add('eval')
            idx += 1

        if verbose:
            p_success(f'>>> {target_pdb.nrec_tags(keys=["assim"])} records tagged "assim"')
            p_success(f'>>> {target_pdb.nrec_tags(keys=["eval"])} records tagged "eval"')

    def load_clim(self, tag, path_dict=None, rename_dict=None, anom_period=None, time_name=None, lon_name=None, verbose=False):
        path_dict = self.io_cfg(f'{tag}_path', path_dict, verbose=verbose)
        if rename_dict is not None: rename_dict = self.io_cfg(f'{tag}_rename_dict', rename_dict, verbose=verbose)
        anom_period = self.io_cfg(f'{tag}_anom_period', anom_period, verbose=verbose)
        lon_name = self.io_cfg(f'{tag}_lon_name', lon_name, default='lon', verbose=verbose)
        time_name = self.io_cfg(f'{tag}_time_name', time_name, default='time', verbose=verbose)

        self.__dict__[tag] = {}
        for vn, path in path_dict.items():
            if rename_dict is None:
                vn_in_file = vn
            else:
                vn_in_file = rename_dict[vn]

            if anom_period == 'null':
                self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file, time_name=time_name).wrap_lon(lon_name=lon_name)
            else:
                self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file, time_name=time_name).get_anom(ref_period=anom_period).wrap_lon(lon_name=lon_name)

            self.__dict__[tag][vn].da.name = vn

        if verbose:
            p_success(f'>>> {tag} variables {list(self.__dict__[tag].keys())} loaded')
            p_success(f'>>> job.{tag} created')

    def annualize_clim(self, tag, verbose=False, months=None):
        months = self.io_cfg('prior_annualize_months', months, default=list(range(1, 13)), verbose=verbose)

        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'>>> Processing {vn} ...')
            self.__dict__[tag][vn] = fd.annualize(months=months)

        if verbose:
            p_success(f'>>> job.{tag} updated')

    def regrid_clim(self, tag, verbose=False, lats=None, lons=None, nlat=None, nlon=None, periodic_lon=True):
        if lats is None and lons is None:
            nlat = self.io_cfg('prior_regrid_nlat', nlat, default=42, verbose=verbose)
            nlon = self.io_cfg('prior_regrid_nlon', nlon, default=63, verbose=verbose)
            lats = np.linspace(-90, 90, nlat)
            lons = np.linspace(0, 360, nlon)
        else:
            lats = self.io_cfg('prior_regrid_lats', lats, verbose=verbose)
            lons = self.io_cfg('prior_regrid_lons', lons, verbose=verbose)

        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'>>> Processing {vn} ...')
            self.__dict__[tag][vn] = fd.regrid(lats=lats, lons=lons, periodic_lon=periodic_lon)
    
    def crop_clim(self, tag, lat_min=None, lat_max=None, lon_min=None, lon_max=None, verbose=False):
        lat_min = self.io_cfg(f'prior_lat_min', lat_min, default=-90, verbose=verbose)
        lat_max = self.io_cfg(f'prior_lat_max', lat_max, default=90, verbose=verbose)
        lon_min = self.io_cfg(f'prior_lon_min', lon_min, default=0, verbose=verbose)
        lon_max = self.io_cfg(f'prior_lon_max', lon_max, default=360, verbose=verbose)

        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'>>> Processing {vn} ...')
            self.__dict__[tag][vn] = fd.crop(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        
        
    def calib_psms(self, ptype_psm_dict=None, ptype_season_dict=None, calib_period=None, verbose=False, **kwargs):
        ptype_psm_dict = self.io_cfg(
            'ptype_psm_dict', ptype_psm_dict,
            default={ptype: 'Linear' for ptype in set(self.proxydb.type_list)},
            verbose=verbose)

        ptype_season_dict = self.io_cfg(
            'ptype_season_dict', ptype_season_dict,
            default={ptype: list(range(13)) for ptype in set(self.proxydb.type_list)},
            verbose=verbose)

        calib_period = self.io_cfg(
            'psm_calib_period', calib_period,
            default=[1850, 2015],
            verbose=verbose)

        for pid, pobj in tqdm(self.proxydb.records.items(), total=self.proxydb.nrec, desc='Calibrating the PSMs:'):
            psm_name = ptype_psm_dict[pobj.ptype]

            for vn in psm.__dict__[psm_name]().climate_required:
                pobj.get_clim(self.obs[vn], tag='obs')

            pobj.psm = psm.__dict__[psm_name](pobj)
            if psm_name in ['WhiteNoise']:
                pobj.psm.calibrate(**kwargs)
            elif psm_name == 'Bilinear':
                pobj.psm.calibrate(
                    season_list1=ptype_season_dict[pobj.ptype],
                    season_list2=ptype_season_dict[pobj.ptype], calib_period=calib_period)
            else:
                pobj.psm.calibrate(season_list=ptype_season_dict[pobj.ptype], calib_period=calib_period)

        # give the calibrated records a tag
        for pid, pobj in self.proxydb.records.items():
            if pobj.psm.calib_details is None:
                if verbose: p_warning(f'>>> The PSM for {pid} failed to calibrate.')
            else:
                self.proxydb.records[pid].tags.add('calibrated')
                self.proxydb.records[pid].R = pobj.psm.calib_details['PSMmse']  # assign obs err matrix

        if verbose:
            p_success(f'>>> {self.proxydb.nrec_tags("calibrated")} records tagged "calibrated" with ProxyRecord.psm created')

    def forward_psms(self, verbose=False, **kwargs):
        pdb_calib = self.proxydb.filter(by='tag', keys={'calibrated'})
            
        for pid, pobj in tqdm(pdb_calib.records.items(), total=pdb_calib.nrec, desc='Forwarding the PSMs:'):
            for vn in pobj.psm.climate_required:
                pobj.get_clim(self.prior[vn], tag='model')

            pobj.pseudo = pobj.psm.forward(**kwargs)

        if verbose:
            p_success(f'>>> ProxyRecord.pseudo created for {pdb_calib.nrec} records')

    def run_da(self, recon_period=None, recon_loc_rad=None, recon_timescale=None, nens=None, seed=0, nproc=None, verbose=False, debug=False):
        recon_period = self.io_cfg('recon_period', recon_period, default=[0, 2000], verbose=verbose)
        recon_loc_rad = self.io_cfg('recon_loc_rad', recon_loc_rad, default=25000, verbose=verbose)  # unit: km
        recon_timescale = self.io_cfg('recon_timescale', recon_timescale, default=1, verbose=verbose)  # unit: yr
        nens = self.io_cfg('nens', nens, default=100, verbose=verbose)

        recon_yrs = np.arange(recon_period[0], recon_period[-1]+1)

        solver = da.EnKF(self.prior, self.proxydb, nens=nens,seed=seed)
        solver.run(
            recon_yrs=recon_yrs,
            recon_loc_rad=recon_loc_rad,
            recon_timescale=recon_timescale,
            verbose=verbose, debug=debug)

        self.recon_fields = solver.recon_fields
        if verbose: p_success(f'>>> job.recon_fields created')

    def run_mc(self, recon_period=None, recon_loc_rad=None, recon_timescale=None, output_full_ens=None, save_dtype=np.float32,
               recon_seeds=None, assim_frac=None, save_dirpath=None, compress_params=None, verbose=False):

        t_s = time.time()
        recon_period = self.io_cfg('recon_period', recon_period, default=[0, 2000], verbose=verbose)
        recon_loc_rad = self.io_cfg('recon_loc_rad', recon_loc_rad, default=25000, verbose=verbose)  # unit: km
        recon_timescale = self.io_cfg('recon_timescale', recon_timescale, default=1, verbose=verbose)  # unit: yr
        recon_seeds = self.io_cfg('recon_seeds', recon_seeds, default=np.arange(0, 20), verbose=verbose)
        assim_frac = self.io_cfg('assim_frac', assim_frac, default=0.75, verbose=verbose)
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
        os.makedirs(save_dirpath, exist_ok=True)
        compress_params = self.io_cfg('compress_params', compress_params, default={'zlib': True, 'least_significant_digit': 1}, verbose=verbose)
        output_full_ens = self.io_cfg('output_full_ens', output_full_ens, default=False, verbose=verbose)

        for seed in recon_seeds:
            if verbose: p_header(f'>>> seed: {seed} | max: {recon_seeds[-1]}')

            self.split_proxydb(seed=seed, assim_frac=assim_frac, verbose=False)
            self.run_da(recon_period=recon_period, recon_loc_rad=recon_loc_rad,
                        recon_timescale=recon_timescale, seed=seed, verbose=False)

            recon_savepath = os.path.join(save_dirpath, f'job_r{seed:02d}_recon.nc')
            self.save_recon(recon_savepath, compress_params=compress_params,
                            verbose=verbose, output_full_ens=output_full_ens, dtype=save_dtype)

        t_e = time.time()
        t_used = t_e - t_s
        p_success(f'>>> DONE! Total time used: {t_used/60:.2f} mins.')


    def save(self, save_dirpath=None, filename='job.pkl', verbose=False):
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
        os.makedirs(save_dirpath, exist_ok=True)
        pd.to_pickle(self, os.path.join(save_dirpath, filename))
        if verbose: p_success(f'>>> job saved to: {save_dirpath}')
    
    def save_recon(self, save_path, compress_params=None, verbose=False, output_full_ens=False,
                   output_indices=None, dtype=np.float32):

        compress_params = self.io_cfg(
            'compress_params', compress_params,
            default={'zlib': True, 'least_significant_digit': 1},
            verbose=False)

        output_indices = self.io_cfg(
            'output_indices', output_indices,
            default=['gm', 'nhm', 'shm', 'nino3.4'],
            verbose=False)

        ds = xr.Dataset()
        year = np.arange(self.configs['recon_period'][0], self.configs['recon_period'][1]+1)
        for vn, fd in self.recon_fields.items():
            nyr, nens, nlat, nlon = np.shape(fd)
            da = xr.DataArray(fd,
                coords={
                    'year': year,
                    'ens': np.arange(nens),
                    'lat': self.prior[vn].lat,
                    'lon': self.prior[vn].lon,
                })

            # output indices
            if 'gm' in output_indices: ds[f'{vn}_gm'] = utils.geo_mean(da)
            if 'nhm' in output_indices: ds[f'{vn}_shm'] = utils.geo_mean(da, lat_min=0)
            if 'shm' in output_indices: ds[f'{vn}_shm'] = utils.geo_mean(da, lat_max=0)
            if vn in ['tas', 'sst']:
                if 'nino3.4' in output_indices:
                    ds['nino3.4'] = utils.geo_mean(da, lat_min=-5, lat_max=5, lon_min=np.mod(-170, 360), lon_max=np.mod(-120, 360))
                if 'nino1+2' in output_indices:
                    ds['nino1+2'] = utils.geo_mean(da, lat_min=-10, lat_max=10, lon_min=np.mod(-90, 360), lon_max=np.mod(-80, 360))
                if 'nino3' in output_indices:
                    ds['nino3'] = utils.geo_mean(da, lat_min=-5, lat_max=5, lon_min=np.mod(-150, 360), lon_max=np.mod(-90, 360))
                if 'nino4' in output_indices:
                    ds['nino4'] = utils.geo_mean(da, lat_min=-5, lat_max=5, lon_min=np.mod(160, 360), lon_max=np.mod(-150, 360))
                if 'wpi' in output_indices:
                    ds['wpi'] = utils.geo_mean(da, lat_min=-10, lat_max=10, lon_min=np.mod(120, 360), lon_max=np.mod(150, 360))
                if 'tpi' in output_indices:
                    v1 = utils.geo_mean(da, lat_min=25, lat_max=45, lon_min=np.mod(140, 360), lon_max=np.mod(-145, 360))
                    v2 = utils.geo_mean(da, lat_min=-10, lat_max=10, lon_min=np.mod(170, 360), lon_max=np.mod(-90, 360))
                    v3 = utils.geo_mean(da, lat_min=-50, lat_max=-15, lon_min=np.mod(150, 360), lon_max=np.mod(-160, 360))
                    ds['tpi'] = v2 - (v1 + v3)/2

            if not output_full_ens: da = da.mean(dim='ens')
            ds[vn] = da

        encoding_dict = {}
        for k in self.recon_fields.keys():
            encoding_dict[k] = compress_params

        # mark the pids being assimilated
        pdb_assim = self.proxydb.filter(by='tag', keys=['assim'])
        pdb_eval = self.proxydb.filter(by='tag', keys=['eval'])
        ds.attrs = {
            'pids_assim': pdb_assim.pids,
            'pids_eval': pdb_eval.pids,
        }
        ds.to_netcdf(save_path, encoding=encoding_dict)

        if verbose: p_success(f'>>> Reconstructed fields saved to: {save_path}')

    def prepare_cfg(self, cfg_path, seeds=None, verbose=False):
        t_s = time.time()

        with open(cfg_path, 'r') as f:
            self.configs = yaml.safe_load(f)

        if seeds is not None:
            self.configs['recon_seeds'] = seeds
            p_header(f'>>> Settings seeds: {seeds}')

        if verbose:
            p_success(f'>>> job.configs loaded')
            pp.pprint(self.configs)

        self.load_proxydb(self.configs['proxydb_path'], verbose=verbose)
        if 'pids' in self.configs:
            self.filter_proxydb(by='pid', keys=self.configs['pids'], verbose=verbose)
        self.annualize_proxydb(
            months=self.configs['annualize_proxydb_months'],
            ptypes=self.configs['annualize_proxydb_ptypes'])

        if 'prior_rename_dict' in self.configs:
            prior_rename_dict = self.configs['prior_rename_dict']
        else:
            prior_rename_dict = None

        if 'obs_rename_dict' in self.configs:
            obs_rename_dict = self.configs['obs_rename_dict']
        else:
            obs_rename_dict = None

        self.load_clim(tag='prior', path_dict=self.configs['prior_path'],
                       anom_period=self.configs[f'prior_anom_period'],
                       rename_dict=prior_rename_dict, verbose=verbose)
        self.load_clim(tag='obs', path_dict=self.configs['obs_path'],
                       anom_period=self.configs[f'obs_anom_period'],
                       rename_dict=obs_rename_dict, verbose=verbose)
        self.calib_psms(ptype_psm_dict=self.configs['ptype_psm_dict'],
                        ptype_season_dict=self.configs['ptype_season_dict'], verbose=verbose)
        self.forward_psms(verbose=verbose)

        if 'prior_annualize_months' in self.configs:
            self.annualize_clim(tag='prior', months=self.configs['prior_annualize_months'], verbose=verbose)

        if 'prior_regrid_nlat' in self.configs:
            self.regrid_clim(tag='prior', nlat=self.configs['prior_regrid_nlat'], 
                             nlon=self.configs['prior_regrid_nlon'], verbose=verbose)

        self.save_cfg(save_dirpath=self.configs['save_dirpath'], verbose=verbose)
        self.save(save_dirpath=self.configs['save_dirpath'], verbose=verbose)

        t_e = time.time()
        t_used = t_e - t_s
        p_success(f'>>> DONE! Total time used: {t_used/60:.2f} mins.')


def run_cfg(cfg_path, seeds=None, run_mc=True, verbose=False):
    # get job_save_path
    job = ReconJob()
    with open(cfg_path, 'r') as f:
        job.configs = yaml.safe_load(f)

    job_save_path = os.path.join(job.configs['save_dirpath'], 'job.pkl')

    # load precalculated job.pkl if exists, or prepare the job
    if os.path.exists(job_save_path):
        job = pd.read_pickle(job_save_path)
        p_success(f'>>> {job_save_path} loaded')
    else:
        job.prepare_cfg(cfg_path, seeds=seeds, verbose=verbose)

    if seeds is not None and np.size(seeds) == 1:
        seeds = [seeds]
    
    # run the Monte-Carlo iterations
    if run_mc:
        job.run_mc(
            recon_period=job.configs['recon_period'],
            recon_loc_rad=job.configs['recon_loc_rad'],
            recon_timescale=job.configs['recon_timescale'],
            recon_seeds=seeds,
            assim_frac=job.configs['assim_frac'],
            compress_params=job.configs['compress_params'],
            output_full_ens=job.configs['output_full_ens'],
            verbose=verbose,
        )