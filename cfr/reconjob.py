import os
import copy
import time
from shutil import ReadError
import numpy as np
import yaml
from tqdm import tqdm
import pandas as pd
import random
import glob
from .climate import ClimateField
from .proxy import ProxyDatabase, ProxyRecord
try:
    from graphem import GraphEM, Graph
    from graphem.solver import verif_stats as graphem_verif_stats
    from graphem.solver import KCV
    from sklearn.model_selection import KFold
except:
    pass

try:
    from . import psm
except:
    pass

import xarray as xr
from . import utils
from . import da
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
    ''' The class for a reconstruction Job.

    Args:
        configs (dict, optional): a dictionary of configurations. Defaults to None.
        verbose (bool, optional): print verbose information. Defaults to False.
    '''
    def __init__(self, configs=None, verbose=False):
        ''' Initialize a reconstruction job.
        '''
        self.configs = {} if configs is None else configs
        if verbose:
            p_header(f'>>> job.configs:')
            pp.pprint(self.configs)

    def io_cfg(self, k, v, default=None, verbose=False):
        ''' Add-to or read-from configurations.

        Args:
            k (str): the name of a configuration item
            v (object): any value of the configuration item
            default (object): the default value of the configuration item
            verbose (bool, optional): print verbose information. Defaults to False.
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
        ''' Right a configurations item to `self.configs`.

        Args:
            k (str): the name of a configuration item
            v (object): any value of the configuration item
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        self.configs.update({k: v})
        if verbose: p_header(f'>>> job.configs["{k}"] = {v}')

    def mark_pids(self, verbose=False):
        ''' Mark proxy IDs to `self.configs`.

        Args:
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        self.write_cfg('pids', self.proxydb.pids, verbose=verbose)

    def erase_cfg(self, keys, verbose=False):
        ''' Erase configuration items from `self.configs`.

        Args:
            keys (list): a list of configuration item strings.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        for k in keys:
            self.configs.pop(k)
            if verbose: p_success(f'>>> job.configs["{k}"] dropped')

    def save_cfg(self, save_dirpath=None, verbose=False):
        ''' Save `self.configs` to a directory.

        Args:
            save_dirpath (str): the directory path for saving `self.configs`.
                The filename will be `configs.yml`.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = os.path.join(save_dirpath, 'configs.yml') 
        with open(save_path, 'w') as f:
            yaml.dump(self.configs, f)

        if verbose: p_success(f'>>> job.configs saved to: {save_path}')


    def copy(self):
        ''' Make a deep copy of the object itself.
        '''
        return copy.deepcopy(self)

    def load_proxydb(self, path=None, verbose=False, **kwargs):
        ''' Load the proxy database from a `pandas.DataFrame`.

        Args:
            path (str, optional): the path to the pickle file of the `pandas.DataFrame`. Defaults to None.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
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
        ''' Filter the proxy database.

        Args:
            inplace (bool): if True, the annualized proxy database will replace the current `self.proxydb`.
            verbose (bool, optional): print verbose information. Defaults to False.
        
        See :py:mod:`cfr.proxy.ProxyDatabase.filter()` for more information.
        '''
        if inplace:
            self.proxydb = self.proxydb.filter(*args, **kwargs)

            if verbose:
                p_success(f'>>> {self.proxydb.nrec} records remaining')
                p_success(f'>>> job.proxydb updated')

        else:
            pdb = self.proxydb.filter(*args, **kwargs)
            return pdb

    def annualize_proxydb(self, months=None, ptypes=None, inplace=True, verbose=False, **kwargs):
        ''' Annualize the proxy database.
        
        Args:
            months (list): the list of months for annualization.
            ptypes (list): the list of proxy types.
            inplace (bool): if True, the annualized proxy database will replace the current `self.proxydb`.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
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
        ''' Clear the tags for each proxy record in the proxy database.
        
        Args:
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        for pid, pobj in self.proxydb.records.items():
            pobj.tags = []

        if verbose:
            p_success(f'>>> job.proxydb updated with tags cleared')

    def split_proxydb(self, tag='calibrated', assim_frac=None, seed=0, verbose=False):
        ''' Split the proxy database.

        Args:
            tag (str, optional): the tag for filtering the proxy database. Defaults to 'calibrated'.
            assim_frac (float, optional): the fraction of proxies for assimilation. Defaults to None.
            seed (int, optional): random seed. Defaults to 0.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
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
        ''' Load grided climate data, either model simulations or instrumental observations.

        Args:
            tag (str): the tag to denote identity; either 'prior' or 'obs.
            path_dict (dict): the dictionary of paths of climate data files with keys to be the variable names,
                e.g., 'tas' and 'pr', etc.
            rename_dict (dict): the dictionary for renaming the variable names in the climate data files.
            anom_period (tuple or list): the time period for computing the anomaly.
            time_name (str): the name of the time dimension in the climate data files.
            lon_name (str): the name of the longitude dimension in the climate data files.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
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
                self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file, time_name=time_name).wrap_lon(lon_name=lon_name, time_name=time_name)
            else:
                if time_name == 'time':
                    self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file, time_name=time_name).get_anom(ref_period=anom_period).wrap_lon(lon_name=lon_name, time_name=time_name)
                elif time_name == 'year':
                    self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file, time_name=time_name).center(ref_period=anom_period, time_name=time_name).wrap_lon(lon_name=lon_name, time_name=time_name)

            self.__dict__[tag][vn].da.name = vn

        if verbose:
            p_success(f'>>> {tag} variables {list(self.__dict__[tag].keys())} loaded')
            p_success(f'>>> job.{tag} created')

    def annualize_clim(self, tag, verbose=False, months=None):
        ''' Annualize the grided climate data, either model simulations or instrumental observations.

        Args:
            tag (str): the tag to denote identity; either 'prior' or 'obs.
            months (list): the list of months for annualization.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        months = self.io_cfg('prior_annualize_months', months, default=list(range(1, 13)), verbose=verbose)

        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'>>> Processing {vn} ...')
            self.__dict__[tag][vn] = fd.annualize(months=months)

        if verbose:
            p_success(f'>>> job.{tag} updated')

    def regrid_clim(self, tag, verbose=False, lats=None, lons=None, nlat=None, nlon=None, periodic_lon=True):
        ''' Regrid the grided climate data, either model simulations or instrumental observations.

        Args:
            tag (str): the tag to denote identity; either 'prior' or 'obs.
            lats (list or numpy.array): the latitudes of the regridded grid.
            lons (list or numpy.array): the longitudes of the regridded grid.
            nlat (int): the number of latitudes of the regridded grid; effective when `lats = None`.
            nlon (int): the number of longitudes of the regridded grid; effective when `lons = None`..
            periodic_lon (bool): if True, then assume the original longitudes form a loop.
        '''
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
        ''' Crop the grided climate data, either model simulations or instrumental observations.

        Args:
            tag (str): the tag to denote identity; either 'prior' or 'obs.
            lat_min (float): the minimum latitude of the cropped grid.
            lat_max (float): the maximum latitude of the cropped grid.
            lon_min (float): the minimum longitude of the cropped grid.
            lon_max (float): the maximum longitude of the cropped grid.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        lat_min = self.io_cfg(f'prior_lat_min', lat_min, default=-90, verbose=verbose)
        lat_max = self.io_cfg(f'prior_lat_max', lat_max, default=90, verbose=verbose)
        lon_min = self.io_cfg(f'prior_lon_min', lon_min, default=0, verbose=verbose)
        lon_max = self.io_cfg(f'prior_lon_max', lon_max, default=360, verbose=verbose)

        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'>>> Processing {vn} ...')
            self.__dict__[tag][vn] = fd.crop(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        
        
    def calib_psms(self, ptype_psm_dict=None, ptype_season_dict=None, calib_period=None,
                   use_predefined_R=False, verbose=False, **kwargs):
        ''' Calibrate the PSMs.

        Args:
            ptype_psm_dict (dict): the dictionary to denote the PSM for each proxy type.
            ptype_season_dict (dict): the dictionary to denote the seasonality for each proxy type.
            calib_period (tuple or list): the time period for calibration.
            use_predefined_R (bool): use the predefined observation error covariance instead of by calibration.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
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

        for pid, pobj in tqdm(self.proxydb.records.items(), total=self.proxydb.nrec, desc='Calibrating the PSMs'):
            psm_name = ptype_psm_dict[pobj.ptype]

            if psm_name in ['TempPlusNoise']:
                for vn in psm.__dict__[psm_name]().climate_required:
                    if 'clim' not in pobj.__dict__ or f'model.{vn}' not in pobj.clim:
                        pobj.get_clim(self.prior[vn], tag='model')
            else:
                for vn in psm.__dict__[psm_name]().climate_required:
                    if 'clim' not in pobj.__dict__ or f'obs.{vn}' not in pobj.clim:
                        pobj.get_clim(self.obs[vn], tag='obs')


            pobj.psm = psm.__dict__[psm_name](pobj)
            if psm_name in ['TempPlusNoise']:
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
                if verbose: p_warning(f'>>> PSM for {pid} failed to be calibrated.')
            else:
                self.proxydb.records[pid].tags.add('calibrated')
                if not use_predefined_R:
                    self.proxydb.records[pid].R = pobj.psm.calib_details['PSMmse']  # assign obs err variance

        if verbose:
            p_success(f'>>> {self.proxydb.nrec_tags("calibrated")} records tagged "calibrated" with ProxyRecord.psm created')

    def forward_psms(self, verbose=False, **kwargs):
        ''' Forward the PSMs.

        Args:
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        pdb_calib = self.proxydb.filter(by='tag', keys={'calibrated'})
            
        for pid, pobj in tqdm(pdb_calib.records.items(), total=pdb_calib.nrec, desc='Forwarding the PSMs'):
            for vn in pobj.psm.climate_required:
                if 'clim' not in pobj.__dict__ or f'model.{vn}' not in pobj.clim:
                    pobj.get_clim(self.prior[vn], tag='model')

            pobj.pseudo = pobj.psm.forward(**kwargs)

        if verbose:
            p_success(f'>>> ProxyRecord.pseudo created for {pdb_calib.nrec} records')

    def run_da(self, recon_period=None, recon_loc_rad=None, recon_timescale=None,
               recon_sampling_mode=None, recon_sampling_dist=None, recon_vars=None,
               normal_sampling_sigma=None, normal_sampling_cutoff_factor=None,
               trim_prior=None, nens=None, seed=0, verbose=False, debug=False,
               allownan=None):
        ''' Run the data assimilation workflows.

        Args:
            recon_period (tuple or list): the time period for reconstruction.
            recon_loc_rad (float): the localization radius; unit: km.
            recon_timescale (int or float): the timescale for reconstruction.
            recon_sampling_mode (str): 'fixed' or 'rolling' window for prior sampling.
            recon_sampling_dist (str): 'normal' or 'uniform' distribution for prior sampling.
            recon_vars (list): the list of variables to reconstruct. Defaults to ['tas'].
            normal_sampling_sigma (str): the standard deviation of the normal distribution for prior sampling.
            normal_sampling_cutoff_factor (int): the cutoff factor for the window for prior sampling.
            allownan (bool): if True, NaNs in prior is allowed.
            nens (int): the ensemble size.
            seed (int): the random seed.
            verbose (bool, optional): print verbose information. Defaults to False.
            debug (bool): if True, the debug mode is turned on and more information will be printed out.
        '''
        recon_period = self.io_cfg('recon_period', recon_period, default=[0, 2000], verbose=verbose)
        recon_loc_rad = self.io_cfg('recon_loc_rad', recon_loc_rad, default=25000, verbose=verbose)  # unit: km
        recon_timescale = self.io_cfg('recon_timescale', recon_timescale, default=1, verbose=verbose)  # unit: yr
        recon_sampling_mode = self.io_cfg('recon_sampling_mode', recon_sampling_mode, default='fixed', verbose=verbose)
        recon_vars = self.io_cfg('recon_vars', recon_vars, default=['tas'], verbose=verbose)
        trim_prior = self.io_cfg('trim_prior', trim_prior, default=True, verbose=verbose)
        allownan = self.io_cfg('allownan', allownan, default=False, verbose=verbose)
        if recon_sampling_mode == 'rolling':
            recon_sampling_dist = self.io_cfg('recon_sampling_dist', recon_sampling_dist, default='normal', verbose=verbose)
            normal_sampling_sigma = self.io_cfg('normal_sampling_sigma', normal_sampling_sigma, verbose=verbose)
            normal_sampling_cutoff_factor = self.io_cfg('normal_sampling_cutoff_factor', normal_sampling_cutoff_factor, default=3, verbose=verbose)

        nens = self.io_cfg('nens', nens, default=100, verbose=verbose)

        recon_yrs = np.arange(recon_period[0], recon_period[-1]+1, recon_timescale)

        self.da_solver = da.EnKF(self.prior, self.proxydb, recon_vars=recon_vars, nens=nens, seed=seed)
        self.da_solver.run(
            recon_yrs=recon_yrs,
            recon_loc_rad=recon_loc_rad,
            recon_timescale=recon_timescale,
            recon_sampling_mode=recon_sampling_mode,
            recon_sampling_dist=recon_sampling_dist,
            trim_prior=trim_prior,
            normal_sampling_sigma=normal_sampling_sigma,
            normal_sampling_cutoff_factor=normal_sampling_cutoff_factor,
            verbose=verbose, debug=debug, allownan=allownan)

        self.recon_fields = self.da_solver.recon_fields
        if verbose: p_success(f'>>> job.recon_fields created')

    def run_da_mc(self, recon_period=None, recon_loc_rad=None, recon_timescale=None, nens=None,
               output_full_ens=None, recon_sampling_mode=None, recon_sampling_dist=None, recon_vars=None,
               normal_sampling_sigma=None, normal_sampling_cutoff_factor=None, trim_prior=None,
               recon_seeds=None, assim_frac=None, save_dirpath=None, compress_params=None,
               allownan=None, verbose=False):
        ''' Run the Monte-Carlo iterations of data assimilation workflows.

        Args:
            recon_period (tuple or list): the time period for reconstruction.
            recon_loc_rad (float): the localization radius; unit: km.
            recon_timescale (int or float): the timescale for reconstruction.
            recon_sampling_mode (str): 'fixed' or 'rolling' window for prior sampling.
            recon_sampling_dist (str): 'normal' or 'uniform' distribution for prior sampling.
            recon_vars (list): the list of variables to reconstruct. Defaults to ['tas'].
            normal_sampling_sigma (str): the standard deviation of the normal distribution for prior sampling.
            normal_sampling_cutoff_factor (int): the cutoff factor for the window for prior sampling.
            output_full_ens (bool): if True, the full ensemble fields will be stored to netCDF files.
            nens (int): the ensemble size.
            recon_seed (int): the random seeds.
            allownan (bool): if True, NaNs in prior is allowed.
            assim_frac (float, optional): the fraction of proxies for assimilation. Defaults to None.
            verbose (bool, optional): print verbose information. Defaults to False.
            save_dirpath (str): the directory path for saving the reconstruction results.
            compress_params (dict): the paramters for compression when storing the reconstruction results to netCDF files.
        '''

        t_s = time.time()
        recon_period = self.io_cfg('recon_period', recon_period, default=[0, 2000], verbose=verbose)
        recon_loc_rad = self.io_cfg('recon_loc_rad', recon_loc_rad, default=25000, verbose=verbose)  # unit: km
        recon_timescale = self.io_cfg('recon_timescale', recon_timescale, default=1, verbose=verbose)  # unit: yr
        recon_vars = self.io_cfg('recon_vars', recon_vars, default=['tas'], verbose=verbose)
        nens = self.io_cfg('nens', nens, default=100, verbose=verbose)
        recon_seeds = self.io_cfg('recon_seeds', recon_seeds, default=np.arange(0, 20), verbose=verbose)
        assim_frac = self.io_cfg('assim_frac', assim_frac, default=0.75, verbose=verbose)
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
        os.makedirs(save_dirpath, exist_ok=True)
        compress_params = self.io_cfg('compress_params', compress_params, default={'zlib': True, 'least_significant_digit': 2}, verbose=verbose)
        output_full_ens = self.io_cfg('output_full_ens', output_full_ens, default=False, verbose=verbose)
        recon_sampling_mode = self.io_cfg('recon_sampling_mode', recon_sampling_mode, default='fixed', verbose=verbose)
        trim_prior = self.io_cfg('trim_prior', trim_prior, default=True, verbose=verbose)
        allownan = self.io_cfg('allownan', allownan, default=False, verbose=verbose)
        if recon_sampling_mode == 'rolling':
            normal_sampling_sigma = self.io_cfg('normal_sampling_sigma', normal_sampling_sigma, verbose=verbose)
            normal_sampling_cutoff_factor = self.io_cfg('normal_sampling_cutoff_factor', normal_sampling_cutoff_factor, default=3, verbose=verbose)
            recon_sampling_dist = self.io_cfg('recon_sampling_dist', recon_sampling_dist, default='normal', verbose=verbose)

        for seed in recon_seeds:
            if verbose: p_header(f'>>> seed: {seed} | max: {recon_seeds[-1]}')

            self.split_proxydb(seed=seed, assim_frac=assim_frac, verbose=False)
            self.run_da(recon_period=recon_period, recon_loc_rad=recon_loc_rad, nens=nens,
                        trim_prior=trim_prior,
                        recon_sampling_mode=recon_sampling_mode,
                        recon_sampling_dist=recon_sampling_dist,
                        normal_sampling_sigma=normal_sampling_sigma,
                        normal_sampling_cutoff_factor=normal_sampling_cutoff_factor,
                        recon_timescale=recon_timescale, seed=seed,
                        allownan=allownan, verbose=False)

            recon_savepath = os.path.join(save_dirpath, f'job_r{seed:02d}_recon.nc')
            self.save_recon(recon_savepath, compress_params=compress_params, mark_assim_pids=True,
                            verbose=verbose, output_full_ens=output_full_ens, grid='prior')

        t_e = time.time()
        t_used = t_e - t_s
        p_success(f'>>> DONE! Total time used: {t_used/60:.2f} mins.')


    def save(self, save_dirpath=None, filename='job.pkl', verbose=False):
        ''' Save the ReconJob object to a pickle file.

        Args:
            save_dirpath (str): the directory path for saving the :py:mod:`cfr.ReconJob` object.
            filename (str): the filename of the to-be-saved :py:mod:`cfr.ReconJob` object.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
        os.makedirs(save_dirpath, exist_ok=True)
        new = self.copy()
        for tag in ['prior', 'obs']:
            if hasattr(self, tag):
                for k, v in self.__dict__[tag].items():
                    savepath = os.path.join(save_dirpath, f'{tag}_{k}.nc')
                    v.da.to_netcdf(savepath)
                    if verbose: p_success(f'>>> {tag}_{k} saved to: {savepath}')
                    del(new.__dict__[tag][k].da)

        for pid, pobj in self.proxydb.records.items():
            if hasattr(pobj, 'clim'):
                del(pobj.clim)

        savepath = os.path.join(save_dirpath, filename)
        pd.to_pickle(new, savepath)
        if verbose: p_success(f'>>> job saved to: {savepath}')

    def load(self, save_dirpath=None, filename='job.pkl', verbose=False):
        ''' Load a ReconJob object from a pickle file.

        Args:
            save_dirpath (str): the directory path for saving the :py:mod:`cfr.ReconJob` object.
            filename (str): the filename of the to-be-saved :py:mod:`cfr.ReconJob` object.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        job = pd.read_pickle(os.path.join(save_dirpath, filename))
        if verbose: p_success(f'>>> job is loaded')

        for tag in ['prior', 'obs']:
            paths = sorted(glob.glob(os.path.join(save_dirpath, f'{tag}_*.nc')))
            for p in paths:
                if os.path.exists(p):
                    vn = os.path.basename(p).split(f'{tag}_')[-1].split('.nc')[0]
                    job.__dict__[tag][vn].da = xr.load_dataarray(p)
                    if verbose: p_success(f'>>> job.{tag}["{vn}"].da is loaded')

        for k, v in job.__dict__.items():
            self.__dict__[k] = v
    
    def save_recon(self, save_path, compress_params=None, verbose=False, output_full_ens=False,
                   mark_assim_pids=False, output_indices=None, grid='prior'):
        ''' Save the reconstruction results.

        Args:
            tag (str): 'da' or 'graphem'
            save_path (str): the path for saving the reconstruciton results.
            verbose (bool, optional): print verbose information. Defaults to False.
            output_full_ens (bool): if True, the full ensemble fields will be stored to netCDF files.
            output_indices (list): the list of indices to output; supported indices:

                * 'nino3.4'
                * 'nino1+2'
                * 'nino3'
                * 'nino4'
                * 'tpi'
                * 'wp'
            compress_params (dict): the paramters for compression when storing the reconstruction results to netCDF files.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''

        compress_params = self.io_cfg(
            'compress_params', compress_params,
            default={'zlib': True, 'least_significant_digit': 2},
            verbose=False)

        output_indices = self.io_cfg(
            'output_indices', output_indices,
            default=['gm', 'nhm', 'shm', 'nino3.4'],
            verbose=False)

        ds = xr.Dataset()
        if 'recon_time' in self.configs:
            year = self.configs['recon_time']
        elif 'recon_period' in self.configs:
            year = np.arange(self.configs['recon_period'][0], self.configs['recon_period'][1]+1, self.configs['recon_timescale'])
        else:
            raise ValueError('Unknown reconstruction years.')

        for vn, fd in self.recon_fields.items():
            if len(np.shape(fd)) == 3:
                fd = fd[:, np.newaxis, :, :]  # add the axis for ens

            nyr, nens, nlat, nlon = np.shape(fd)

            da = xr.DataArray(fd,
                dims=['year', 'ens', 'lat', 'lon'],
                coords={
                    'year': year,
                    'ens': np.arange(nens),
                    'lat': self.prior[vn].lat if grid == 'prior' else self.obs[vn].lat,
                    'lon': self.prior[vn].lon if grid == 'prior' else self.obs[vn].lon,
                })

            # output indices
            if 'gm' in output_indices: ds[f'{vn}_gm'] = utils.geo_mean(da)
            if 'nhm' in output_indices: ds[f'{vn}_nhm'] = utils.geo_mean(da, lat_min=0)
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
        if mark_assim_pids:
            pdb_assim = self.proxydb.filter(by='tag', keys=['assim'])
            pdb_eval = self.proxydb.filter(by='tag', keys=['eval'])
            ds.attrs = {
                'pids_assim': pdb_assim.pids,
                'pids_eval': pdb_eval.pids,
            }

        # save to netCDF files
        ds.to_netcdf(save_path, encoding=encoding_dict)

        if verbose: p_success(f'>>> Reconstructed fields saved to: {save_path}')


    def prep_da_cfg(self, cfg_path, seeds=None, verbose=False):
        ''' Prepare the configuration items.

        Args:
            cfg_path (str): the path of the configuration YAML file.
            seeds (list, optional): the list of random seeds.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
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

    def prep_graphem(self, recon_time=None, calib_time=None,  recon_period=None, recon_timescale=None, calib_period=None, verbose=False):
        ''' A shortcut of the steps for GraphEM data preparation

        Args:
            recon_time (array list, optional): the time points to reconstruct
            calib_time (array list, optional): the time points for calibration
            recon_period (tuple, optional): the reconstruction timespan.
                Effective when `recon_time` or `calib_time` is None. Defaults to None.
            recon_timescale (float, optional): the reconstruction timescale. Defaults to None.
                Effective when `recon_time` or `calib_time` is None. Defaults to None.
            calib_period (tuple, optional): the calibration timespan. Defaults to None.
            verbose (bool, optional): print verbose information. Defaults to False.
        '''
        if recon_time is None or calib_time is None:
            recon_period = self.io_cfg('recon_period', recon_period, default=(1001, 2000), verbose=verbose)
            recon_timescale = self.io_cfg('recon_timescale', recon_timescale, default=1, verbose=verbose)  # unit: yr
            calib_period = self.io_cfg('calib_period', calib_period, default=(1850, 2000), verbose=verbose)

            recon_time = np.arange(recon_period[0], recon_period[1]+1, recon_timescale)
            calib_time = np.arange(calib_period[0], calib_period[1]+1, recon_timescale)
        else:
            recon_time = self.io_cfg('recon_time', recon_time, verbose=verbose)
            calib_time = self.io_cfg('calib_time', calib_time, verbose=verbose)

        self.graphem_params = {}
        self.graphem_params['recon_time'] = recon_time
        self.graphem_params['calib_time'] = calib_time
        if verbose: p_success(f'>>> job.graphem_params["recon_time"] created')
        if verbose: p_success(f'>>> job.graphem_params["calib_time"] created')

        vn = list(self.obs.keys())[0]
        obs = self.obs[vn]  
        obs_nt = obs.da.shape[0]
        obs_2d = obs.da.values.reshape(obs_nt, -1)
        obs_npos = np.shape(obs_2d)[-1]

        recon_idx = [list(obs.time).index(t) for t in recon_time]
        self.graphem_params['obs_2d'] = obs_2d[recon_idx]
        if verbose: p_success(f'>>> job.graphem_params["field_obs"] created')

        nt = np.size(recon_time)
        field = np.ndarray((nt, obs_npos)) 
        field[:] = np.nan

        field_calib_idx = [list(recon_time).index(t) for t in calib_time]  
        self.graphem_params['calib_idx'] = field_calib_idx
        if verbose: p_success(f'>>> job.graphem_params["calib_idx"] created')

        obs_calib_idx = [list(obs.time).index(t) for t in calib_time]
        field[field_calib_idx] = obs_2d[obs_calib_idx] #align matrices
        self.graphem_params['field'] = field  
        if verbose: p_success(f'>>> job.graphem_params["field"] created')

        lonlat = np.ndarray((obs_npos+self.proxydb.nrec, 2))

        k = 0
        for i in range(np.size(obs.da.lon)):
            for j in range(np.size(obs.da.lat)):
                lonlat[k] = [np.mod(obs.lon[i], 360), obs.lat[j]]
                k += 1

        df_proxy = pd.DataFrame(index=recon_time)
        for pid, pobj in self.proxydb.records.items():
            series = pd.Series(index=pobj.time, data=pobj.value, name=pid)
            df_proxy = pd.concat([df_proxy, series], axis=1)
            lonlat[k] = [np.mod(pobj.lon, 360), pobj.lat]
            k += 1

        mask = [True if i in recon_time else False for i in df_proxy.index.values]
        df_proxy = df_proxy[mask]
        
        self.graphem_params['df_proxy'] = df_proxy 
        self.graphem_params['proxy'] = df_proxy.values
        
        if verbose: p_success(f'>>> job.graphem_params["df_proxy"] created')
        if verbose: p_success(f'>>> job.graphem_params["proxy"] created')

        self.graphem_params['lonlat'] = lonlat
        if verbose: p_success(f'>>> job.graphem_params["lonlat"] created')

    def graphem_kcv(self, cv_time, ctrl_params, graph_type='neighborhood', stat='MSE', n_splits=5):
        ''' k-fold cross-validation
        
        Arguments
        ---------
        
        cv_time : array-like, 1d
            explain how it differs from recon_time or calib_time
            
        ctrl_params : array-like, 1d
            array of control parameters to try
            
        graph_type : str
            type of graph. Either "neighborhood" or "glasso"
            
        stat: str
            name of objective function. Choices are "MSE", "RE", "CE" or "R2".
            
        n_splits: int
            number of splits (default = 5)
        
        '''
        kf = KFold(n_splits=n_splits)
        cv_stats = np.empty((kf.n_splits, len(ctrl_params))) # stats for a scalar: TODO: generalize to the grid of job.graphem_params['field']
        adjs = {}
        i = 0
        for train_idx, test_idx in kf.split(cv_time):
            p_header(f'>>> Processing fold {i+1}:')
            train = cv_time[train_idx]
            for j, param in enumerate(ctrl_params):
                p_header(f'>>> parameter = {param}')
                j_cv = self.copy()

                # specify calibration period
                j_cv.prep_graphem(
                    recon_time = cv_time,
                    calib_time = train,  
                    verbose=False)

                # declare graph object
                g_cv = Graph(
                    j_cv.graphem_params['lonlat'],
                    j_cv.graphem_params['field'],
                    j_cv.graphem_params['proxy'])
                
                # estimate graph
                if graph_type == "neighborhood":
                    g_cv.neigh_adj(cutoff_radius=param)
                elif graph_type == "glasso":
                    g_cv.glasso_adj(target_FF=param, target_FP=param)
                    
                adjs[(i+1, param)] = g_cv
                
                # run graphem with this graph
                j_cv.run_graphem(
                    save_recon=False,
                    verbose=False,
                    estimate_graph=False,
                    graph=g_cv.adj)

                # compute verification statistics
                field_r = j_cv.graphem_solver.field_r
                target = j_cv.graphem_params['obs_2d']
                cv_stats[i, j] = graphem_verif_stats(
                    field_r, target,
                    j_cv.graphem_params['calib_idx']).__dict__[stat].mean()
            i += 1
            
        kcv_res = KCV(cv_stats, cutoff_radii=ctrl_params, adjs=adjs)

        return kcv_res


    def run_graphem(self, save_recon=True, save_dirpath=None, save_filename=None,
                    load_precalc_solver=False, solver_save_path=None,
                    compress_params=None, verbose=False,
                    **fit_kws):
        ''' Run the GraphEM solver, essentially the :py:meth: `GraphEM.solver.GraphEM.fit` method

        Note that the arguments for :py:meth: `GraphEM.solver.GraphEM.fit` can be appended in the
        argument list of this function directly. For instance, to pass a pre-calculated graph, use
        `estimate_graph=False` and `graph=g.adj`, where `g` is the :py:`Graph` object.

        Args:
            save_dirpath (str): the path to save the related results
            load_precalculated (bool, optional): load the precalculated `Graph` object. Defaults to False.
            verbose (bool, optional): print verbose information. Defaults to False.
            fit_kws (dict): the arguments for :py:meth: `GraphEM.solver.GraphEM.fit`

        See also:
            cfr.graphem.solver.GraphEM.fit : fitting the GraphEM method

        '''
        compress_params = self.io_cfg('compress_params', compress_params, default={'zlib': True, 'least_significant_digit': 1}, verbose=verbose)

        if save_recon:
            save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
            save_filename = self.io_cfg('save_filename', save_filename, default='job_r01_recon.nc', verbose=verbose)
            os.makedirs(save_dirpath, exist_ok=True)

        if load_precalc_solver:
            self.graphem_solver = pd.read_pickle(solver_save_path)
            if verbose: p_success(f'job.graphem_solver created with the existing result at: {solver_save_path}')
        else:
            self.graphem_solver = GraphEM()
            fit_kwargs = {
                'lonlat': self.graphem_params['lonlat'],
                'graph_method': 'neighborhood',
            }
            fit_kwargs.update(fit_kws)
            self.graphem_solver.fit(
                self.graphem_params['field'],
                self.graphem_params['proxy'],
                self.graphem_params['calib_idx'],
                verbose=verbose,
                **fit_kwargs)

            if verbose: p_success(f'job.graphem_solver created and saved to: {solver_save_path}')

            if solver_save_path is not None:
                pd.to_pickle(self.graphem_solver, solver_save_path)
                if verbose: p_success(f'job.graphem_solver saved to: {solver_save_path}')

        nt = np.shape(self.graphem_params['field'])[0]
        vn = list(self.obs.keys())[0]
        _, nlat, nlon = np.shape(self.obs[vn].da.values)
        self.recon_fields = {}  # to make it multivariable reconstruction ready
        self.recon_fields[vn] = self.graphem_solver.field_r.reshape((nt, nlat, nlon))
        if verbose: p_success(f'>>> job.recon_fields created')

        if save_recon:
            recon_savepath = os.path.join(save_dirpath, save_filename)
            self.save_recon(recon_savepath, compress_params=compress_params, verbose=verbose, grid='obs')


def run_da_cfg(cfg_path, seeds=None, run_mc=True, verbose=False):
    ''' Run the reconstruction job according to a configuration YAML file.

    Args:
        cfg_path (str): the path of the configuration YAML file.
        run_mc (bool): if False, the reconstruction part will not executed for the convenience of checking the preparation part.
        seeds (list, optional): the list of random seeds.
        verbose (bool, optional): print verbose information. Defaults to False.
    '''
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
        job.prep_da_cfg(cfg_path, seeds=seeds, verbose=verbose)

    if seeds is not None and np.size(seeds) == 1:
        seeds = [seeds]
    
    # run the Monte-Carlo iterations
    if run_mc:
        job.run_da_mc(
            recon_period=job.configs['recon_period'],
            recon_loc_rad=job.configs['recon_loc_rad'],
            recon_timescale=job.configs['recon_timescale'],
            recon_seeds=seeds,
            assim_frac=job.configs['assim_frac'],
            compress_params=job.configs['compress_params'],
            output_full_ens=job.configs['output_full_ens'],
            verbose=verbose,
        )