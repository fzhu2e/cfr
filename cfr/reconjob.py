import os
import copy
from shutil import ReadError
import numpy as np
import yaml
from tqdm import tqdm
import pandas as pd
from .climate import ClimateField
from .proxy import ProxyDatabase, ProxyRecord
from . import psm

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

    def atrf_cfg(self, k, v, default=None, verbose=False):
        ''' Add-to or read-from configurations
        '''
        if v is not None:
            self.configs.update({k: v})
            if verbose: p_header(f'>>> job.configs["{k}"] = {v}')
        elif k in self.configs:
            v = self.configs[k]
        elif default is not None:
            v = default
        else:
            raise ValueError(f'{k} not properly set.')

        return v

    def copy(self):
        return copy.deepcopy(self)

    def load_proxydb(self, path=None, verbose=False, **kwargs):
        path = self.atrf_cfg('proxydb_path', path, verbose=verbose)

        _, ext =  os.path.splitext(path)
        if ext.lower() == '.pkl':
            df = pd.read_pickle(path)
        else:
            raise ReadError(f'The extention of the file [{ext}] is not supported. Support list: [.pkl, ] .')

        self.proxydb = ProxyDatabase().from_df(df, **kwargs)
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records loaded')
            p_success(f'>>> job.proxydb created')

    def filter_proxydb(self, *args, verbose=False, **kwargs):
        self.proxydb = self.proxydb.filter(*args, **kwargs)
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records remaining')
            p_success(f'>>> job.proxydb updated')

    def annualize_proxydb(self, verbose=False, **kwargs):
        self.proxydb = self.proxydb.annualize(**kwargs)
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records remaining')
            p_success(f'>>> job.proxydb updated')

    def split_proxydb(self, proxy_assim_frac=0.75, seed=0, verbose=False):
        if proxy_assim_frac is not None:
            self.configs.update({'proxy_assim_frac': proxy_assim_frac})
            if verbose: p_header(f'>>> job.configs["proxy_assim_frac"] = {proxy_assim_frac}')
        else:
            proxy_assim_frac = self.configs['proxy_assim_frac']
        

    def load_gridded(self, tag, path_dict=None, rename_dict=None, center_period=None, lon_name='lon', verbose=False):
        path_dict = self.atrf_cfg(f'{tag}_path', path_dict, verbose=verbose)

        self.__dict__[tag] = {}
        for vn, path in path_dict.items():
            if rename_dict is None:
                vn_in_file = vn
            else:
                vn_in_file = rename_dict[vn]

            self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file).center(center_period).wrap_lon(lon_name=lon_name)
            self.__dict__[tag][vn].da.name = vn

        if verbose:
            p_success(f'>>> instrumental observation variables {list(self.__dict__[tag].keys())} loaded')
            p_success(f'>>> job.{tag} created')

    def annualize_ds(self, tag, verbose=False, **kwargs):
        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'>>> Processing {vn} ...')
            self.__dict__[tag][vn] = fd.annualize(**kwargs)

        if verbose:
            p_success(f'>>> job.{tag} updated')

    def regrid_ds(self, tag, verbose=False, lats=None, lons=None, nlat=None, nlon=None, periodic_lon=True):
        if lats is not None: 
            lats_new = lats
        elif nlat is not None:
            lats_new = np.linspace(-90, 90, nlat)
        else:
            raise ValueError('lats or nlat should be set')

        if lons is not None:
            lons_new = lons
        elif nlon is not None:
            lons_new = np.linspace(0, 360, nlon)
        else:
            raise ValueError('lons or nlon should be set')

        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'>>> Processing {vn} ...')
            self.__dict__[tag][vn] = fd.regrid(lats=lats_new, lons=lons_new, periodic_lon=periodic_lon)
        
    def calib_psms(self, proxydb=None, ptype_psm_dict=None, ptype_season_dict=None, calib_period=None, verbose=False):
        ptype_psm_dict = self.atrf_cfg(
            'ptype_psm_dict', ptype_psm_dict,
            default={ptype: 'Linear' for ptype in set(self.proxydb.type_list)},
            verbose=verbose)

        ptype_season_dict = self.atrf_cfg(
            'ptype_season_dict', ptype_season_dict,
            default={ptype: 'Linear' for ptype in set(self.proxydb.type_list)},
            verbose=verbose)

        calib_period = self.atrf_cfg(
            'psm_calib_period', calib_period,
            default=(1850, 2015),
            verbose=verbose)

        self.psms = {}
        for pid, pobj in tqdm(self.proxydb.records.items(), total=self.proxydb.nrec, desc='Calibrating the PSMs:'):
            psm_name = ptype_psm_dict[pobj.ptype]

            for vn in psm.__dict__[psm_name]().climate_required:
                pobj.get_clim(self.obs[vn], tag='obs')

            self.psms[pid] = psm.__dict__[psm_name](pobj)
            self.psms[pid].calibrate(season_list=ptype_season_dict[pobj.ptype], calib_period=calib_period)

        
        # drop out the proxy record whose PSM is failed to be calibrated
        pids_to_remove = []
        for pid, mdl in self.psms.items():
            if mdl.calib_details is None:
                if verbose: p_warning(f'>>> The PSM for {pid} failed to calibrate.')
                pids_to_remove.append(pid)

        for pid in pids_to_remove:
            if pid in self.proxydb.records: self.proxydb.records.pop(pid)
            if pid in self.psms: self.psms.pop(pid)

        self.proxydb.refresh()

        if verbose:
            p_success(f'>>> job.psm created for {self.proxydb.nrec} records')
            p_success(f'>>> job.proxydb_bak created')
            p_success(f'>>> job.proxydb updated')

    def forward_psms(self, verbose=False, **kwargs):
        self.ppdb = ProxyDatabase()
        for pid, mdl in tqdm(self.psms.items(), total=self.proxydb.nrec, desc='Forwarding the PSMs:'):
            for vn in mdl.climate_required:
                self.proxydb.records[pid].get_clim(self.prior[vn], tag='model')

            self.ppdb += mdl.forward(**kwargs)

        if verbose:
            p_success(f'>>> job.ppdb created for {self.proxydb.nrec} records')

    def run_da(self, recon_period=None, recon_loc_rad=None, recon_timescale=None, verbose=False, debug=False):
        if recon_period is not None:
            self.configs['recon_period'] = recon_period
            if verbose: p_header(f'>>> job.configs["recon_period"] = {recon_period}')
        else:
            recon_period = self.configs['recon_period']

        if recon_timescale is not None:
            self.configs['recon_timescale'] = recon_timescale
            if verbose: p_header(f'>>> job.configs["recon_timescale"] = {recon_timescale}')
        else:
            recon_timescale = self.configs['recon_timescale']

        if recon_loc_rad is not None:
            self.configs['recon_loc_rad'] = recon_loc_rad
            if verbose: p_header(f'>>> job.configs["recon_loc_rad"] = {recon_loc_rad}')
        else:
            recon_loc_rad = self.configs['recon_loc_rad']

        recon_yrs = np.arange(recon_period[0], recon_period[-1]+1)
        Xb_aug = np.append(self.Xb, self.Ye_assim, axis=0)
        Xb_aug = np.append(Xb_aug, self.Ye_eval, axis=0)
        Xb_aug_coords = np.append(self.Xb_coords, self.Ye_assim_coords, axis=0)
        Xb_aug_coords = np.append(Xb_aug_coords, self.Ye_eval_coords, axis=0)

        nt = np.size(recon_yrs)
        nrow, nens = np.shape(Xb_aug)

        Xa = np.ndarray((nt, nrow, nens))
        for yr_idx, target_yr in enumerate(tqdm(recon_yrs, desc='KF updating')):
            Xa[yr_idx] = self.update_yr(target_yr, Xb_aug, Xb_aug_coords, recon_loc_rad, recon_timescale, verbose=verbose, debug=debug)

        recon_fields = {}
        for vn, irow in self.Xb_var_irow.items():
            _, nlat, nlon = np.shape(self.prior.fields[vn].value)
            recon_fields[vn] = Xa[:, irow[0]:irow[-1]+1, :].reshape((nt, nlat, nlon, nens))
            recon_fields[vn] = np.moveaxis(recon_fields[vn], -1, 1)

        self.recon_fields = recon_fields
        if verbose: p_success(f'>>> job.recon_fields created')

    def run(self, recon_seeds=None, verbose=False):
        if recon_seeds is not None:
            self.configs.update({'recon_seeds': recon_seeds})
            if verbose: p_header(f'>>> job.configs["recon_seeds"] = {recon_seeds}')
        else:
            recon_seeds = self.configs['recon_seeds']

        for seed in recon_seeds:
            if verbose: p_header(f'>>> seed: {seed} | max: {recon_seeds[-1]}')

        

    def save(self, job_dirpath=None, verbose=False):
        if job_dirpath is not None:
            self.configs.update({'job_dirpath': job_dirpath})
            if verbose: p_header(f'>>> job.configs["job_dirpath"] = {job_dirpath}')
        else:
            job_dirpath = self.configs['job_dirpath']

        os.makedirs(job_dirpath, exist_ok=True)
        pd.to_pickle(self, os.path.join(job_dirpath, 'job.pkl'))
        if verbose: p_success(f'>>> job saved to: {job_dirpath}')