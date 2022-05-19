import os
import copy
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

    def erase_cfg(self, keys, verbose=False):
        for k in keys:
            self.configs.pop(k)
            if verbose: p_success(f'>>> job.configs["{k}"] dropped')


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

    def annualize_proxydb(self, ptypes=None, inplace=True, verbose=False, **kwargs):
        if ptypes is None:
            if inplace:
                self.proxydb = self.proxydb.annualize(**kwargs)
            else:
                pdb = self.proxydb.annualize(**kwargs)
                return pdb
        else:
            pdb_filtered = self.proxydb.filter(by='ptype', keys=ptypes)
            pdb_ann = pdb_filtered.annualize(**kwargs)
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

    def load_clim(self, tag, path_dict=None, rename_dict=None, center_period=None, lon_name='lon', verbose=False):
        path_dict = self.io_cfg(f'{tag}_path', path_dict, verbose=verbose)

        self.__dict__[tag] = {}
        for vn, path in path_dict.items():
            if rename_dict is None:
                vn_in_file = vn
            else:
                vn_in_file = rename_dict[vn]

            self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file).center(ref_period=center_period).wrap_lon(lon_name=lon_name)
            self.__dict__[tag][vn].da.name = vn

        if verbose:
            p_success(f'>>> instrumental observation variables {list(self.__dict__[tag].keys())} loaded')
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
        
        
    def calib_psms(self, ptype_psm_dict=None, ptype_season_dict=None, calib_period=None, verbose=False):
        ptype_psm_dict = self.io_cfg(
            'ptype_psm_dict', ptype_psm_dict,
            default={ptype: 'Linear' for ptype in set(self.proxydb.type_list)},
            verbose=verbose)

        ptype_season_dict = self.io_cfg(
            'ptype_season_dict', ptype_season_dict,
            default={ptype: 'Linear' for ptype in set(self.proxydb.type_list)},
            verbose=verbose)

        calib_period = self.io_cfg(
            'psm_calib_period', calib_period,
            default=(1850, 2015),
            verbose=verbose)

        for pid, pobj in tqdm(self.proxydb.records.items(), total=self.proxydb.nrec, desc='Calibrating the PSMs:'):
            psm_name = ptype_psm_dict[pobj.ptype]

            for vn in psm.__dict__[psm_name]().climate_required:
                pobj.get_clim(self.obs[vn], tag='obs')

            pobj.psm = psm.__dict__[psm_name](pobj)
            if psm_name == 'Bilinear':
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

    def run_da(self, recon_period=None, recon_loc_rad=None, recon_timescale=None, verbose=False, debug=False):
        recon_period = self.io_cfg('recon_period', recon_period, default=[0, 2000], verbose=verbose)
        recon_loc_rad = self.io_cfg('recon_loc_rad', recon_loc_rad, default=25000, verbose=verbose)  # unit: km
        recon_timescale = self.io_cfg('recon_timescale', recon_timescale, default=1, verbose=verbose)  # unit: yr

        recon_yrs = np.arange(recon_period[0], recon_period[-1]+1)

        solver = da.EnKF(self.prior, self.proxydb)
        solver.run(
            recon_yrs=recon_yrs,
            recon_loc_rad=recon_loc_rad,
            recon_timescale=recon_timescale,
            verbose=verbose, debug=debug)

        self.recon_fields = solver.recon_fields
        if verbose: p_success(f'>>> job.recon_fields created')

    def run_mc(self, recon_period=None, recon_loc_rad=None, recon_timescale=None,
               recon_seeds=None, assim_frac=None, save_dirpath=None, verbose=False):
        recon_period = self.io_cfg('recon_period', recon_period, default=[0, 2000], verbose=verbose)
        recon_loc_rad = self.io_cfg('recon_loc_rad', recon_loc_rad, default=25000, verbose=verbose)  # unit: km
        recon_timescale = self.io_cfg('recon_timescale', recon_timescale, default=1, verbose=verbose)  # unit: yr
        recon_seeds = self.io_cfg('recon_seeds', recon_seeds, default=np.arange(0, 20), verbose=verbose)
        assim_frac = self.io_cfg('assim_frac', assim_frac, default=0.75, verbose=verbose)
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)

        for seed in recon_seeds:
            if verbose: p_header(f'>>> seed: {seed} | max: {recon_seeds[-1]}')
            self.split_proxydb(seed=seed, assim_frac=assim_frac, verbose=verbose)
            self.run_da(recon_period=recon_period, recon_loc_rad=recon_loc_rad, recon_timescale=recon_timescale)
            # save_recon()

        p_success('>>> DONE!')
        

    def save(self, save_dirpath=None, filename='job.pkl', verbose=False):
        save_dirpath = self.io_cfg('save_dirpath', save_dirpath, verbose=verbose)
        os.makedirs(save_dirpath, exist_ok=True)
        pd.to_pickle(self, os.path.join(save_dirpath, filename))
        if verbose: p_success(f'>>> job saved to: {save_dirpath}')
    
    def save_recon(self, save_path, compress_dict={'zlib': True, 'least_significant_digit': 1}, verbose=False,
               output_geo_mean=False, target_lats=[], target_lons=[], output_full_ens=False, dtype=np.float32):

        output_dict = {}
        for vn, fd in self.recon_fields.items():
            nyr, nens, nlat, nlon = np.shape(fd)
            if output_full_ens:
                output_var = np.array(fd, dtype=dtype)
                output_dict[vn] = (('year', 'ens', 'lat', 'lon'), output_var)
            else:
                output_var = np.array(fd.mean(axis=1), dtype=dtype)
                output_dict[vn] = (('year', 'lat', 'lon'), output_var)

            lats, lons = self.prior.fields[vn].lat, self.prior.fields[vn].lon
            try:
                gm_ens = np.ndarray((nyr, nens), dtype=dtype)
                nhm_ens = np.ndarray((nyr, nens), dtype=dtype)
                shm_ens = np.ndarray((nyr, nens), dtype=dtype)
                for k in range(nens):
                    gm_ens[:,k], nhm_ens[:,k], shm_ens[:,k] = utils.global_hemispheric_means(fd[:,k,:,:], lats)

                output_dict[f'{vn}_gm_ens'] = (('year', 'ens'), gm_ens)
                output_dict[f'{vn}_nhm_ens'] = (('year', 'ens'), nhm_ens)
                output_dict[f'{vn}_shm_ens'] = (('year', 'ens'), shm_ens)
            except:
                if verbose: p_warning(f'LMRt: job.save_recon() >>> Global hemispheric means cannot be calculated')

            if vn == 'tas':
                try:
                    nino_ind = utils.nino_indices(fd, lats, lons)
                    nino12 = nino_ind['nino1+2']
                    nino3 = nino_ind['nino3']
                    nino34 = nino_ind['nino3.4']
                    nino4 = nino_ind['nino4']
                    wpi = nino_ind['wpi']

                    nino12 = np.array(nino12, dtype=dtype)
                    nino3 = np.array(nino3, dtype=dtype)
                    nino34 = np.array(nino34, dtype=dtype)
                    nino4 = np.array(nino4, dtype=dtype)

                    output_dict['nino1+2'] = (('year', 'ens'), nino12)
                    output_dict['nino3'] = (('year', 'ens'), nino3)
                    output_dict['nino3.4'] = (('year', 'ens'), nino34)
                    output_dict['nino4'] = (('year', 'ens'), nino4)
                    output_dict['wpi'] = (('year', 'ens'), wpi)
                except:
                    if verbose: p_warning(f'LMRt: job.save_recon() >>> NINO or West Pacific Indices cannot be calculated')

                # calculate tripole index (TPI)
                try:
                    tpi = calc_tpi(fd, lats, lons)
                    tpi = np.array(tpi, dtype=dtype)
                    output_dict['tpi'] = (('year', 'ens'), tpi)
                except:
                    if verbose: p_warning(f'LMRt: job.save_recon() >>> Tripole Index (TPI) cannot be calculated')

            if output_geo_mean:
                geo_mean_ts = geo_mean(fd, lats, lons, target_lats, target_lons)
                output_dict['geo_mean'] = (('year', 'ens'), geo_mean_ts)

        ds = xr.Dataset(
            data_vars=output_dict,
            coords={
                'year': np.arange(self.configs['recon_period'][0], self.configs['recon_period'][1]+1),
                'ens': np.arange(nens),
                'lat': lats,
                'lon': lons,
            })

        if compress_dict is not None:
            encoding_dict = {}
            for k in output_dict.keys():
                encoding_dict[k] = compress_dict

            ds.to_netcdf(save_path, encoding=encoding_dict)
        else:
            ds.to_netcdf(save_path)

        if verbose: p_header(f'LMRt: job.save_recon() >>> Reconstructed fields saved to: {save_path}')