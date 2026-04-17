import torch
import numpy as np
import xarray as xr
import colorama as ca
import collections.abc

def check_cuda():
    p_hint(f'CUDA availability: {torch.cuda.is_available()}')
    p_hint(f'Available GPUs: {torch.cuda.device_count()}')

def p_header(text):
    print(ca.Fore.CYAN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_hint(text):
    print(ca.Fore.LIGHTBLACK_EX + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_success(text):
    print(ca.Fore.GREEN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_fail(text):
    print(ca.Fore.RED + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_warning(text):
    print(ca.Fore.YELLOW + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def gcd(lat1, lon1, lat2, lon2, radius=6378.137):
    ''' 2D Great Circle Distance [km]

    Args:
        radius (float): Earth radius
    '''
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = radius * c
    return dist

def states2ds(states, ds):
    original_shapes = {vn: ds[vn].shape for vn in ds.data_vars}
    original_dims = {vn: ds[vn].dims for vn in ds.data_vars}
    original_coords = {vn: ds[vn].coords for vn in ds.data_vars}
    # print(original_shapes)
    # print(original_dims)
    # print(original_coords)
    ds_out = xr.Dataset()
    start_loc = 0
    for vn in ds.data_vars:
        if 'ens' in original_dims[vn]:
            end_loc = start_loc + int(np.prod(original_shapes[vn][:-1]))
            # p_hint(f'{np.prod(original_shapes[vn][:-1]) = }')
        else:
            end_loc = start_loc + int(np.prod(original_shapes[vn]))
            # p_hint(f'{np.prod(original_shapes[vn]) = }')
        # print(vn, start_loc, end_loc)

        # p_hint(f'{vn = }')
        # p_hint(f'{start_loc = }')
        # p_hint(f'{end_loc = }')
        # p_hint(f'{np.shape(states) = }')
        # p_hint(f'{np.shape(states[start_loc:end_loc]) = }')
        data = states[start_loc:end_loc].reshape(original_shapes[vn])
        nan_mask = np.isnan(ds[vn].values)
        data[nan_mask] = np.nan

        ds_out[vn] = xr.DataArray(
            data,
            dims=original_dims[vn],
            coords=original_coords[vn],
        )
        start_loc = end_loc
        ds_out[vn].attrs = ds[vn].attrs

    return ds_out
        

# def gcd_3d(loc1, loc2, radius=6371.0):
#     ''' 3D Great Circle Distance [km]

#     Args:
#         loc1 (tuple): lat1 [degree], lon1 [degree], depth1 [km]
#         loc2 (tuple): lat2 [degree], lon2 [degree], depth2 [km]
#         radius (float): Earth radius
#     '''
#     lat1, lon1, depth1 = loc1
#     lat2, lon2, depth2 = loc2

#     # Convert degrees to radians
#     lat1, lon1 = np.radians(lat1), np.radians(lon1)
#     lat2, lon2 = np.radians(lat2), np.radians(lon2)
    
#     # Calculate radial distances (Earth's radius minus depth)
#     r1 = radius - depth1
#     r2 = radius - depth2
    
#     # Compute central angle component
#     central_angle = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    
#     # Compute the 3D distance
#     distance_3d = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * central_angle)
    
#     return distance_3d

def str2list(s, sep=','):
    l = [int(ss.strip()) for ss in s.split(sep)]
    return l

def lon360(lon180):
    return np.mod(lon180, 360)

def lon180(lon360):
    return np.mod(lon360 + 180, 360) - 180

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(url: str, fname: str, chunk_size=1024, show_bar=True, verify=True):
    resp = requests.get(url, stream=True, verify=verify)
    total = int(resp.headers.get('content-length', 0))
    if show_bar:
        with open(fname, 'wb') as file, tqdm(
            desc='Fetching data',
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
    else:
        with open(fname, 'wb') as file:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)

proxydb_url_dict = {
    'PAGES2kv2': 'https://github.com/fzhu2e/cfr-data/raw/main/pages2kv2.json',
    'pseudoPAGES2k/ppwn_SNRinf_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNRinf_rta.nc',
    'pseudoPAGES2k/ppwn_SNR10_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR10_rta.nc',
    'pseudoPAGES2k/ppwn_SNR2_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR2_rta.nc',
    'pseudoPAGES2k/ppwn_SNR1_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR1_rta.nc',
    'pseudoPAGES2k/ppwn_SNR0.5_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR0.5_rta.nc',
    'pseudoPAGES2k/ppwn_SNR0.25_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR0.25_rta.nc',
    'pseudoPAGES2k/ppwn_SNRinf_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNRinf_fta.nc',
    'pseudoPAGES2k/ppwn_SNR10_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR10_fta.nc',
    'pseudoPAGES2k/ppwn_SNR2_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR2_fta.nc',
    'pseudoPAGES2k/ppwn_SNR1_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR1_fta.nc',
    'pseudoPAGES2k/ppwn_SNR0.5_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR0.5_fta.nc',
    'pseudoPAGES2k/ppwn_SNR0.25_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/ppwn_SNR0.25_fta.nc',
    'pseudoPAGES2k/tpwn_SNR10_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR10_rta.nc',
    'pseudoPAGES2k/tpwn_SNR2_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR2_rta.nc',
    'pseudoPAGES2k/tpwn_SNR1_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR1_rta.nc',
    'pseudoPAGES2k/tpwn_SNR0.5_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR0.5_rta.nc',
    'pseudoPAGES2k/tpwn_SNR0.25_rta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR0.25_rta.nc',
    'pseudoPAGES2k/tpwn_SNR10_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR10_fta.nc',
    'pseudoPAGES2k/tpwn_SNR2_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR2_fta.nc',
    'pseudoPAGES2k/tpwn_SNR1_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR1_fta.nc',
    'pseudoPAGES2k/tpwn_SNR0.5_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR0.5_fta.nc',
    'pseudoPAGES2k/tpwn_SNR0.25_fta': 'https://github.com/fzhu2e/paper-pseudoPAGES2k/raw/main/data/tpwn_SNR0.25_fta.nc',
}

climfd_url_dict = {
    'iCESM_past1000historical/tas': 'https://atmos.washington.edu/~rtardif/LMR/prior/tas_sfc_Amon_iCESM_past1000historical_085001-200512.nc',
    'iCESM_past1000historical/pr': 'https://atmos.washington.edu/~rtardif/LMR/prior/pr_sfc_Amon_iCESM_past1000historical_085001-200512.nc',
    'iCESM_past1000historical/d18O': 'https://atmos.washington.edu/~rtardif/LMR/prior/d18O_sfc_Amon_iCESM_past1000historical_085001-200512.nc',
    'iCESM_past1000historical/psl': 'https://atmos.washington.edu/~rtardif/LMR/prior/psl_sfc_Amon_iCESM_past1000historical_085001-200512.nc',
    'iCESM_past1000/tas': 'https://atmos.washington.edu/~rtardif/LMR/prior/tas_sfc_Amon_iCESM_past1000_085001-184912.nc',
    'iCESM_past1000/pr': 'https://atmos.washington.edu/~rtardif/LMR/prior/pr_sfc_Amon_iCESM_past1000_085001-184912.nc',
    'iCESM_past1000/d18O': 'https://atmos.washington.edu/~rtardif/LMR/prior/d18O_sfc_Amon_iCESM_past1000_085001-184912.nc',
    'iCESM_past1000/psl': 'https://atmos.washington.edu/~rtardif/LMR/prior/psl_sfc_Amon_iCESM_past1000_085001-184912.nc',
    'gistemp1200_ERSSTv4': 'https://github.com/fzhu2e/cfr-data/raw/main/gistemp1200_ERSSTv4.nc.gz',
    'gistemp1200_GHCNv4_ERSSTv5': 'https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz',
    'CRUTSv4.07/tas': 'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.07/cruts.2304141047.v4.07/tmp/cru_ts4.07.1901.2022.tmp.dat.nc.gz',
    'CRUTSv4.07/pr': 'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.07/cruts.2304141047.v4.07/pre/cru_ts4.07.1901.2022.pre.dat.nc.gz',
    '20CRv3/tas': 'https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/2mSI-MO/air.2m.mon.mean.nc',
    '20CRv3/pr': 'https://downloads.psl.noaa.gov/Datasets/20thC_ReanV3/Monthlies/sfcSI-MO/prate.mon.mean.nc',
    'HadCRUTv5': 'https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc',
    'HadCRUT4.6_GraphEM': 'https://github.com/fzhu2e/cfr-data/raw/main/HadCRUT4.6_GraphEM_median.nc',
    'GPCCv2020': 'https://downloads.psl.noaa.gov//Datasets/gpcc/monitor/precip.monitor.mon.total.1x1.v2020.nc',
}

ensts_url_dict = {
    'BC09_NINO34': 'https://github.com/fzhu2e/cfr-data/raw/main/BC09_NINO34.csv',
}
