import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cftime
from termcolor import cprint

def p_header(text):
    return cprint(text, 'cyan', attrs=['bold'])

def p_hint(text):
    return cprint(text, 'grey', attrs=['bold'])

def p_success(text):
    return cprint(text, 'green', attrs=['bold'])

def p_fail(text):
    return cprint(text, 'red', attrs=['bold'])

def p_warning(text):
    return cprint(text, 'yellow', attrs=['bold'])


def clean_ts(ts, ys):
    ''' Delete the NaNs in the time series and sort it with time axis ascending

    Parameters
    ----------
    ts : array
        The time axis of the time series, NaNs allowed
    ys : array
        A time series, NaNs allowed

    Returns
    -------
    ts : array
        The time axis of the time series without NaNs
    ys : array
        The time series without NaNs
    '''
    ys = np.array(ys, dtype=np.float)
    ts = np.array(ts, dtype=np.float)
    assert ys.size == ts.size, 'The size of time axis and data value should be equal!'

    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ts = ts[~np.isnan(ys_tmp)]
    ts_tmp = np.copy(ts)
    ys = ys[~np.isnan(ts_tmp)]
    ts = ts[~np.isnan(ts_tmp)]

    # sort the time series so that the time axis will be ascending
    sort_ind = np.argsort(ts)
    ys = ys[sort_ind]
    ts = ts[sort_ind]

    # handle duplicated time points
    t_count = {}
    value_at_t = {}
    for i, t in enumerate(ts):
        if t not in t_count:
            t_count[t] = 1
            value_at_t[t] = ys[i]
        else:
            t_count[t] += 1
            value_at_t[t] += ys[i]

    ys = []
    ts = []
    for t, v in value_at_t.items():
        ys.append(v / t_count[t])
        ts.append(t)

    ts = np.array(ts)
    ys = np.array(ys)

    return ts, ys

def gcd(lat1, lon1, lat2, lon2):
	'''
	Calculate the great circle distance between two points
	on the earth (specified in decimal degrees)

	Parameters:
	-----------
	lat1: float
		Latitude of first location (degrees)
	lon1: float
		Longitude of first location (degrees)
	lat2: float
		Latitude of second location (degrees)
	lon2: float
		Longitude of second location (degrees)
		
	Returns:
	--------
	km: float
		Great circle distance between the two locations (in kilometers)
	'''
	# convert decimal degrees to radians
	lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

	# haversine formula
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
	c = 2 * np.arcsin(np.sqrt(a))

	# 6367 km is the radius of the Earth
	km = 6367 * c
	return km

def annualize(time=None, value=None, da=None, months=list(range(1, 13))):
    # convert negative months to support expressions like [-12, 1, 2]
    months = np.abs(months)

    if da is None:
        dates = year_float2datetime(time)
        da_wk = xr.DataArray(
            value, dims=['time'], coords={'time': dates}
        )
    else:
        da_wk = da

    old_yrs = np.array(list(set(np.floor(datetime2year_float(da_wk.time.values)))))

    sda = da_wk.sel(time=da_wk.time.dt.month.isin(months))
    anchor = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    idx = months[-1]-1
    sda_ann = sda.resample(time=f'A-{anchor[idx]}').mean()
    time_ann = np.floor(datetime2year_float(sda_ann.time.values))
    value_ann = sda_ann.values

    # drop the last one since its caused by months cross two years, e.g., [12, 1, 2]
    if len(old_yrs) < len(time_ann):
        time_ann = time_ann[:-1]
        value_ann = value_ann[:-1, ...]
        sda_ann = sda_ann.drop_isel({'time': -1})

    if da is None:
        return time_ann, value_ann
    else:
        return sda_ann

def annualize_var(year_float, var, resolution='month', weights=None):
    ''' Annualize a variable array
    Args:
        var (ndarray): the target variable array with 1st dim to be year
        year_float (1-D array): the time axis of the variable array
        weights (ndarray): the weights that shares the same shape of the target variable array
    Returns:
        var_ann (ndarray): the annualized variable array
        year_ann (1-D array): the time axis of the annualized variable array
    '''
    var = np.array(var)
    year_float = np.array(year_float)

    ndims = len(np.shape(var))
    dims = ['time']
    for i in range(ndims-1):
        dims.append(f'dim{i+1}')

    time = year_float2datetime(year_float, resolution=resolution)

    if weights is not None:
        weights_da = xr.DataArray(weights, dims=dims, coords={'time': time})

        coeff = np.ndarray(np.shape(weights))
        for i, gp in enumerate(list(weights_da.groupby('time.year'))):
            year, value = gp
            k = np.shape(value)[0]
            coeff[k*i:k*(i+1)] = value / np.sum(value, axis=0)

        del weights, weights_da  # save the memory

        var = np.multiply(coeff, var)
        var_da = xr.DataArray(var, dims=dims, coords={'time': time})
        var_ann = var_da.groupby('time.year').sum('time')

    else:
        var_da = xr.DataArray(var, dims=dims, coords={'time': time})
        var_ann = var_da.groupby('time.year').mean('time')

    var_ann = var_ann.values

    year_ann = np.sort(list(set([t.year for t in time])))
    return year_ann, var_ann

def ymd2year_float(year, month, day):
    ''' Convert a set of (year, month, day) to an array of floats in unit of year
    '''
    year_float = []
    for y, m, d in zip(year, month, day):
        date = datetime(year=y, month=m, day=d)
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_part = date - fst_day
        year_length = lst_day - fst_day
        year_float.append(y + year_part/year_length)

    year_float = np.array(year_float)
    return year_float

def datetime2year_float(date):
    ''' Convert a list of dates to floats in year
    '''
    if isinstance(date[0], np.datetime64):
        date = pd.to_datetime(date)

    year = [d.year for d in date]
    month = [d.month for d in date]
    day = [d.day for d in date]

    year_float = ymd2year_float(year, month, day)

    return year_float

def year_float2datetime(year_float, resolution='day'):
    ''' Convert an array of floats in unit of year to a datetime time; accuracy: one day
    '''
    # if np.min(year_float) < 0:
    #     raise ValueError('Cannot handel negative years. Please truncate first.')

    year = np.array([int(y) for y in year_float], dtype=int)
    month = np.zeros(np.size(year), dtype=int)
    day = np.zeros(np.size(year), dtype=int)

    for i, y in enumerate(year):
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_length = lst_day - fst_day

        year_part = (year_float[i] - y)*year_length + timedelta(minutes=1)  # to fix the numerical error
        date = year_part + fst_day
        month[i] = date.month
        day[i] = date.day

    if resolution == 'day':
        time = [cftime.datetime(y, m, d, 0, 0, 0, 0, 0, 0, has_year_zero=True) for y, m, d in zip(year, month, day)]
    elif resolution == 'month':
        time = [cftime.datetime(y, m, 1, 0, 0, 0, 0, 0, 0, has_year_zero=True) for y, m in zip(year, month)]

    return time

def year_float2dates(year_float):
    ''' Convert an array of floats in unit of year to a datetime time; accuracy: one day
    '''
    # if np.min(year_float) < 0:
    #     raise ValueError('Cannot handel negative years. Please truncate first.')

    year = np.array([int(y) for y in year_float], dtype=int)
    dates = []

    for i, y in enumerate(year):
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_length = lst_day - fst_day

        year_part = (year_float[i] - y)*year_length + timedelta(minutes=1)  # to fix the numerical error
        date = year_part + fst_day
        dates.append(date)


    return dates

def regrid_field(field, lat, lon, lat_new, lon_new):
    import spharm
    nlat_old, nlon_old = np.size(lat), np.size(lon)
    nlat_new, nlon_new = np.size(lat_new), np.size(lon_new)
    spec_old = spharm.Spharmt(nlon_old, nlat_old, gridtype='regular', legfunc='computed')
    spec_new = spharm.Spharmt(nlon_new, nlat_new, gridtype='regular', legfunc='computed')

    field_new = []
    for field_old in field:
        regridded_field =  spharm.regrid(spec_old, spec_new, field_old, ntrunc=None, smooth=None)
        field_new.append(regridded_field)

    field_new = np.array(field_new)
    return field_new

def regrid_field_curv_rect(field, lat_curv, lon_curv, dlat=1, lon_range=None, lat_range=None, dlon=1, roi=None, roi_factor=1e6, fill_value=np.nan):
    import pyresample
    ''' Regrid a curvilinear grid to a linear rectilinear grid

    Note that the range of lon_curve should be (-180, 180)
    '''
    ndim_curv = np.size(np.shape(lat_curv))
    if ndim_curv == 1:
        lon_curv, lat_curv = np.meshgrid(lon_curv, lat_curv)

    if lon_range is not None:
        lons = np.arange(lon_range[0], lon_range[-1]+dlon, dlon)
    else:
        lons = np.arange(np.min(lon_curv), np.max(lon_curv)+dlon, dlon)

    if lat_range is not None:
        lats = np.arange(lat_range[0], lat_range[-1]+dlat, dlat)
    else:
        lats = np.arange(np.min(lat_curv), np.max(lat_curv)+dlat, dlat)

    lon_rect, lat_rect = np.meshgrid(lons, lats)
    lon_rect = pyresample.utils.wrap_longitudes(lon_rect)

    if roi is None:
        roi = roi_factor * np.abs(dlon)

    old_grid = pyresample.geometry.SwathDefinition(lons=lon_curv, lats=lat_curv)
    new_grid = pyresample.geometry.SwathDefinition(lons=lon_rect, lats=lat_rect)

    nt, nlat, nlon = np.shape(field)
    rgd_data = np.ndarray(shape=(nt, np.size(lats), np.size(lons)))
    for i, d in enumerate(field):
        rgd_data[i] = pyresample.kd_tree.resample_nearest(
            old_grid, d, new_grid,
            radius_of_influence=roi,
            fill_value=np.nan
        )

    return rgd_data, lats, lons

def coefficient_efficiency(ref, test, valid=None):
    """ Compute the coefficient of efficiency for a test time series, with respect to a reference time series.

    Inputs:
    test:  test array
    ref:   reference array, of same size as test
    valid: fraction of valid data required to calculate the statistic

    Note: Assumes that the first dimension in test and ref arrays is time!!!

    Outputs:
    CE: CE statistic calculated following Nash & Sutcliffe (1970)
    """

    # check array dimensions
    dims_test = test.shape
    dims_ref  = ref.shape
    # print('dims_test: ', dims_test, ' dims_ref: ', dims_ref)

    if len(dims_ref) == 3:   # 3D: time + 2D spatial
        dims = dims_ref[1:3]
    elif len(dims_ref) == 2: # 2D: time + 1D spatial
        dims = dims_ref[1:2]
    elif len(dims_ref) == 1: # 0D: time series
        dims = 1
    else:
        print('In coefficient_efficiency(): Problem with input array dimension! Exiting...')
        SystemExit(1)

    CE = np.zeros(dims)

    # error
    error = test - ref

    # CE
    numer = np.nansum(np.power(error,2),axis=0)
    denom = np.nansum(np.power(ref-np.nanmean(ref,axis=0),2),axis=0)
    CE    = 1. - np.divide(numer,denom)

    if valid:
        nbok  = np.sum(np.isfinite(ref),axis=0)
        nball = float(dims_ref[0])
        ratio = np.divide(nbok,nball)
        indok  = np.where(ratio >= valid)
        indbad = np.where(ratio < valid)
        dim_indbad = len(indbad)
        testlist = [indbad[k].size for k in range(dim_indbad)]
        if not all(v == 0 for v in testlist):
            if isinstance(dims,(tuple,list)):
                CE[indbad] = np.nan
            else:
                CE = np.nan

    return CE