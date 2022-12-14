from turtle import color
import numpy as np
import pandas as pd
from scipy import integrate, signal, stats
from tqdm import tqdm
from multiprocessing import cpu_count
try:
    import statsmodels.formula.api as smf
    from pathos.multiprocessing import ProcessingPool as Pool
    import fbm
    import PyVSL
except:
    pass

from . import utils
from .proxy import ProxyRecord

def clean_df(df, mask=None):
    pd.options.mode.chained_assignment = None
    if mask is not None:
        df_cleaned = df.loc[mask]
    else:
        df_cleaned = df

    for col in df.columns:
        df_cleaned.dropna(subset=[col], inplace=True)

    return df_cleaned

class TempPlusNoise:
    ''' A PSM that adds noise to the input climate signal.

    Args:
        pobj (cfr.proxy.ProxyRecord): the proxy record object
        climate_required (cfr.climate.ClimateField): the required climate field object for running this PSM
    '''
    def __init__(self, pobj=None, climate_required=['tas']):
        self.pobj = pobj
        self.climate_required = climate_required

    def calibrate(self, SNR=10, seasonality=list(range(1, 13)), vn='model.tas',
                  seed=None, noise='white', colored_noise_kws=None):
        ''' Calibrate the PSM.'''
        sigma = np.nanstd(self.pobj.clim[vn].da.values) / SNR
        if noise == 'white':
            rng = np.random.default_rng(seed)
            self.noise = rng.normal(0, sigma, np.size(self.pobj.clim[vn].da.values))
        elif noise == 'colored':
            colored_noise_kws = {} if colored_noise_kws is None else colored_noise_kws
            _colored_noise_kws = {'seed': seed, 'alpha': 1, 't': self.pobj.clim[vn].time}
            _colored_noise_kws.update(colored_noise_kws)
            self.noise = utils.colored_noise(**_colored_noise_kws)

        calib_details = {
            'PSMmse': np.mean(self.noise**2),
            'SNR': SNR,
            'seasonality': seasonality,
        }
        self.calib_details = calib_details

    def forward(self, vn='model.tas', no_noise=False):
        ''' Forward the PSM.'''
        if no_noise:
            value = self.pobj.clim[vn].da.values 
        else:
            value=self.pobj.clim[vn].da.values + self.noise

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=self.pobj.clim[vn].time,
            value=value,
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
            seasonality=self.calib_details['seasonality'],
        )

        return pp

class Linear:
    ''' A PSM that is based on univariate linear regression.

    Args:
        pobj (cfr.proxy.ProxyRecord): the proxy record object
        climate_required (cfr.climate.ClimateField): the required climate field object for running this PSM
    '''
    def __init__(self, pobj=None, climate_required=['tas']):
        self.pobj = pobj
        self.climate_required = climate_required

    def calibrate(self, calib_period=None, nobs_lb=25, metric='fitR2adj',
        season_list=[list(range(1, 13))], annualize_exog=True, exog_name='obs.tas', **fit_args):
        exog = self.pobj.clim[exog_name]

        if type(season_list[0]) is not list:
            season_list = [season_list]

        score_list = []
        mdl_list = []
        df_list = []
        sn_list = []
        exog_colname = exog_name.split('.')[-1]
        if not annualize_exog:
            season_list = ['monthly']

        for sn in season_list:
            exog_ann = exog.annualize(months=sn) if annualize_exog else exog
            df_exog = pd.DataFrame({'time': exog_ann.time, exog_colname: exog_ann.da.values})
            df_proxy = pd.DataFrame({'time': self.pobj.time, 'proxy': self.pobj.value})
            df = df_proxy.dropna().merge(df_exog.dropna(), how='inner', on='time')
            df.set_index('time', drop=True, inplace=True)
            df.sort_index(inplace=True)
            df.astype(np.float)
            if calib_period is not None:
                mask = (df.index>=calib_period[0]) & (df.index<=calib_period[1])
                df = clean_df(df, mask=mask)

            formula_spell = f'proxy ~ {exog_colname}'
            nobs = len(df)
            if nobs < nobs_lb:
                print(f'The number of overlapped data points is {nobs} < {nobs_lb}. Skipping ...')
            else:
                mdl = smf.ols(formula=formula_spell, data=df).fit(**fit_args)
                fitR2adj =  mdl.rsquared_adj,
                mse = np.mean(mdl.resid**2),
                score = {
                    'fitR2adj': fitR2adj,
                    'mse': mse,
                }
                score_list.append(score[metric])
                mdl_list.append(mdl)
                df_list.append(df)
                sn_list.append(sn)

        if len(score_list) > 0:
            opt_idx_dict = {
                'fitR2adj': np.argmax(score_list),
                'mse': np.argmin(score_list),
            }

            opt_idx = opt_idx_dict[metric]
            opt_mdl = mdl_list[opt_idx]
            opt_sn = sn_list[opt_idx]

            calib_details = {
                'df': df_list[opt_idx],
                'nobs': opt_mdl.nobs,
                'fitR2adj': opt_mdl.rsquared_adj,
                'PSMresid': opt_mdl.resid,
                'PSMmse': np.mean(opt_mdl.resid**2),
                'SNR': np.std(opt_mdl.predict()) / np.std(opt_mdl.resid),
                'seasonality': opt_sn,
            }

            self.calib_details = calib_details
            self.model = opt_mdl
        else:
            self.calib_details = None
            self.model = None

    def forward(self, exog_name='model.tas'):
        sn = self.calib_details['seasonality']
        exog = self.pobj.clim[exog_name]
        if sn != 'monthly':
            exog = exog.annualize(months=sn)
        exog_colname = exog_name.split('.')[-1]
        exog_dict = {
            exog_colname: exog.da.values,
        }

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=exog.time,
            value=np.array(self.model.predict(exog=exog_dict).values),
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
            seasonality=self.calib_details['seasonality'],
        )

        return pp


class Bilinear:
    ''' A PSM that is based on bivariate linear regression.

    Args:
        pobj (cfr.proxy.ProxyRecord): the proxy record object
        climate_required (cfr.climate.ClimateField): the required climate field object for running this PSM
    '''
    def __init__(self, pobj=None, climate_required = ['tas', 'pr']):
        self.pobj = pobj
        self.climate_required = climate_required

    def calibrate(self, calib_period=None, nobs_lb=25, metric='fitR2adj',
        season_list1=[list(range(1, 13))], season_list2=[list(range(1, 13))],
        annualize_exog=True, exog1_name='obs.tas', exog2_name='obs.pr', **fit_args):
        exog1 = self.pobj.clim[exog1_name]
        exog2 = self.pobj.clim[exog2_name]

        score_list = []
        mdl_list = []
        df_list = []
        sn_list = []
        exog1_colname = exog1_name.split('.')[-1]
        exog2_colname = exog2_name.split('.')[-1]
        if not annualize_exog:
            season_list1 = ['monthly']
            season_list2 = ['monthly']

        for sn1 in season_list1:
            exog1_ann = exog1.annualize(months=sn1) if annualize_exog else exog1
            df_exog1 = pd.DataFrame({'time': exog1_ann.time, exog1_colname: exog1_ann.da.values})
            df_proxy = pd.DataFrame({'time': self.pobj.time, 'proxy': self.pobj.value})
            df = df_proxy.dropna().merge(df_exog1.dropna(), how='inner', on='time')
            df_copy = df.copy()
            for sn2 in season_list2:
                exog2_ann = exog2.annualize(months=sn2) if annualize_exog else exog2
                df_exog2 = pd.DataFrame({'time': exog2_ann.time, exog2_colname: exog2_ann.da.values})
                df = df.merge(df_exog2.dropna(), how='inner', on='time')
                df.set_index('time', drop=True, inplace=True)
                df.sort_index(inplace=True)
                df.astype(np.float)
                if calib_period is not None:
                    mask = (df.index>=calib_period[0]) & (df.index<=calib_period[1])
                    df = clean_df(df, mask=mask)

                formula_spell = f'proxy ~ {exog1_colname} + {exog2_colname}'
                nobs = len(df)
                if nobs < nobs_lb:
                    print(f'The number of overlapped data points is {nobs} < {nobs_lb}. Skipping ...')
                else:
                    mdl = smf.ols(formula=formula_spell, data=df).fit(**fit_args)
                    fitR2adj =  mdl.rsquared_adj,
                    mse = np.mean(mdl.resid**2),
                    score = {
                        'fitR2adj': fitR2adj,
                        'mse': mse,
                    }
                    score_list.append(score[metric])
                    mdl_list.append(mdl)
                    df_list.append(df)
                    sn_list.append((sn1, sn2))

                df = df_copy.copy()

        if len(score_list) > 0:
            opt_idx_dict = {
                'fitR2adj': np.argmax(score_list),
                'mse': np.argmin(score_list),
            }

            opt_idx = opt_idx_dict[metric]
            opt_mdl = mdl_list[opt_idx]
            opt_sn = sn_list[opt_idx]

            calib_details = {
                'df': df_list[opt_idx],
                'nobs': opt_mdl.nobs,
                'fitR2adj': opt_mdl.rsquared_adj,
                'PSMresid': opt_mdl.resid,
                'PSMmse': np.mean(opt_mdl.resid**2),
                'SNR': np.std(opt_mdl.predict()) / np.std(opt_mdl.resid),
                'seasonality': opt_sn,
                # 'score_list': score_list,  # for debug purpose
            }

            self.calib_details = calib_details
            self.model = opt_mdl
        else:
            self.calib_details = None
            self.model = None

    def forward(self, exog1_name='model.tas', exog2_name='model.pr'):
        exog1 = self.pobj.clim[exog1_name]
        exog2 = self.pobj.clim[exog2_name]
        sn1, sn2 = self.calib_details['seasonality']
        if sn1 != 'monthly':
            exog1 = exog1.annualize(months=sn1)
        if sn2 != 'monthly':
            exog2 = exog2.annualize(months=sn2)

        common_yrs = np.intersect1d(exog1.da.year, exog2.da.year)
        exog1.da = exog1.da.sel(year=common_yrs)
        exog2.da = exog2.da.sel(year=common_yrs)

        exog1_colname = exog1_name.split('.')[-1]
        exog2_colname = exog2_name.split('.')[-1]
        exog_dict = {
            exog1_colname: exog1.da.values,
            exog2_colname: exog2.da.values,
        }

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=exog1.da.year,
            value=np.array(self.model.predict(exog=exog_dict).values),
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
            seasonality=self.calib_details['seasonality'],
        )

        return pp


class Ice_d18O():
    ''' The ice core d18O model adopted from PRYSM (https://github.com/sylvia-dee/PRYSM)

    Args:
        pobj (cfr.proxy.ProxyRecord): the proxy record object
        climate_required (cfr.climate.ClimateField): the required climate field object for running this PSM
    '''
    def __init__(self, pobj=None, tas_name='model.tas', pr_name='model.pr', psl_name='model.psl', d18O_name='model.d18O',
                 climate_required=['tas', 'pr', 'psl', 'd18O']):
        self.pobj = pobj
        self.tas_name = tas_name
        self.pr_name = pr_name
        self.psl_name = psl_name
        self.d18O_name = d18O_name
        self.climate_required = climate_required

    def forward(self, nproc=None):
        ''' The ice d18O model

        It takes model simulated montly tas, pr, psl, d18O as input.
        '''
        def ice_sensor(year, d18Op, pr, alt_diff=0.):
            ''' Icecore sensor model

            The ice core sensor model calculates precipitation-weighted del18OP (i.e. isotope ratio is weighted by
            the amount of precipitation that accumulates) and corrects for temperature and altitude bias between model
            and site ([Yurtsever, 1975], 0.3/100m [Vogel et al., 1975]).

            Args:
                year (1d array: time): time axis [year in float]
                d18Op (2d array: location, time): d18O of precipitation [permil]
                pr (2d array: location, time): precipitation rate [kg m-2 s-1]
                alt_diff 12d array: location): actual Altitude-Model Altitude [meters]

            Returns:
                d18Oice (2d array: location, year in int): annualizd d18O of ice [permil]

            References:
                Yurtsever, Y., Worldwide survey of stable isotopes in precipitation., Rep. Isotope Hydrology Section, IAEA, 1975.

            '''
            # Altitude Effect: cooling and precipitation of heavy isotopes.
            # O18 ~0.15 to 0.30 permil per 100m.

            alt_eff = -0.25
            alt_corr = (alt_diff/100.)*alt_eff

            year_ann, d18Op_weighted = utils.annualize_var(year, d18Op, weights=pr)
            d18O_ice = d18Op_weighted + alt_corr

            return d18O_ice


        def diffusivity(rho, T=250, P=0.9, rho_d=822, b=1.3):
            '''
            DOCSTRING: Function 'diffusivity'
            Description: Calculates diffusivity (in m^2/s) as a function of density.

            Inputs:
            P: Ambient Pressure in Atm
            T: Temperature in K
            rho: density profile (kg/m^3)
            rho_d: 822 kg/m^2 [default], density at which ice becomes impermeable to diffusion

            Defaults are available for all but rho, so only one argument need be entered.

            Note values for diffusivity in air:

            D16 = 2.1e-5*(T/273.15)^1.94*1/P
            D18 = D16/1.0285
            D2 = D16/1.0251
            D17 = D16/((D16/D18)^0.518)

            Reference: Johnsen et al. (2000): Diffusion of Stable isotopes in polar firn and ice:
            the isotope effect in firn diffusion

            '''

            # Set Constants

            R = 8.314478                                                # Gas constant
            m = 18.02e-3                                                # molar weight of water (in kg)
            alpha18 = np.exp(11.839/T-28.224e-3)                    # ice-vapor fractionation for oxygen 18
            p = np.exp(9.5504+3.53*np.log(T)-5723.265/T-0.0073*T)     # saturation vapor pressure
            Po = 1.                                                 # reference pressure, atmospheres
            rho_i = 920.  # kg/m^3, density of solid ice

            # Set diffusivity in air (units of m^2/s)

            Da = 2.1e-5*np.power((T/273.15), 1.94)*(Po/P)
            Dai = Da/1.0285

            # Calculate Tortuosity

            invtau = np.zeros(len(rho))
            #  for i in range(len(rho)):
            #      if rho[i] <= rho_i/np.sqrt(b):
            #          # invtau[i]=1.-1.3*np.power((rho[i]/rho_d),2)
            #          invtau[i] = 1.-1.3*np.power((rho[i]/rho_i), 2)
            #      else:
            #          invtau[i] = 0.

            selector =rho <= rho_i/np.sqrt(b)
            invtau[selector] = 1.-1.3*(rho[selector]/rho_i)**2

            D = m*p*invtau*Dai*(1/rho-1/rho_d)/(R*T*alpha18)

            return D


        def densification(Tavg, bdot, rhos, z):  # ,model='hljohnsen'):
            ''' Calculates steady state snow/firn depth density profiles using Herron-Langway type models.



            Args:
                Tavg: 10m temperature in celcius ## CELCIUS!  # fzhu: should be in K now
                bdot: accumulation rate in mwe/yr or (kg/m2/yr)
                rhos: surface density in kg/m3
                z: depth in true_metres

                model can be: {'HLJohnsen' 'HerronLangway' 'LiZwally' 'Helsen' 'NabarroHerring'}
                default is herronlangway. (The other models are tuned for non-stationary modelling (Read Arthern et al.2010 before applying in steady state).

            Returns:
                rho: density (kg/m3) for all z-values.
                zieq: ice equivalent depth for all z-values.
                t: age for all z-values (only taking densification into account.)

                Example usage:
                z=0:300
                [rho,zieq,t]=densitymodel(-31.5,177,340,z,'HerronLangway')
                plot(z,rho)

            References:
                Herron-Langway type models. (Arthern et al. 2010 formulation).
                Aslak Grinsted, University of Copenhagen 2010
                Adapted by Sylvia Dee, Brown University, 2017
                Optimized by Feng Zhu, University of Southern California, 2017

            '''
            rhoi = 920.
            rhoc = 550.
            rhow = 1000.
            rhos = 340.
            R = 8.314

            # Tavg=248.
            # bdot=0.1
            # Herron-Langway with Johnsen et al 2000 corrections.
            # Small corrections to HL model which are not in Arthern et al. 2010

            c0 = 0.85*11*(bdot/rhow)*np.exp(-10160./(R*Tavg))
            c1 = 1.15*575*np.sqrt(bdot/rhow)*np.exp(-21400./(R*Tavg))

            k0 = c0/bdot  # ~g4
            k1 = c1/bdot

            # critical depth at which rho=rhoc
            zc = (np.log(rhoc/(rhoi-rhoc))-np.log(rhos/(rhoi-rhos)))/(k0*rhoi)  # g6

            ix = z <= zc  # find the z's above and below zc
            upix = np.where(ix)  # indices above zc
            dnix = np.where(~ix)  # indices below zc

            q = np.zeros((z.shape))  # pre-allocate some space for q, rho
            rho = np.zeros((z.shape))

            # test to ensure that this will not blow up numerically if you have a very very long core.
            # manually set all super deep layers to solid ice (rhoi=920)
            NUM = k1*rhoi*(z-zc)+np.log(rhoc/(rhoi-rhoc))

            numerical = np.where(NUM <= 100.0)
            blowup = np.where(NUM > 100.0)

            q[dnix] = np.exp(k1*rhoi*(z[dnix]-zc)+np.log(rhoc/(rhoi-rhoc)))  # g7
            q[upix] = np.exp(k0*rhoi*z[upix]+np.log(rhos/(rhoi-rhos)))  # g7

            rho[numerical] = q[numerical]*rhoi/(1+q[numerical])  # [g8] modified by fzhu to fix inconsistency of array size
            rho[blowup] = rhoi

            # only calculate this if you want zieq
            tc = (np.log(rhoi-rhos)-np.log(rhoi-rhoc))/c0  # age at rho=rhoc [g17]
            t = np.zeros((z.shape))  # pre allocate a vector for age as a function of z
            t[upix] = (np.log(rhoi-rhos)-np.log(rhoi-rho[upix]))/c0  # [g16] above zc
            t[dnix] = (np.log(rhoi-rhoc)-np.log(rhoi+0.0001-rho[dnix]))/c1 + tc  # [g16] below zc
            tdiff = np.diff(t)

            # make sure time keeps increasing even after we reach the critical depth.
            if np.any(tdiff == 0.00):
                inflection = np.where(tdiff == 0.0)
                lineardepth_change = t[inflection][0]

                for i in range(len(t)):
                    if t[i] > lineardepth_change:
                        t[i] = t[i-1] + 1e-5

            zieq = t*bdot/rhoi  # [g15]

            return rho, zieq, t


        def ice_archive(d18Oice, pr_ann, tas_ann, psl_ann, nproc=8):
            ''' Accounts for diffusion and compaction in the firn.

            Args:
                d18Oice (1d array: year in int): annualizd d18O of ice [permil]
                pr_ann (1d array: year in int): precipitation rate [kg m-2 s-1]
                tas_ann (1d array: year in int): annualizd atomspheric temerature [K]
                psl_ann (1d array: year in int): annualizd sea level pressure [Pa]
                nproc (int): the number of processes for multiprocessing

            Returns:
                ice_diffused (1d array: year in int): archived ice d18O [permil]

            '''
            # ======================================================================
            # A.0: Initialization
            # ======================================================================
            # accumulation rate [m/yr]
            # note that the unit of pr_ann is [kg m-2 s-1], so need to divide by density [kg m-3] and convert the time
            yr2sec_factor = 3600*24*365.25
            accum = pr_ann/1000*yr2sec_factor

            # depth horizons (accumulation per year corresponding to depth moving down-core)
            bdown = accum[::-1]
            bmean = np.mean(bdown)
            depth = np.sum(bdown)
            depth_horizons = np.cumsum(bdown)
            dz = np.min(depth_horizons)/10.  # step in depth [m]

            Tmean = np.mean(tas_ann)  # unit in [K]
            Pmean = np.mean(psl_ann)*9.8692e-6  # unit in [Atm]

            # contants
            rho_s = 300.  # kg/m^3, surface density
            rho_d = 822.  # kg/m^2, density at which ice becomes impermeable to diffusion
            rho_i = 920.  # kg/m^3, density of solid ice

            # ======================================================================
            # A.1: Compaction Model
            # ======================================================================
            z = np.arange(0, depth, dz) + dz  # linear depth scale

            # set density profile by calling densification function
            rho, zieq, t = densification(Tmean, bmean, rho_s, z)

            rho = rho[:len(z)]  # cutoff the end
            time_d = np.cumsum(dz/bmean*rho/rho_i)
            ts = time_d*yr2sec_factor  # convert time in years to ts in seconds

            # integrate diffusivity along the density gradient to obtain diffusion length
            D = diffusivity(rho, Tmean, Pmean, rho_d, bmean)

            D = D[:-1]
            rho = rho[:-1]
            diffs = np.diff(z)/np.diff(time_d)
            diffs = diffs[:-1]

            # Integration using the trapezoidal method

            # IMPORTANT: once the ice reaches crtiical density (solid ice), there will no longer
            # be any diffusion. There is also numerical instability at that point. Set Sigma=1E-13 for all
            # points below that threshold.

            # Set to 915 to be safe.
            solidice = np.where(rho >= rho_d-5.0)
            diffusion = np.where(rho < rho_d-5.0)

            dt = np.diff(ts)
            sigma_sqrd_dummy = 2*np.power(rho, 2)*dt*D
            sigma_sqrd = integrate.cumtrapz(sigma_sqrd_dummy)
            diffusion_array = diffusion[0]
            diffusion_array = diffusion_array[diffusion_array < len(sigma_sqrd)]  # fzhu: to avoid the boundary index error
            diffusion = np.array(diffusion_array)

            #  rho=rho[0:-1] # modified by fzhu to fix inconsistency of array size
            #  sigma=np.zeros((len(rho)+1)) # modified by fzhu to fix inconsistency of array size
            sigma = np.zeros((len(rho)))
            sigma[diffusion] = np.sqrt(1/np.power(rho[diffusion],2)*sigma_sqrd[diffusion]) # modified by fzhu to fix inconsistency of array size
            #sigma[solidice]=np.nanmax(sigma) #max diffusion length in base of core // set in a better way. max(sigma)
            sigma[solidice] = sigma[diffusion][-1]
            sigma = sigma[:-1]

            # ======================================================================
            # A.2. Diffusion Profile
            # ======================================================================
            # Load water isotope series
            del18 = np.flipud(d18Oice)  # NOTE YOU MIGHT NOT NEED FLIP UD here. Our data goes forward in time.

            # interpolate over depths to get an array of dz values corresponding to isotope values for convolution/diffusion
            iso_interp = np.interp(z, depth_horizons, del18)

            # Return a warning if the kernel length is approaching 1/2 that of the timeseries.
            # This will result in spurious numerical effects.

            zp = np.arange(-100, 100, dz)
            if (len(zp) >= 0.5*len(z)):
                print("Warning: convolution kernel length (zp) is approaching that of half the length of timeseries. Kernel being clipped.")
                bound = 0.20*len(z)*dz
                zp = np.arange(-bound, bound, dz)

            #  print('start for loop ...')
            #  start_time = time.time()

            rm = np.nanmean(iso_interp)
            cdel = iso_interp-rm

            diffused_final = np.zeros(len(iso_interp))
            if nproc == 1:
                for i in tqdm(range(len(sigma))):
                    sig = sigma[i]
                    part1 = 1./(sig*np.sqrt(2.*np.pi))
                    part2 = np.exp(-zp**2/(2*sig**2))
                    G = part1*part2
                    #  diffused = np.convolve(G, cdel, mode='same')*dz  # fzhu: this is way too slow
                    diffused = signal.fftconvolve(cdel, G, mode='same')*dz  # put cdel in the front to keep the same length as before
                    diffused += rm  # remove mean and then put back
                    diffused_final[i] = diffused[i]

            else:
                #  print('Multiprocessing: nproc = {}'.format(nproc))

                def conv(sig, i):
                    part1 = 1./(sig*np.sqrt(2.*np.pi))
                    part2 = np.exp(-zp**2/(2*sig**2))
                    G = part1*part2
                    diffused = signal.fftconvolve(cdel, G, mode='same')*dz
                    diffused += rm  # remove mean and then put back

                    return diffused[i]

                res = Pool(nproc).map(conv, sigma, range(len(sigma)))
                diffused_final[:len(res)] = np.array(res)

            #  print('for loop: {:0.2f} s'.format(time.time()-start_time))

            # take off the first few and last few points used in convolution
            diffused_timeseries = diffused_final[0:-3]

            # Now we need to pack our data back into single year data units based on the depths and year interpolated data
            final_iso = np.interp(depth_horizons, z[0:-3], diffused_timeseries)
            ice_diffused = final_iso

            return ice_diffused

        ################
        # run the model

        # annualize the data
        tas_ann = self.pobj.clim[self.tas_name].annualize()
        pr_ann = self.pobj.clim[self.pr_name].annualize()
        psl_ann = self.pobj.clim[self.psl_name].annualize()

        # sensor model
        d18O_ice = ice_sensor(self.pobj.clim[self.pr_name].time, self.pobj.clim[self.d18O_name].da.values, self.pobj.clim[self.pr_name].da.values)

        # diffuse model
        if nproc is None: nproc = cpu_count()
        ice_diffused = ice_archive(d18O_ice, pr_ann.da.values, tas_ann.da.values, psl_ann.da.values, nproc=nproc)

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=tas_ann.time,
            value=ice_diffused[::-1],
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
        )

        return pp

class Lake_VarveThickness():
    ''' The varve thickness model.

    It takes summer temperature as input (JJA for NH and DJF for SH).
    '''
    def __init__(self, pobj=None, model_tas_name='model.tas', H=None, shape=None, mean=None, SNR=None, seed=None,
                 climate_required=['tas']):
        self.pobj = pobj
        self.H = H
        self.shape = shape
        self.mean = mean
        self.SNR = SNR
        self.seed = seed
        self.model_tas_name = model_tas_name
        self.climate_required = climate_required

    def forward(self):

        def simpleVarveModel(signal, H=0.75, shape=1.5, mean=1, SNR=.25, seed=0):
            ''' The python version of the simple varve model (in R lang) by Nick McKay
                Adapted by Feng Zhu

            Args:
                signal (array): the signal matrix, each row is a time series
                H (float): Hurst index, should be in (0, 1)
            '''
            # first, gammify the input
            gamSig = gammify(signal, shape=shape, mean=mean, seed=seed)

            # create gammafied autocorrelated fractal brownian motion series
            #     wa = Spectral.WaveletAnalysis()
            #     fBm_ts = wa.fBMsim(N=np.size(signal), H=H)
            fBm_ts = fbm.fgn(np.size(signal), H)
            gamNoise = gammify(fBm_ts, shape=shape, mean=mean, seed=seed)

            # combine the signal with the noise, based on the SNR
            varves = (gamSig*SNR + gamNoise*(1/SNR)) / (SNR + 1/SNR)

            res = {
                'signal': signal,
                'gamSig': gamSig,
                'fBm_ts': fBm_ts,
                'gamNoise': gamNoise,
                'varves': varves,
                'H': H,
                'shape': shape,
                'mean': mean,
                'SNR': SNR,
                'seed': seed,
            }

            return res

        def gammify(X, shape=1.5, mean=1, jitter=False, seed=0):
            ''' Transform each **row** of data matrix X to a gaussian distribution using the inverse Rosenblatt transform
            '''
            X = np.matrix(X)
            n = np.shape(X)[0]  # number of rows
            p = np.shape(X)[1]  # number of columns

            np.random.seed(seed)
            if jitter:
                # add tiny random numbers to aviod ties
                X += np.random.normal(0, np.std(X)/1e6, n*p).reshape(n, p)

            Xn = np.matrix(np.zeros(n*p).reshape(n, p))
            for j in range(n):
                # sort the data in ascending order and retain permutation indices
                R = stats.rankdata(X[j, :])
                # the cumulative distribution function
                CDF = R/p - 1/(2*p)
                # apply the inverse Rosenblatt transformation
                rate = shape/mean
                Xn[j, :] = stats.gamma.ppf(CDF, shape, scale=1/rate)  # Xn is now gamma distributed

            return Xn
        ###########################

        # update the params
        kwargs = {
            'H': 0.75,
            'shape': 1.5,
            'mean': 1,
            'SNR': 0.25,
            'seed': 0,
        }

        input_args = {
            'H': self.H,
            'shape': self.shape,
            'mean': self.mean,
            'SNR': self.SNR,
            'seed': self.seed,
        }

        for k, v in kwargs.items():
            if input_args[k] is not None:
                kwargs[k] = input_args[k]

        # run the model
        if self.pobj.lat > 0:
            tas_ann = self.pobj.clim[self.model_tas_name].annualize(months=[6, 7, 8])
        else:
            tas_ann = self.pobj.clim[self.model_tas_name].annualize(months=[12, 1, 2])

        varve_res = simpleVarveModel(tas_ann.da.values, **kwargs)

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=tas_ann.time,
            value=np.array(varve_res['varves'])[0],
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
        )

        return pp

class Coral_SrCa:
    ''' The coral Sr/Ca model
    '''
    def __init__(self, pobj=None, model_tos_name='model.tos', b=10.553, a=None, seed=None, climate_required='tos'):
        self.pobj = pobj
        self.model_tos_name = model_tos_name
        self.b = b
        self.a = a
        self.seed = seed
        self.climate_required = climate_required

    def forward(self):
        ''' Sensor model for Coral Sr/Ca = a * tos + b

        Args:
            tos (1-D array): sea surface temperature in [degC]
        '''
        if self.a is None:
            mu = -0.06
            std = 0.01
            self.a = stats.norm.rvs(loc=mu, scale=std, random_state=self.seed)

        SrCa = self.a*self.pobj.clim[self.model_tos_name].da.values + self.b

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=self.pobj.clim[self.model_tos_name].time,
            value=SrCa,
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
        )

        return pp

class Coral_d18O:
    ''' The PSM is based on the forward model published by [Thompson, 2011]:
       <Thompson, D. M., T. R. Ault, M. N. Evans, J. E. Cole, and J. Emile-Geay (2011),
       Comparison of observed and simulated tropical climate trends using a forward
       model of coral \u03b418O, Geophys.Res.Lett., 38, L14706, doi:10.1029/2011GL048224.>
       Returns a numpy array that is the same size and shape as the input vectors for SST, SSS.
    '''
    def __init__(self, pobj=None, model_tos_name='model.tos', model_d18Osw_name='model.d18Osw',
        climate_required=['tos', 'd18Osw'], species='default',
        b1=0.3007062, b2=0.2619054, b3=0.436509, b4=0.1552032, b5=0.15):
        self.pobj = pobj
        self.model_tos_name = model_tos_name
        self.model_d18Osw_name = model_d18Osw_name
        self.species = species
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.climate_required = climate_required

    def forward(self):
        def pseudocoral(sst, sss=None, d18O=None, species="default", lat=None, lon=None, 
                b1=0.3007062, b2=0.2619054, b3=0.436509, b4=0.1552032, b5=0.15):

            """
            DOCSTRING: Function 'pseudocoral' produces a d18O-coral record given SST, SSS, and global position.
               The function is based on the forward model published by [Thompson, 2011]:
               <Thompson, D. M., T. R. Ault, M. N. Evans, J. E. Cole, and J. Emile-Geay (2011),
               Comparison of observed and simulated tropical climate trends using a forward
               model of coral \u03b418O, Geophys.Res.Lett., 38, L14706, doi:10.1029/2011GL048224.>
               Returns a numpy array that is the same size and shape as the input vectors for SST, SSS.

            Input parameters:
                Latitude    [lat]       [-90, 90]
                Longitude   [lon]       [0, 360]
                SST         [SST]       SST ANOMALY Units = degrees Celsius
                SSS         [SSS]       SSS ANOMALY Units = PSU
                Please note that input files must be read in as 1-D vectors.

            Output:
                Returns a Numpy Array, ['coral'], which is saved in the main program.
                Optional Inputs: Please note that the function will use default params for
                d18O vs. SSS, a, and b unless specified and placed in the appropriate location as shown below.
                delta_18O [permil]: If the user does not have an input field for d18O, you must put -1 in its
                position in the call and use the equation for SSS below.
                **Note: [always use d18O (seawater) if available!]**
                The user may specify values for parameters a and b in the coral d18O forward model,
                which are currently set to the following default values:
                a = coral - SST regression, accounting for species-dependent variations in the SST-18O dependence.
                The code assigns this value universally at -.22, but published values vary by species/genus (i.e. Moses, 2006)
                Specify keyword input 'species=' to set slope:
                species =
                        "Default":      a = -0.22
                        "Porites_sp":   a = -.26178
                        "Porites_lob":  a = -.19646
                        "Porites_lut":  a = -.17391
                        "Porites_aus":  a = -.21
                        "Montast":      a = -.22124
                        "Diploas":      a = -.14992
                b = d18O-SSS slope, as defined by location in: Legrande & Schmidt, (2006)
                        <LeGrande, A. N., and G. A. Schmidt (2006), Global gridded data set of the oxygen isotopic          composition in seawater, Geophys. Res. Lett., 33, L12604, doi:10.1029/2006GL026011.>
                        b1 = 0.31   [Red Sea]
                        b2 = 0.27   [Tropical Pacific]
                        b3 = 0.45   [South Pacific]
                        b4 = 0.16   [Indian Ocean]
                        b5 = 0.15   [Tropical Atlantic Ocean/Caribbean]
            Example of use: Call function 'pseudocoral' to calculate d18O_coral timeseries, lacking d18O sw
            but specifying own numeric parameters for a, b.
            pseudocoral(lat,lon,SST,SSS,-1,species="default",b1,b2,b3,b4,b5)
            """
        #====================================================================



        # Define slope of coral response to SST, a
            #Also, have you considered accounting for species-dependent variations in the a1 value (the SST-18O dependence)?
            #The code assigns this value universally at -.22, but published values vary by species/genus (i.e. Moses, 2006).
            #I've copied and pasted a chunk of code from one of my scripts that does this brute-force... Hope it helps.

            if species == "Porites_sp":
                a=-.26178
            elif species == "Porites_lob":
                a=-.19646
            elif species == "Porites_lut":
                a=-.17391
            elif species == "Porites_aus":
                a=-.21
            elif species== "Montast":
                a=-.22124
            elif species == "Diploas":
                a=-.14992
            elif species == "default":
                a=-0.22
            else:
                a=-0.22

        # Define SSS-d18O(sw) slopes by region. Note that we multiply all slopes by VPDB/VSMOW [0.0020052/0.0020672] ~ 0.97 to correct d18O values.

        # Because d18Osw is reported relative to VSMOW, we have to convert the slope to VPDB so
        # that the units of the forward model output are in VPDB (and thus directly comparable
        # to published d18O records, in units of VPDB). The way this affects the forward model is
        # in the d18Osw-SSS slope.

            V_corr = 0.97002

            slope1 = V_corr*b1      # Red Sea
            slope2 = V_corr*b2      # Tropical Pacific
            slope3 = V_corr*b3      # South Pacific
            slope4 = V_corr*b4      # Indian Ocean
            slope5 = V_corr*b5      # Tropical Atlantic

        # Given Lat/Lon pair, assign correct value to 'b'
            if lon>=32.83 and lon<=43.5 and lat>=12.38 and lat<=28.5:    # Red Sea
                b=slope1
            elif lon<=120.:              # Indian Ocean
                b=slope4
            elif lon>=270. and lat>=10.:    #Tropical Atlantic ~300 E longitude
                b=slope5
            elif lat> -5. and lat<=13.:   # Tropical Pacific
                b=slope2
            elif lat<= -5.:              # South Pacific
                b=slope3
            else:
                b=slope2            # Default = Trop. Pacific.

        #====================================================================

        # Form an array of pseudocoral data
            if np.min(sst) > 200:
                # if in [K], convert to [degC]
                sst -= 273.15

            if d18O is None:
                coral = a*sst+b*sss
            else:
                coral = a*sst + 0.97002*d18O  # SMOW/PDB = 0.97002

        #====================================================================

            return coral

        ####################
        # run the model

        value = pseudocoral(
            self.pobj.clim[self.model_tos_name].da.values,
            d18O=self.pobj.clim[self.model_d18Osw_name].da.values,
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            species=self.species,
            b1=self.b1,
            b2=self.b2,
            b3=self.b3,
            b4=self.b4,
            b5=self.b5,
        )

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=self.pobj.clim[self.model_tos_name].time,
            value=value,
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
        )

        return pp


class VSLite:
    ''' The VS-Lite tree-ring width model that takes monthly tas, pr as input.
    '''
    def __init__(self, pobj=None,
            obs_tas_name='obs.tas', obs_pr_name='obs.pr',
            model_tas_name='model.tas', model_pr_name='model.pr',
            climate_required=['tas', 'pr'],
        ):
        self.pobj = pobj
        self.obs_tas_name = obs_tas_name
        self.obs_pr_name = obs_pr_name
        self.model_tas_name = model_tas_name
        self.model_pr_name = model_pr_name
        self.climate_required = climate_required

    def calibrate(self, calib_period=[1901, 2000], method='Bayesian'):
        proxy_time = self.pobj.time
        proxy_value = self.pobj.value

        obs_tas_time = self.pobj.clim[self.obs_tas_name].time
        obs_tas_value = self.pobj.clim[self.obs_tas_name].da.values
        if np.min(obs_tas_value) > 200:
            # if in [K], convert to [degC]
            obs_tas_value -= 273.15

        obs_pr_time = self.pobj.clim[self.obs_pr_name].time
        obs_pr_value = self.pobj.clim[self.obs_pr_name].da.values
        if np.max(obs_pr_value) < 1:
            # if in [kg m-2 s-1], convert to [mm/month]
            obs_pr_value *=  3600*24*30

        proxy_syear = np.min(proxy_time)
        proxy_eyear = np.max(proxy_time)
        obs_syear = np.min(np.floor(obs_tas_time))
        obs_eyear = np.max(np.floor(obs_tas_time))
        calib_syear = np.max([proxy_syear, obs_syear, calib_period[0]])
        calib_eyear = np.min([proxy_eyear, obs_eyear, calib_period[1]])
        if calib_period is not None:
            mask_T = (obs_tas_time>=calib_syear) & (obs_tas_time<calib_eyear)
            mask_P = (obs_pr_time>=calib_syear) & (obs_pr_time<calib_eyear)
            mask_TRW = (proxy_time>=calib_syear) & (proxy_time<calib_eyear)
            T = obs_tas_value[mask_T]
            P = obs_pr_value[mask_P]
            TRW = proxy_value[mask_TRW]
        else:
            T = obs_tas_value
            P = obs_pr_value
            TRW = proxy_value

        if method == 'Bayesian':
            res_dict = PyVSL.est_params(T, P, self.pobj.lat, TRW)
        else:
            raise ValueError('Wrong estimation method for VSLite!')

        self.calib_details = {
            'T1': res_dict['T1'],
            'T2': res_dict['T2'],
            'M1': res_dict['M1'],
            'M2': res_dict['M2'],
        }

    def forward(self, **vsl_kwargs):
        tas_time = self.pobj.clim[self.model_tas_name].time
        syear = np.min(np.floor(tas_time))
        eyear = np.max(np.floor(tas_time))

        if hasattr(self, 'calib_details'):
            kwargs = {}
            for k in ['T1', 'T2', 'M1', 'M2']:
                kwargs[k] = self.calib_details[k]

            vsl_kwargs.update(kwargs)

        model_tas_value = self.pobj.clim[self.model_tas_name].da.values
        model_pr_value = self.pobj.clim[self.model_pr_name].da.values

        if np.min(model_tas_value) > 200:
            # if in [K], convert to [degC]
            model_tas_value -= 273.15

        if np.max(model_pr_value) < 1:
            # if in [kg m-2 s-1], convert to [mm/month]
            model_pr_value *=  3600*24*30

        vsl_res = PyVSL.VSL(
            syear, eyear, self.pobj.lat,
            model_tas_value,
            model_pr_value, **vsl_kwargs
        )

        pp = ProxyRecord(
            pid=self.pobj.pid,
            time=np.arange(syear, eyear+1),
            value=vsl_res['trw'],
            lat=self.pobj.lat,
            lon=self.pobj.lon,
            ptype=self.pobj.ptype,
            value_name=self.pobj.value_name,
            value_unit=self.pobj.value_unit,
            time_name=self.pobj.time_name,
            time_unit=self.pobj.time_unit,
        )

        return pp