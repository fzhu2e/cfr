import numpy as np
import xarray as xr
import x4c
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from .. import utils

def gaspari_cohn(dist, loc_radius):
    '''
    Vectorized Gaspari-Cohn localization function.
    
    Args:
        dist (ndarray): Distance(s) between model state and observation.
        loc_radius (float): Localization radius (distance beyond which covariance is set to zero).

    Reference:
        Gaspari, G., Cohn, S.E., 1999. Construction of correlation functions in two and three dimensions.
        Quarterly Journal of the Royal Meteorological Society 125, 723-757. https://doi.org/10.1002/qj.49712555417
    '''
    # Normalize the distances
    r = np.abs(dist) / loc_radius
    
    # Initialize the result array with zeros
    f = np.zeros_like(r)
    
    # Eq. (4.10) in Gaaspari & Coh (1999)
    mask1 = r <= 1
    f[mask1] = -r[mask1]**5 / 4 + r[mask1]**4 / 2 + 5/8 * r[mask1]**3 - 5/3 * r[mask1]**2 + 1
    
    mask2 = (r > 1) & (r <= 2)
    f[mask2] = r[mask2]**5 / 12 - r[mask2]**4 / 2 + 5/8 * r[mask2]**3  + 5/3 * r[mask2]**2 - 5 * r[mask2] + 4 - 2/3 / r[mask2]

    f[f<0] = 0 # force f >= 0
    return f

def gaspari_cohn_DASH(dist, loc_radius, scale=0.5):
    """
    Implements a Gaspari-Cohn 5th order polynomial localization function following DASH.
    
    Parameters:
        dist (ndarray): An array of distances.
        loc_radius (float): The cutoff radius, beyond which weights are zero.
        scale (float or str, optional): The length scale for the polynomial.
            Must be on the interval 0 < scale <= 0.5, or 'optimal' to use the optimal
            length scale as described by Lorenc (2003). Default is 0.5.
    
    Returns:
        weights (ndarray): Covariance localization weights with the same shape as distances.
    """
    # Set the scale if 'optimal' is specified
    if isinstance(scale, str) and scale == 'optimal':
        scale = np.sqrt(10 / 3)
    
    # Define length scale and localization radius
    c = scale * loc_radius
    
    # Preallocate weights array with ones
    weights = np.ones_like(dist)
    
    # Calculate mask arrays for the different distance ranges
    outside_radius = dist > loc_radius
    inside_scale = dist <= c
    in_between = ~inside_scale & ~outside_radius

    # Apply Gaspari-Cohn polynomial
    X = dist / c
    weights[outside_radius] = 0
    weights[in_between] = X[in_between]**5 / 12 - 0.5 * X[in_between]**4 + 0.625 * X[in_between]**3 + (5 / 3) * X[in_between]**2 - 5 * X[in_between] + 4 - 2 / (3 * X[in_between])
    weights[inside_scale] = -0.25 * X[inside_scale]**5 + 0.5 * X[inside_scale]**4 + 0.625 * X[inside_scale]**3 - (5 / 3) * X[inside_scale]**2 + 1
    
    # Ensure weights are non-negative due to rounding errors
    weights[weights < 0] = 0
    
    return weights



class EnSRF:
    '''Ensemble Square Root Filter (EnSRF) with localization.

    Implements the deterministic EnSRF update following Whitaker & Hamill (2002),
    with Gaspari-Cohn covariance localization.

    Parameters
    ----------
    X : ndarray (n, N)
        Ensemble of prior state vectors.
    Y : ndarray (m, N)
        Ensemble of forward estimates, Y = H(X).
    y : ndarray (m, 1)
        Observation vector.
    R : ndarray (m, m)
        Observation error covariance matrix.
    L : ndarray (n, m), optional
        Localization matrix for state-observation covariance.
    Lobs : ndarray (m, m), optional
        Localization matrix for observation-observation covariance.
    '''
    def __init__(self, X=None, Y=None, y=None, R=None, L=None, Lobs=None):
        self.X = X            # ensemble of the prior state vectors (n x N)
        self.Y = Y            # ensemble of the forward estimates (m x N); Y=H(X)
        self.y = y            # observations (m x 1)
        self.R = R            # obs err matrix (m x m)
        self.L = L            # localization matrix (n x m)
        self.Lobs = Lobs      # localization matrix (m x m)

    def update(self, debug=False):
        ''' Perform an EnSRF update with localization. '''
        N = self.X.shape[1]  # Ensemble size

        # Compute the ensemble mean
        Xm = np.mean(self.X, axis=1, keepdims=True)
        Xp = self.X - Xm

        Ym = np.mean(self.Y, axis=1, keepdims=True)
        Yp = self.Y - Ym

        # Observation error covariance matrix
        Ycov = (Yp @ Yp.T) / (N - 1)

        # Localize the obs err covariance matrix
        if self.Lobs is not None:
            Ycov_loc = Ycov * self.Lobs
        else:
            Ycov_loc = Ycov

        C =  Ycov_loc + self.R

        # Kalman gain matrix
        XYcov = (Xp @ Yp.T) / (N - 1)

        # Localize the Kalman gain
        if self.L is not None:
            XYcov_loc = XYcov * self.L
        else:
            XYcov_loc = XYcov

        K = XYcov_loc @ np.linalg.inv(C)

        # Observation innovation
        d = self.y - Ym

        # Update the ensemble mean
        Xm_updated = Xm + K @ d

        # Update the ensemble perturbations
        # self.T = np.eye(N) - (Yp.T @ np.linalg.inv(C)) @ Yp / (N - 1)   # Eq. (9) in Tippett et al. (2003), single obs case
        # generalized to multiple obs case
        S = (Yp.T @ np.linalg.inv(self.R) @ Yp) / (N - 1)
        eigvals, eigvecs = np.linalg.eigh(np.eye(N) + S)
        self.T = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T

        Xp_updated = Xp @ self.T

        # Combine updated mean and perturbations
        self.X_updated = Xm_updated + Xp_updated

        if debug:
            self.Xm = Xm
            self.Xp = Xp
            self.Xm_updated = Xm_updated
            self.Xp_updated = Xp_updated
            self.Ym = Ym
            self.Yp = Yp
            self.C = C
            self.K = K
            self.d = d
            self.Ycov = Ycov
            self.Ycov_loc = Ycov_loc
            self.XYcov = XYcov
            self.XYcov_loc = XYcov_loc

    def plot_T(self, cmap='viridis'):
        ''' Check the T matrix
        '''
        data = self.T.T
        nx, ny = np.shape(data)

        # Define color levels
        vmin, vmax = 0, 1
        num_colors = 11
        levels = np.linspace(vmin, vmax, num_colors)

        # Create a discrete colormap
        cmap = plt.get_cmap(cmap, num_colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        cmap.set_under('k')
        cmap.set_over('red')

        # Create X, Y grid for pcolormesh
        x = np.arange(nx + 1)
        y = np.arange(ny + 1)
        X, Y = np.meshgrid(x, y)

        # Create the plot
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(X, Y, data.T, cmap=cmap, norm=norm, shading='auto')

        # Customize colorbar
        cbar = plt.colorbar(pcm, ax=ax, extend="both")
        cbar.set_ticks(levels)
        cbar.set_label('T Matrix')

        ax.set_title(f'min={data.min():.2f}, max={data.max():.2f}')
        return fig, ax

class EnSRF_DASH:
    '''EnSRF variant following the DASH formulation.

    Uses an alternative square root update for ensemble perturbations
    based on matrix square root of the analysis covariance.

    Parameters
    ----------
    X : ndarray (n, N)
        Ensemble of prior state vectors.
    Y : ndarray (m, N)
        Ensemble of forward estimates, Y = H(X).
    y : ndarray (m, 1)
        Observation vector.
    R : ndarray (m, m)
        Observation error covariance matrix.
    L : ndarray (n, m), optional
        Localization matrix for state-observation covariance.
    Lobs : ndarray (m, m), optional
        Localization matrix for observation-observation covariance.
    '''
    def __init__(self, X=None, Y=None, y=None, R=None, L=None, Lobs=None):
        self.X = X            # ensemble of the prior state vectors (n x N)
        self.Y = Y            # ensemble of the forward estimates (m x N); Y=H(X)
        self.y = y            # observations (m x 1)
        self.R = R            # obs err matrix (m x m)
        self.L = L            # localization matrix (n x m)
        self.Lobs = Lobs      # localization matrix (m x m)

    def update(self, debug=False):
        ''' Perform an EnSRF update with localization. '''
        N = self.X.shape[1]  # Ensemble size

        # Compute the ensemble mean
        Xm = np.mean(self.X, axis=1, keepdims=True)
        Xp = self.X - Xm

        Ym = np.mean(self.Y, axis=1, keepdims=True)
        Yp = self.Y - Ym

        # Observation error covariance matrix
        Ycov = (Yp @ Yp.T) / (N - 1)

        # Localize the obs err covariance matrix
        if self.Lobs is not None:
            Ycov_loc = self.Lobs * Ycov
        else:
            Ycov_loc = Ycov

        C =  Ycov_loc + self.R

        # Kalman gain matrix
        XYcov = (Xp @ Yp.T) / (N - 1)

        # Localize the Kalman gain
        if self.L is not None:
            XYcov_loc = self.L * XYcov
        else:
            XYcov_loc = XYcov

        K = XYcov_loc @ np.linalg.inv(C)

        # Observation innovation
        d = self.y - Ym

        # Update the ensemble mean
        Xm_updated = Xm + K @ d

        # Update the ensemble perturbations
        Ksqrt = sqrtm(C)
        Ksqrt_inv_transpose = np.linalg.inv(Ksqrt).T
        Rcov_sqrt = sqrtm(self.R)
        Ka = K @ Ksqrt_inv_transpose @ np.linalg.inv(Ksqrt + Rcov_sqrt)
        Xp_updated = Xp - Ka @ Yp

        # Combine updated mean and perturbations
        self.X_updated = Xm_updated + Xp_updated

        if debug:
            self.Xm = Xm
            self.Xp = Xp
            self.Ym = Ym
            self.Yp = Yp
            self.C = C
            self.K = K
            self.d = d


class EnOI:
    '''Ensemble Optimal Interpolation (EnOI).

    Updates a target state using a static ensemble for covariance estimation,
    rather than updating the ensemble itself.

    Parameters
    ----------
    X_target : ndarray (n, 1)
        Target state vector to be updated (e.g., monthly prior).
    X : ndarray (n, N)
        Static ensemble of prior state vectors for covariance estimation.
    Y : ndarray (m, N)
        Ensemble of forward estimates, Y = H(X).
    y : ndarray (m, 1)
        Observation vector.
    R : ndarray (m, m)
        Observation error covariance matrix.
    L : ndarray (n, m), optional
        Localization matrix.
    '''
    def __init__(self, X_target=None, X=None, Y=None, y=None, R=None, L=None):
        self.X_target = X_target   # the **monthly** prior state vectors (n x 1)
        self.X = X         # ensemble of the prior state vectors (n x N)
        self.Y = Y         # ensemble of the forward estimates (m x N); Y=H(X)
        self.y = y         # observations (m x 1)
        self.R = R         # obs err matrix (m x m)
        self.L = L         # localization matrix (n x m)

    def update(self, debug=False):
        ''' Perform an EnOI update with localization. '''
        N = self.X.shape[1]  # Ensemble size

        # Compute the ensemble mean
        Xm = np.mean(self.X, axis=1, keepdims=True)
        Xp = self.X - Xm

        Ym = np.mean(self.Y, axis=1, keepdims=True)
        Yp = self.Y - Ym

        # Observation error covariance matrix
        C = (Yp @ Yp.T) / (N - 1) + self.R

        # Kalman gain matrix
        K = (Xp @ Yp.T) / (N - 1) @ np.linalg.inv(C)

        # Localize the Kalman gain
        if self.L is not None:
            K_loc = K * self.L
        else:
            K_loc = K

        # Observation innovation
        d = self.y - Ym

        # the increment
        inc = K_loc @ d

        # update
        self.X_target_updated = self.X_target + inc

        if debug:
            self.Xm = Xm
            self.Xp = Xp
            self.Ym = Ym
            self.Yp = Yp
            self.C = C
            self.K = K
            self.K_loc = K_loc
            self.d = d


class Solver:
    '''High-level data assimilation solver.

    Orchestrates the full DA workflow: proxy system modeling, localization,
    ensemble update, and validation against truth.

    Parameters
    ----------
    prior : Prior
        The prior ensemble.
    obs : Obs
        The observation database.
    prior_target : Prior, optional
        Target prior for EnOI mode.
    '''
    def __init__(self, prior=None, obs=None, prior_target=None):
        self.prior = prior.copy()
        self.obs = obs
        self.prior_target = prior_target

    def prep(self, recon_period=None, recon_season=list(range(1, 13)),
             localize=True, loc_method='gaspari_cohn', loc_radius=2500, dist_vsf=1,
             startover=False, get_clim_kws={}, **fwd_kws):
        ''' Prepare Y=H(X) and the localization matrix for DA

        Args:
            dist_vsf (float, list of float): the vertical scaling factor of the distance

        '''
        if recon_period is None:
            self.time = self.obs.ds.time
        else:
            self.time = self.obs.ds.time.sel(time=slice(recon_period[0], recon_period[-1]))

        if startover or not hasattr(self.prior, 'Y'):
            utils.p_header('>>> Proxy System Modeling: Y = H(X)')
            self.obs.get_clim(self.prior, **get_clim_kws)
            self.obs.get_pseudo(**fwd_kws)
            self.prior.get_Y(self.obs)

        if 'month' in self.prior.ds.dims and recon_season is not None:
            utils.p_header(f'>>> Annualizing prior w/ season: {recon_season}')
            self.prior.ds = self.prior.annualize(months=recon_season)

        if localize and (startover or not hasattr(self.prior, 'dist')):
            loc_func = {
                'gaspari_cohn': gaspari_cohn,
                'gaspari_cohn_DASH': gaspari_cohn_DASH,
            }
            utils.p_header('>>> Computing the localization matrix')
            self.prior.get_dist(self.obs, s=dist_vsf)
            self.L = loc_func[loc_method](self.prior.dist, loc_radius)
            self.obs.get_dist()
            self.Lobs = loc_func[loc_method](self.obs.dist, loc_radius)
        else:
            self.L = None
            self.Lobs = None

    def run_t(self, t, method='EnSRF', debug=False):
        algo = {
            'EnSRF': EnSRF,
            'EnSRF_DASH': EnSRF_DASH,
            'EnOI': EnOI,
        }

        idx = list(self.obs.ds.time).index(t)
        kws = {}
        for m in algo.keys():
            kws[m] = {
                'X': self.prior.X,
                'Y': self.prior.Y,
                'y': self.obs.y[idx][..., np.newaxis],
                'R': self.obs.R,
                'L': self.L,
                'Lobs': self.Lobs,
            }

        if self.prior_target is not None:
            kws['EnOI']['X_target'] = self.prior_target.X

        S = algo[method](**kws[method])
        S.update(debug=debug)

        # utils.p_header('>>> Formatting the posterior')
        if method in ['EnSRF', 'EnSRF_DASH']:
            post = utils.states2ds(S.X_updated, self.prior.ds)
        elif method == 'EnOI':
            post = utils.states2ds(S.X_target_updated, self.prior_target.ds)
        
        return S, post

    def run(self, method='EnSRF', debug=False, nproc=1, chunksize=10):
        utils.p_header('>>> DA update')
        if nproc == 1:
            res = [self.run_t(t, method=method, debug=debug) for t in tqdm(self.time, desc='Updating time slices')]
        else:
            run_t_partial = partial(self.run_t, method=method, debug=debug)
            with Pool(processes=nproc) as p:
                res = list(tqdm(p.imap_unordered(run_t_partial, self.time, chunksize=chunksize), total=len(self.time), desc='Updating time slices'))

        S, post = zip(*res)

        self.S = S
        self.post = xr.concat(post, dim='time').assign_coords({'time': self.time})

    def plot_valid(self, ds_target, ds_prior=None, vn=None, metric='R2', valid_period=None, **plot_kws):
        metric_dict = {
            'R2': r2_score,
            'MSE': mean_squared_error,
            # 'RMSE': mean_squared_error,
            'MAE': mean_absolute_error,
            'MAPE': mean_absolute_percentage_error,
        }
        _plot_kws = {
            'corr': dict(
                cmap='RdBu_r',
                levels=np.linspace(-1, 1, 21),
                cbar_kwargs={'ticks': np.linspace(-1, 1, 6), 'label': '$r$'},
                title='Corr(Posterior, Truth)',
                extend='neither',
            ),
            'R2': dict(
                cmap='RdBu_r',
                levels=np.linspace(0, 1, 21),
                cbar_kwargs={'ticks': np.linspace(0, 1, 6), 'label': '$R^2$'},
                title='$R^2$(Posterior, Truth)',
                extend='min',
            ),
            'MSE': dict(
                cmap='RdBu_r',
                levels=np.linspace(0, 4, 21),
                cbar_kwargs={'ticks': np.linspace(0, 4, 11), 'label': 'MSE'},
                title='MSE(Posterior, Truth)',
                extend='both',
            ),
            'RMSE': dict(
                cmap='RdBu_r',
                levels=np.linspace(0, 2, 21),
                cbar_kwargs={'ticks': np.linspace(0, 2, 11), 'label': 'RMSE'},
                title='RMSE(Posterior, Truth)',
                extend='max',
            ),
            'MAE': dict(
                cmap='RdBu_r',
                levels=np.linspace(0, 2, 21),
                cbar_kwargs={'ticks': np.linspace(0, 2, 11), 'label': 'MAE'},
                title='MAE(Posterior, Truth)',
                extend='both',
            ),
            'MAPE': dict(
                cmap='RdBu_r',
                levels=np.linspace(0, 2, 21),
                cbar_kwargs={'ticks': np.linspace(0, 2, 11), 'label': 'MAPE'},
                title='MAPE(Posterior, Truth)',
                extend='both',
            ),
        }

        if valid_period is None:
            valid_time = self.time
        else:
            valid_time = self.time.sel(time=slice(valid_period[0], valid_period[-1]))
            
        if metric in metric_dict:
            if ds_prior is None:
                valid_map = xr.apply_ufunc(
                    metric_dict[metric],
                    ds_target[vn].sel(time=valid_time),
                    self.post[vn].sel(time=valid_time).mean('ens'),
                    input_core_dims=[['time'], ['time']],  # Apply along the 'time' dimension
                    vectorize=True,
                )
                if metric == 'RMSE': valid_map = np.sqrt(valid_map)
                _plot_kws[metric].update(plot_kws)
                fig, ax = valid_map.x.plot(**_plot_kws[metric])
            else:
                valid_map_prior = xr.apply_ufunc(
                    metric_dict[metric],
                    ds_target[vn].sel(time=valid_time),
                    ds_prior[vn].sel(time=valid_time),
                    input_core_dims=[['time'], ['time']],  # Apply along the 'time' dimension
                    vectorize=True,
                )
                valid_map_post = xr.apply_ufunc(
                    metric_dict[metric],
                    ds_target[vn].sel(time=valid_time),
                    self.post[vn].sel(time=valid_time).mean('ens'),
                    input_core_dims=[['time'], ['time']],  # Apply along the 'time' dimension
                    vectorize=True,
                )
                if metric == 'RMSE':
                    valid_map_prior = np.sqrt(valid_map_prior)
                    valid_map_post = np.sqrt(valid_map_post)

                _plot_kws[metric].update(plot_kws)
                fig, ax = x4c.visual.subplots(
                    nrow=2, ncol=1,
                    ax_loc={'prior': (0, 0), 'post': (1, 0)},
                    projs={'prior': 'Robinson', 'post': 'Robinson'},
                    projs_kws={'prior': {'central_longitude': 180}, 'post': {'central_longitude': 180}},
                    figsize=(12, 6),
                    wspace=0,
                    hspace=0.2,
                )
                _plot_kws_prior = _plot_kws[metric].copy()
                _plot_kws_prior['title'] = _plot_kws_prior['title'].replace('Posterior', 'Prior')
                valid_map_prior.x.plot(ax=ax['prior'], **_plot_kws_prior)
                valid_map_post.x.plot(ax=ax['post'], **_plot_kws[metric])

        elif metric == 'corr':
            if ds_prior is None:
                valid_map = xr.corr(
                    ds_target[vn].sel(time=valid_time),
                    self.post[vn].sel(time=valid_time).mean('ens'),
                    dim='time',
                )
                _plot_kws[metric].update(plot_kws)
                fig, ax = valid_map.x.plot(**_plot_kws[metric])
            else:
                valid_map_prior = xr.corr(
                    ds_target[vn].sel(time=valid_time),
                    ds_prior[vn].sel(time=valid_time),
                    dim='time',
                )
                valid_map_post = xr.corr(
                    ds_target[vn].sel(time=valid_time),
                    self.post[vn].sel(time=valid_time).mean('ens'),
                    dim='time',
                )
                _plot_kws[metric].update(plot_kws)
                fig, ax = x4c.visual.subplots(
                    nrow=2, ncol=1,
                    ax_loc={'prior': (0, 0), 'post': (1, 0)},
                    projs={'prior': 'Robinson', 'post': 'Robinson'},
                    projs_kws={'prior': {'central_longitude': 180}, 'post': {'central_longitude': 180}},
                    figsize=(12, 6),
                    wspace=0,
                    hspace=0.2,
                )
                _plot_kws_prior = _plot_kws[metric].copy()
                _plot_kws_prior.update({'title': 'Corr(Prior, Truth)'})
                valid_map_prior.x.plot(ax=ax['prior'], **_plot_kws_prior)
                valid_map_post.x.plot(ax=ax['post'], **_plot_kws[metric])
        elif metric == 'RMSE':
            da_target = ds_target['tas'].sel(time=valid_time)
            da_post = self.post['tas'].sel(time=valid_time).mean('ens')
            RMSE_post = np.sqrt(((da_target-da_post)**2).mean(dim='time'))
            if ds_prior is not None:
                da_prior = ds_prior['tas'].sel(time=valid_time)
                RMSE_prior = np.sqrt(((da_target-da_prior)**2).mean(dim='time'))

            if ds_prior is None:
                _plot_kws[metric].update(plot_kws)
                fig, ax = RMSE_post.x.plot(**_plot_kws[metric])
            else:
                _plot_kws[metric].update(plot_kws)
                fig, ax = x4c.visual.subplots(
                    nrow=2, ncol=1,
                    ax_loc={'prior': (0, 0), 'post': (1, 0)},
                    projs={'prior': 'Robinson', 'post': 'Robinson'},
                    projs_kws={'prior': {'central_longitude': 180}, 'post': {'central_longitude': 180}},
                    figsize=(12, 6),
                    wspace=0,
                    hspace=0.2,
                )
                _plot_kws_prior = _plot_kws[metric].copy()
                _plot_kws_prior.update({'title': 'RMSE(Prior, Truth)'})
                RMSE_prior.x.plot(ax=ax['prior'], **_plot_kws_prior)
                RMSE_post.x.plot(ax=ax['post'], **_plot_kws[metric])

        elif metric == 'nino3.4':
            da_target = ds_target['tas'].sel(time=valid_time).x.geo_mean('nino3.4')
            da_post = self.post['tas'].sel(time=valid_time).x.geo_mean('nino3.4').mean('ens')
            if ds_prior is not None:
                da_prior = ds_prior['tas'].sel(time=valid_time).x.geo_mean('nino3.4')
                corr_prior = np.corrcoef(da_target.values, da_prior.values)[1, 0]
                # R2_prior = r2_score(da_target.values, da_prior.values)
                RMSE_prior = np.sqrt(mean_squared_error(da_target.values, da_prior.values))

            corr_post = np.corrcoef(da_target.values, da_post.values)[1, 0]
            # R2_post = r2_score(da_target.values, da_post.values)
            RMSE_post = np.sqrt(mean_squared_error(da_target.values, da_post.values))

            fig, ax = da_target.x.plot(color='k', label='Target')
            if ds_prior is not None:
                # da_prior.x.plot(ax=ax, label=f'Prior ($r$={corr_prior:.2f}, $R^2$={R2_prior:.2f}, RMSE={RMSE_prior:.2f})', color='tab:blue')
                da_prior.x.plot(ax=ax, label=f'Prior ($r$={corr_prior:.2f}, RMSE={RMSE_prior:.2f})', color='tab:blue')

            # da_post.x.plot(ax=ax, label=f'Posterior ($r$={corr_post:.2f}, $R^2$={R2_post:.2f}, RMSE={RMSE_post:.2f})', color='tab:orange')
            da_post.x.plot(ax=ax, label=f'Posterior ($r$={corr_post:.2f}, RMSE={RMSE_post:.2f})', color='tab:orange', lw=3)
            ax.legend(bbox_to_anchor=(1, 1))
            ax.set_title(f'NINO3.4')

        return fig, ax