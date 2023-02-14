import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import cm
import pickle

import matplotlib as mpl
import numpy as np
import os
from scipy import stats
from scipy.stats.mstats import mquantiles
from scipy.stats import cumfreq, gaussian_kde
from scipy.integrate import cumtrapz
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerLine2D
import pathlib
import pandas as pd

from cartopy import util as cutil
from . import utils

class PAGES2k:
    ''' A bunch of PAGES2k style settings
    '''
    archive_types = [
        'bivalve',
        'borehole',
        'coral',
        'documents',
        'ice',
        'hybrid',
        'lake',
        'marine',
        'sclerosponge',
        'speleothem',
        'tree',
    ]
    markers = ['p', 'p', 'o', 'v', 'd', '*', 's', 's', '8', 'D', '^']
    markers_dict = dict(zip(archive_types, markers))
    colors = [np.array([ 1.        ,  0.83984375,  0.        ]),
              np.array([ 0.73828125,  0.71484375,  0.41796875]),
              np.array([ 1.        ,  0.546875  ,  0.        ]),
              np.array([ 0.41015625,  0.41015625,  0.41015625]),
              np.array([ 0.52734375,  0.8046875 ,  0.97916667]),
              np.array([ 0.        ,  0.74609375,  1.        ]),
              np.array([ 0.25390625,  0.41015625,  0.87890625]),
              np.array([ 0.54296875,  0.26953125,  0.07421875]),
              np.array([ 1         ,           0,           0]),
              np.array([ 1.        ,  0.078125  ,  0.57421875]),
              np.array([ 0.1953125 ,  0.80078125,  0.1953125 ])]
    colors_dict = dict(zip(archive_types, colors))

class STYLE:
    colors_dict = {
        'coral.calc': sns.xkcd_rgb['yellow'],
        'coral.SrCa': sns.xkcd_rgb['orange'],
        'coral.d18O': sns.xkcd_rgb['amber'],
        'coral.d18Osw': sns.xkcd_rgb['light yellow'],
        'ice.melt': sns.xkcd_rgb['pale blue'],
        'ice.d18O': sns.xkcd_rgb['light blue'],
        'ice.dD': sns.xkcd_rgb['sky blue'],
        'ice.d-excess': sns.xkcd_rgb['sky blue'],
        'ice.isotope_diffusion': sns.xkcd_rgb['sky blue'],
        'ice.hybrid': sns.xkcd_rgb['sky blue'],
        'tree.TRW': sns.xkcd_rgb['green'],
        'tree.MXD': sns.xkcd_rgb['forest green'],
        'tree.ENSO': sns.xkcd_rgb['sea green'],
        'pollen.temp': sns.xkcd_rgb['red'],
        'land.temp': sns.xkcd_rgb['red'],
        'ocean.temp': sns.xkcd_rgb['blue'],
        'tas': sns.xkcd_rgb['pale red'],
        'pr': sns.xkcd_rgb['aqua'],
        'speleothem.d18O': sns.xkcd_rgb['light brown'],
        'speleothem.dD': sns.xkcd_rgb['brown'],
        'bivalve.d18O': sns.xkcd_rgb['gold'],
        'marine.TEX86': sns.xkcd_rgb['brown'],
        'marine.MgCa': sns.xkcd_rgb['brown'],
        'marine.d18O': sns.xkcd_rgb['brown'],
        'marine.MAT': sns.xkcd_rgb['brown'],
        'marine.alkenone': sns.xkcd_rgb['brown'],
        'marine.foram': sns.xkcd_rgb['brown'],
        'marine.diatom': sns.xkcd_rgb['brown'],
        'marine.dinocyst': sns.xkcd_rgb['brown'],
        'marine.radiolaria': sns.xkcd_rgb['brown'],
        'marine.GDGT': sns.xkcd_rgb['brown'],
        'lake.varve_thickness': sns.xkcd_rgb['dark blue'],
        'lake.varve_property': sns.xkcd_rgb['dark blue'],
        'lake.accumulation': sns.xkcd_rgb['dark blue'],
        'lake.chironomid': sns.xkcd_rgb['dark blue'],
        'lake.midge': sns.xkcd_rgb['dark blue'],
        'lake.TEX86': sns.xkcd_rgb['dark blue'],
        'lake.BSi': sns.xkcd_rgb['dark blue'],
        'lake.chrysophyte': sns.xkcd_rgb['dark blue'],
        'lake.reflectance': sns.xkcd_rgb['dark blue'],
        'lake.pollen': sns.xkcd_rgb['dark blue'],
        'lake.alkenone': sns.xkcd_rgb['dark blue'],
        'borehole': sns.xkcd_rgb['peach'],
        'hybrid': sns.xkcd_rgb['maroon'],
        'documents': sns.xkcd_rgb['mauve'],
        'peat.pollen': sns.xkcd_rgb['brown'],
    }

    markers_dict = {
        'coral.calc': 'P',
        'coral.SrCa': 'X',
        'coral.d18O': 'o',
        'coral.d18Osw': 'H',
        'ice.melt': '<',
        'ice.d18O': 'd',
        'ice.dD': '>',
        'ice.d-excess': 'o',
        'ice.isotope_diffusion': 'h',
        'ice.hybrid': '8',
        'tree.TRW': '^',
        'tree.MXD': 'v',
        'tree.ENSO': '^',
        'pollen.temp': 'o',
        'land.temp': 'o',
        'ocean.temp': 'v',
        'tas': '^',
        'pr': 'o',
        'speleothem.d18O': 'o',
        'speleothem.dD': '>',
        'bivalve.d18O': 'o',
        'marine.TEX86': '*',
        'marine.MgCa': 'v',
        'marine.d18O': 'o',
        'marine.MAT': 'H',
        'marine.alkenone': 'h',
        'marine.foram': '8',
        'marine.diatom': '^',
        'marine.dinocyst': '>',
        'marine.radiolaria': '<',
        'marine.GDGT': 's',
        'lake.varve_thickness': 'H',
        'lake.varve_property': 's',
        'lake.accumulation': 'v',
        'lake.chironomid': 'D',
        'lake.midge': '>',
        'lake.TEX86': '*',
        'lake.BSi': '8',
        'lake.chrysophyte': 'd',
        'lake.reflectance': '<',
        'lake.pollen': '^',
        'lake.alkenone': 'h',
        'borehole': '8',
        'hybrid': 'P',
        'documents': 'X',
        'peat.pollen': 'o',
    }

class CartopySettings:
    projection_dict = {
        'Robinson': ccrs.Robinson,
        'NorthPolarStereo': ccrs.NorthPolarStereo,
        'SouthPolarStereo': ccrs.SouthPolarStereo,
        'PlateCarree': ccrs.PlateCarree,
        'AlbersEqualArea': ccrs.AlbersEqualArea,
        'AzimuthalEquidistant': ccrs.AzimuthalEquidistant,
        'EquidistantConic': ccrs.EquidistantConic,
        'LambertConformal': ccrs.LambertConformal,
        'LambertCylindrical': ccrs.LambertCylindrical,
        'Mercator': ccrs.Mercator,
        'Miller': ccrs.Miller,
        'Mollweide': ccrs.Mollweide,
        'Orthographic': ccrs.Orthographic,
        'Sinusoidal': ccrs.Sinusoidal,
        'Stereographic': ccrs.Stereographic,
        'TransverseMercator': ccrs.TransverseMercator,
        'UTM': ccrs.UTM,
        'InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine,
        'RotatedPole': ccrs.RotatedPole,
        'OSGB': ccrs.OSGB,
        'EuroPP': ccrs.EuroPP,
        'Geostationary': ccrs.Geostationary,
        'NearsidePerspective': ccrs.NearsidePerspective,
        'EckertI': ccrs.EckertI,
        'EckertII': ccrs.EckertII,
        'EckertIII': ccrs.EckertIII,
        'EckertIV': ccrs.EckertIV,
        'EckertV': ccrs.EckertV,
        'EckertVI': ccrs.EckertVI,
        'EqualEarth': ccrs.EqualEarth,
        'Gnomonic': ccrs.Gnomonic,
        'LambertAzimuthalEqualArea': ccrs.LambertAzimuthalEqualArea,
        'OSNI': ccrs.OSNI,
    }

def setlabel(ax, label, loc=2, borderpad=0.6, **kwargs):
    ''' Enumerate plots
    Reference: https://stackoverflow.com/questions/22508590/enumerate-plots-in-matplotlib-figure
    '''
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN, np.NaN,color='none',label=label)
    label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,**kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()

def plot_field_map(field_var, lat, lon, levels=50, add_cyclic_point=True,
                   title=None, title_size=20, title_weight='normal', figsize=[10, 8],
                   plot_proxydb=False, proxydb=None, proxydb_lgd_kws=None,
                   proxy_marker=None, proxy_color=None,
                   site_lats=None, site_lons=None, site_marker='o',
                   site_markersize=50, site_color=sns.xkcd_rgb['amber'],
                   projection='Robinson', transform=ccrs.PlateCarree(),
                   proj_args=None, latlon_range=None, central_longitude=180,
                   lon_ticks=[60, 120, 180, 240, 300], lat_ticks=[-90, -45, 0, 45, 90],
                   land_color=sns.xkcd_rgb['light grey'], ocean_color=sns.xkcd_rgb['light grey'],
                   land_zorder=None, ocean_zorder=None, signif_values=None, signif_range=[0.05, 9999], hatch='..',
                   clim=None, cmap=None, cmap_under=None, cmap_over=None, cmap_bad=None, extend=None, mode='latlon', add_gridlines=False,
                   plot_cbar=True, cbar_labels=None, cbar_pad=0.05, cbar_orientation='vertical', cbar_aspect=10,
                   cbar_fraction=0.15, cbar_shrink=0.5, cbar_title=None, cbar_title_x=0.5, cbar_title_y=1.05,
                   fig=None, ax=None):
    ''' Visualize a field on a map.

    Args:
        field_var (numpy.ndarray): field data array
        lat (numpy.array): list of latitudes
        lon (numpy.array): list of longitudes
        levels (int or list, optional): contour levels. Defaults to 50.
        add_cyclic_point (bool, optional): if True, add cyclic point to the field. Defaults to True.
        title (str, optional): title string. Defaults to None.
        title_size (int, optional): font size of the title string. Defaults to 20.
        title_weight (str, optional): font weight of the title string. Defaults to 'normal'.
        figsize (list, optional): figure size. Defaults to [10, 8].
        site_lats (list, optional): list of latitudes for the sites to be plotted on the map. Defaults to None.
        site_lons (list, optional): list of longitudes for the sites to be plotted on the map. Defaults to None.
        site_marker (list, optional): marker for the sites to be plotted on the map. Defaults to 'o'.
        site_markersize (float, optional): marker size for the sites to be plotted on the map. Defaults to 50.
        site_color (str, optional): color of the markers for the sites to be plotted on the map. Defaults to sns.xkcd_rgb['amber'].
        projection (str, optional): map projection. Defaults to 'Robinson'.
        transform (object, optional): map tranform. Defaults to ccrs.PlateCarree().
        proj_args (dict, optional): map projection arguments. Defaults to None.
        latlon_range (list, optional): for regional plotting; equals to [lat_min, lat_max, lon_min, lon_max]. Defaults to None.
        central_longitude (int, optional): central longitude for the map projection. Defaults to 180.
        lon_ticks (list, optional): _description_. Defaults to [60, 120, 180, 240, 300].
        lat_ticks (list, optional): _description_. Defaults to [-90, -45, 0, 45, 90].
        land_color (str, optional): _description_. Defaults to sns.xkcd_rgb['light grey'].
        ocean_color (str, optional): _description_. Defaults to sns.xkcd_rgb['light grey'].
        land_zorder (float, optional): _description_. Defaults to None.
        ocean_zorder (float, optional): _description_. Defaults to None.
        signif_values (float, optional): _description_. Defaults to None.
        signif_range (list, optional): _description_. Defaults to [0.05, 9999].
        hatch (str, optional): _description_. Defaults to '..'.
        clim (list, optional): _description_. Defaults to None.
        cmap (str, optional): colormap. Defaults to None.
        cmap_under (str, optional): color under the limit. Defaults to None.
        cmap_over (str, optional): color over the limit. Defaults to None.
        cmap_bad (str, optional): color for bad values. Defaults to None.
        extend (str, optional): if True, extend the colorbar. Defaults to None.
        mode (str, optional): _description_. Defaults to 'latlon'.
        add_gridlines (bool, optional): _description_. Defaults to False.
        plot_cbar (bool, optional): if True, plot the colorbar. Defaults to True.
        cbar_labels (_type_, optional): colorbar labels. Defaults to None.
        cbar_pad (float, optional): colorbar padding space. Defaults to 0.05.
        cbar_orientation (str, optional): colorbar orientation. Defaults to 'vertical'.
        cbar_aspect (int, optional): colorbar aspect. Defaults to 10.
        cbar_fraction (float, optional): colorbar fraction. Defaults to 0.15.
        cbar_shrink (float, optional): colorbar shrink. Defaults to 0.5.
        cbar_title (_type_, optional): colorbar title string. Defaults to None.
        cbar_title_x (float, optional): colorbar title location x. Defaults to 0.5.
        cbar_title_y (float, optional): colorbar title location y. Defaults to 1.05.
        fig (object, optional): `matplotlib.figure`. Defaults to None.
        ax (object, optional): `matplotlib.axes`. Defaults to None.

    '''

    if add_cyclic_point:
        if mode == 'latlon':
            field_var_c, lon_c = cutil.add_cyclic_point(field_var, lon)
            if signif_values is not None:
                signif_values_c, lon_c = cutil.add_cyclic_point(signif_values, lon)
            lat_c = lat
        elif mode == 'mesh':
            if len(np.shape(lat)) == 1:
                lon, lat = np.meshgrid(lon, lat, sparse=False, indexing='xy')
            if central_longitude == 180:
                lon = np.mod(lon+180, 360) - 180

            nx, ny = np.shape(field_var)

            lon_c = np.ndarray((nx, ny+1))
            lat_c = np.ndarray((nx, ny+1))
            field_var_c = np.ndarray((nx, ny+1))
            if signif_values is not None:
                signif_values_c = np.ndarray((nx, ny+1))

            lon_c[:, :-1] = lon
            lon_c[:, -1] = lon[:, 0]

            lat_c[:, :-1] = lat
            lat_c[:, -1] = lat[:, 0]

            field_var_c[:, :-1] = field_var
            field_var_c[:, -1] = field_var[:, 0]

            if signif_values is not None:
                signif_values_c[:, :-1] = signif_values
                signif_values_c[:, -1] = signif_values[:, 0]
    else:
        field_var_c, lat_c, lon_c = field_var, lat, lon
        if signif_values is not None:
            signif_values_c = signif_values

    if ax is None or fig is None:
        fig = plt.figure(figsize=figsize)

        proj_args = {} if proj_args is None else proj_args
        proj_args_default = {'central_longitude': central_longitude}
        proj_args_default.update(proj_args)
        projection = CartopySettings.projection_dict[projection](**proj_args_default)
        ax = plt.subplot(projection=projection)

    if title:
        plt.title(title, fontsize=title_size, fontweight=title_weight)

    if latlon_range:
        lat_min, lat_max, lon_min, lon_max = latlon_range
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=transform)
        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        lon_ticks = np.array(lon_ticks)
        lat_ticks = np.array(lat_ticks)
        mask_lon = (lon_ticks >= lon_min) & (lon_ticks <= lon_max)
        mask_lat = (lat_ticks >= lat_min) & (lat_ticks <= lat_max)
        ax.set_xticks(lon_ticks[mask_lon], crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks[mask_lat], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor=land_color, zorder=land_zorder)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor=ocean_color, zorder=ocean_zorder)
    ax.coastlines(zorder=99)

    if add_gridlines:
        ax.gridlines(edgecolor='gray', linestyle=':', crs=transform)

    cmap = plt.get_cmap(cmap)
    if cmap_under is not None:
        cmap.set_under(cmap_under)
    if cmap_over is not None:
        cmap.set_over(cmap_over)
    if cmap_bad is not None:
        cmap.set_bad(cmap_bad)

    if mode == 'latlon':
        im = ax.contourf(lon_c, lat_c, field_var_c, levels, transform=transform, cmap=cmap, extend=extend)

    elif mode == 'mesh':
        if type(levels) is int:
            levels = MaxNLocator(nbins=levels).tick_values(np.nanmax(field_var_c), np.nanmin(field_var_c))
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        im = ax.pcolormesh(lon_c, lat_c, field_var_c, transform=transform, cmap=cmap, norm=norm)

    if clim:
        im.set_clim(clim)

    if signif_values is not None:
        ax.contourf(lon_c, lat_c, signif_values_c, signif_range, transform=transform, hatches=[hatch], colors='none')

    if plot_cbar:
        cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
                        fraction=cbar_fraction, shrink=cbar_shrink)

        if cbar_labels is not None:
            cbar.set_ticks(cbar_labels)

        if cbar_title:
            cbar.ax.set_title(cbar_title, x=cbar_title_x, y=cbar_title_y)

    if site_lats is not None and site_lons is not None:
        if type(site_lats) is not dict:
            ax.scatter(site_lons, site_lats, s=site_markersize, c=site_color, marker=site_marker, edgecolors='k',
                       zorder=99, transform=transform, cmap=cmap)
        else:
            for name in site_lats.keys():
                ax.scatter(site_lons[name], site_lats[name], s=site_markersize[name], c=site_color[name],
                           marker=site_marker[name], edgecolors='k', zorder=99, transform=transform, cmap=cmap)

    if plot_proxydb:
        proxydb_lgd_kws = {} if proxydb_lgd_kws is None else proxydb_lgd_kws
        ptypes = []
        ptype_labels = []
        for k, v in proxydb.type_dict.items():
            ptypes.append(k)
            ptype_labels.append(f'{k} (n={v})')

        site_lats, site_lons = {}, {}
        for ptype in ptypes:
            site_lats[ptype] = []
            site_lons[ptype] = []

        for _, pobj in proxydb.records.items():
            site_lats[pobj.ptype].append(pobj.lat)
            site_lons[pobj.ptype].append(pobj.lon)

        s_plots = []
        for ptype in ptypes:
            if ptype in STYLE.markers_dict and proxy_marker is None:
                marker = STYLE.markers_dict[ptype]
            else:
                marker = proxy_marker

            if ptype in STYLE.colors_dict and proxy_color is None:
                color = STYLE.colors_dict[ptype]
            else:
                color = proxy_color

            s_plots.append(
                ax.scatter(
                    site_lons[ptype], site_lats[ptype], marker=marker,
                    c=color, edgecolor='k', s=site_markersize, transform=ccrs.PlateCarree()
                )
            )

        ax.legend(
            s_plots, ptype_labels,
            **proxydb_lgd_kws,
        )
        

    return fig, ax

def plot_proxies(df, year=np.arange(2001), lon_col='lon', lat_col='lat', type_col='type', time_col='time',
                 title=None, title_weight='normal', markers_dict=None, colors_dict=None,
                 plot_timespan=None,  plot_xticks=[850, 1000, 1200, 1400, 1600, 1800, 2000],
                 figsize=[10, 10], projection='Robinson', proj_args=None, central_longitude=180, markersize=50,
                 plot_count=False, nrow=2, ncol=1, wspace=0.5, hspace=0.3, return_gs=False,
                 lgd_ncol=None, lgd_anchor_upper=(1, 0), lgd_anchor_lower=(1, -0.05),lgd_frameon=False,
                 enumerate_ax=False, enumerate_prop={'weight': 'bold', 'size': 30}, p=STYLE, stock_img=True,
                 enumerate_anchor_map=[0, 1], enumerate_anchor_count=[0, 1], map_grid_idx=0, count_grid_idx=1):
    ''' Visualize proxies.

    Args:
        df (pandas.DataFrame): proxy database in `pandas.DataFrame`.
        year (list): the years for counting proxy numbers.
        p (object): the style object that contains dictionaries of markers and colors

    '''

    fig = plt.figure(figsize=figsize)

    if not plot_count:
        nrow = 1
        ncol = 1

    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=wspace, hspace=hspace)

    proj_args = {} if proj_args is None else proj_args
    proj_args_default = {'central_longitude': central_longitude}
    proj_args_default.update(proj_args)
    projection = CartopySettings.projection_dict[projection](**proj_args_default)

    ax = {}
    ax['map'] = plt.subplot(gs[map_grid_idx], projection=projection)
    if stock_img:
        ax['map'].stock_img()

    if title:
        ax['map'].set_title(title, fontweight=title_weight)

    ax['map'].set_global()
    ax['map'].add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

    # plot markers by archive types
    if markers_dict is None:
        markers_dict = p.markers_dict
    if colors_dict is None:
        colors_dict = p.colors_dict

    s_plots = []
    type_names = []
    type_set = np.unique(df[type_col])
    max_count = []
    for ptype in type_set[::-1]:
        selector = df[type_col] == ptype
        max_count.append(len(df[selector]))
        type_names.append(f'{ptype} (n={max_count[-1]})')
        lons = list(df[selector][lon_col])
        lats = list(df[selector][lat_col])
        s_plots.append(
            ax['map'].scatter(
                lons, lats, marker=markers_dict[ptype],
                c=colors_dict[ptype], edgecolor='k', s=markersize, transform=ccrs.PlateCarree()
            )
        )

    if lgd_ncol is None:
        lgd_ncol = len(type_set) // 20 + 1

    ax['map'].legend(
        s_plots, type_names,
        scatterpoints=1,
        bbox_to_anchor=lgd_anchor_upper,
        loc='lower left',
        ncol=lgd_ncol,
        frameon=lgd_frameon,
    )

    if plot_count:
        ax['count'] = plt.subplot(gs[count_grid_idx])
        df_count = {}
        for ptype in type_set[::-1]:
            df_count[ptype] = pd.DataFrame(index=year)

        for index, row in df.iterrows():
            ptype = row[type_col]
            time = np.array(row[time_col]).astype(int)
            time = time[~np.isnan(time)]
            time = time[time<np.max(year)+1]
            time = np.sort(list(set(time)))  # remove the duplicates for monthly data
            ts = pd.Series(index=time, data=1, name=row['pid'])
            df_count[ptype] = pd.concat([df_count[ptype], ts], axis=1)

        proxy_count = {}
        for ptype in df_count.keys():
            proxy_count[ptype] = df_count[ptype].sum(axis=1)

        cumu_count = np.zeros_like(year, dtype=int)
        cumu_last = np.copy(cumu_count)
        for ptype in type_set:
            cumu_count += proxy_count[ptype]
            ax['count'].fill_between(
                year, cumu_last, cumu_count,
                facecolor=colors_dict[ptype],
                label=f'{ptype}',
                alpha=0.8,
            )
            cumu_last = np.copy(cumu_count)

        ax['count'].set_xlabel('Year (AD)')
        ax['count'].set_ylabel('number of proxies')
        if plot_timespan is not None:
            ax['count'].set_xlim(plot_timespan)
            ax['count'].set_xticks(plot_xticks)
        handles, labels = ax['count'].get_legend_handles_labels()
        ax['count'].legend(handles[::-1], labels[::-1], frameon=lgd_frameon, ncol=lgd_ncol, bbox_to_anchor=lgd_anchor_lower, loc='lower left')

        if enumerate_ax:
            setlabel(ax['map'], '(a)', prop=enumerate_prop, bbox_to_anchor=enumerate_anchor_map)
            setlabel(ax['count'], '(b)', prop=enumerate_prop, bbox_to_anchor=enumerate_anchor_count)

    if not return_gs:
        return fig, ax
    else:
        return fig, ax, gs

def plot_proxy_age_map(df, lon_col='lon', lat_col='lat', type_col='type', time_col='time',
                       title=None, title_weight='normal', font_scale=1.5,
                       figsize=[12, 10], projection=ccrs.Robinson, central_longitude=0, markersize=150,
                       plot_cbar=True, marker_color=None, transform=ccrs.PlateCarree(), p=STYLE,
                       add_nino34_box=False, add_nino12_box=False, add_box=False, add_box_lf=None, add_box_ur=None):

    fig = plt.figure(figsize=figsize)

    projection = projection(central_longitude=central_longitude)
    ax_map = plt.subplot(projection=projection)

    if title:
        ax_map.set_title(title, fontweight=title_weight)

    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

    if add_nino12_box:
        x, y = [-90, -90, -80, -80, -90], [0, -10, -10, 0, 0]
        ax_map.plot(x, y, '--', transform=transform, color='gray')

    if add_nino34_box:
        x, y = [-170, -170, -120, -120, -170], [5, -5, -5, 5, 5]
        ax_map.plot(x, y, '--', transform=transform, color='gray')

    if add_box:
        lf_lat, lf_lon = add_box_lf
        ur_lat, ur_lon = add_box_ur
        x, y = [lf_lon, lf_lon, ur_lon, ur_lon, lf_lon], [ur_lat, lf_lat, lf_lat, ur_lat, ur_lat]
        ax_map.plot(x, y, '--', transform=transform, color='gray')

    color_norm = Normalize(vmin=0, vmax=1000)

    cmap = cm.get_cmap('viridis_r', 10)
    cmap.set_under(sns.xkcd_rgb['cream'])
    cmap.set_over('black')

    # plot markers by archive types
    s_plots = []
    type_names = []
    type_set = np.unique(df[type_col])
    max_count = []
    for ptype in type_set:
        selector = df[type_col] == ptype
        max_count.append(len(df[selector]))
        type_names.append(f'{ptype} (n={max_count[-1]})')
        lons = list(df[selector][lon_col])
        lats = list(df[selector][lat_col])
        ages = []
        for idx, row in df[selector].iterrows():
            ages.append(1950-np.min(row['time']))

        if marker_color is None:
            s_plots.append(
                ax_map.scatter(
                    lons, lats, marker=p.markers_dict[ptype], cmap=cmap, norm=color_norm,
                    c=ages, edgecolor='k', s=markersize, transform=ccrs.PlateCarree()
                )
            )
        else:
            s_plots.append(
                ax_map.scatter(
                    lons, lats, marker=p.markers_dict[ptype], cmap=cmap, norm=color_norm,
                    c=marker_color, edgecolor='k', s=markersize, transform=ccrs.PlateCarree()
                )
            )

    if plot_cbar:
        cbar_lm = plt.colorbar(s_plots[0], orientation='vertical',
                               pad=0.05, aspect=10, extend='min',
                               ax=ax_map, fraction=0.05, shrink=0.5)

        cbar_lm.ax.set_title(r'age [yrs]', y=1.05)
        cbar_lm.set_ticks([0, 200, 400, 600, 800, 1000])

    return fig

def in_notebook():
    ''' Check if the code is executed in a Jupyter notebook
    
    Returns
    -------
    
    bool
    
    '''
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


def showfig(fig, close=True):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    close : bool
        if True, close the figure automatically

    '''
    # if in_notebook:
    #     try:
    #         from IPython.display import display
    #     except ImportError as error:
    #         # Output expected ImportErrors.
    #         print(f'{error.__class__.__name__}: {error.message}')

    #     display(fig)

    # else:
    #     plt.show()

    plt.show()

    if close:
        closefig(fig)

def closefig(fig=None):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    '''
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()

def savefig(fig, path=None, settings={}, verbose=True):
    ''' Save a figure to a path

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        the figure to save
    path : str
        the path to save the figure, can be ignored and specify in "settings" instead
    settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified in settings if not assigned with the keyword argument;
          it can be any existed or non-existed path, with or without a suffix;
          if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
        
    '''
    if path is None and 'path' not in settings:
        raise ValueError('"path" must be specified, either with the keyword argument or be specified in `settings`!')

    savefig_args = {'bbox_inches': 'tight', 'path': path}
    savefig_args.update(settings)

    path = pathlib.Path(savefig_args['path'])
    savefig_args.pop('path')

    dirpath = path.parent
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f'Directory created at: "{dirpath}"')

    path_str = str(path)
    if path.suffix not in ['.eps', '.pdf', '.png', '.ps']:
        path = pathlib.Path(f'{path_str}.pdf')

    fig.savefig(path_str, **savefig_args)
    plt.close(fig)

    if verbose:
        print(f'Figure saved at: "{str(path)}"')


def set_style(style='journal', font_scale=1.0):
    ''' Modify the visualization style
    
    This function is inspired by [Seaborn](https://github.com/mwaskom/seaborn).
    See a demo in the example_notebooks folder on GitHub to look at the different styles
    
    Parameters
    ----------
    
    style : {journal,web,matplotlib,_spines, _nospines,_grid,_nogrid}
        set the styles for the figure:
            - journal (default): fonts appropriate for paper
            - web: web-like font (e.g. ggplot)
            - matplotlib: the original matplotlib style
            In addition, the following options are available:
            - _spines/_nospines: allow to show/hide spines
            - _grid/_nogrid: allow to show gridlines (default: _grid)
    
    font_scale : float
        Default is 1. Corresponding to 12 Font Size. 
    
    '''
    font_dict = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    }

    style_dict = {}
    if 'journal' in style:
        style_dict.update({
            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.grid': True,
            'grid.color': 'lightgrey',
            'grid.linestyle': '--',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })
    elif 'web' in style:
        style_dict.update({
            'figure.facecolor': 'white',

            'axes.axisbelow': True,
            'axes.facecolor': 'whitesmoke',
            'axes.edgecolor': 'lightgrey',
            'axes.grid': True,
            'grid.color': 'white',
            'grid.linestyle': '-',
            'xtick.direction': 'out',
            'ytick.direction': 'out',

            'text.color': 'grey',
            'axes.labelcolor': 'grey',
            'xtick.color': 'grey',
            'ytick.color': 'grey',

            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })
    else:
        raise ValueError(f'Style [{style}] not availabel!')

    if '_spines' in style:
        style_dict.update({
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True,
        })
    elif '_nospines' in style:
        style_dict.update({
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,
        })

    if '_grid' in style:
        style_dict.update({
            'axes.grid': True,
        })
    elif '_nogrid' in style:
        style_dict.update({
            'axes.grid': False,
        })

    # modify font size based on font scale
    font_dict.update({k: v * font_scale for k, v in font_dict.items()})

    for d in [style_dict, font_dict]:
        mpl.rcParams.update(d)


def plot_sea_res(res, style='ticks', font_scale=2, figsize=[6, 6],
                 ls='-o', lw=3, color='k', label=None, label_shade=None, alpha=1, shade_alpha=0.3,
                 ylim=None, xlim=None, plot_mode='composite_qs', lgd_individual_yrs=False,
                 signif_alpha=0.5, signif_color=sns.xkcd_rgb['grey'], signif_text_loc_fix=(0.1, -0.01),
                 signif_fontsize=10, signif_lw=1, indi_style='o', indi_alpha=0.5,
                 xlabel='Years relative to event year', ylabel='T anom. (K)', plot_lgd=False,
                 xticks=None, yticks=None, title=None, plot_signif=True, ax=None):
    ''' Plot SEA results
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plot_mode in res.keys():
        if plot_mode == 'composite_qs':
            ax.plot(res['composite_yr'], res['composite_qs'][1], ls, color=color, label=label, lw=lw, alpha=alpha)
            ax.fill_between(res['composite_yr'], res['composite_qs'][0], res['composite_qs'][-1], facecolor=color, alpha=shade_alpha, label=label_shade)
        elif plot_mode == 'composite':
            ax.plot(res['composite_yr'], res['composite'], ls, color=color, label=label, lw=lw, alpha=alpha)
        elif plot_mode == 'composite_norm':
            ax.plot(res['composite_yr'], res['composite_qs'][1], ls, color=color, label=label, lw=lw, alpha=1)
            for i, individual_curve in enumerate(res['composite_norm'][0, :, :, 0]):
                if lgd_individual_yrs:
                    lb = res['events'][i]
                    clr = None
                else:
                    lb = 'individual events' if i==0 else None
                    clr = color

                ax.plot(res['composite_yr'], individual_curve, indi_style, label=lb, lw=1, alpha=indi_alpha, color=clr)
    else:
        raise KeyError('Wrong plot_mode!')

    if 'qs_signif' not in res.keys():
        plot_signif = False

    if plot_signif:
        for i, qs_v in enumerate(res['qs_signif']):
            ax.plot(res['composite_yr'], res['composite_qs_signif'][i], '-.', color=signif_color, alpha=signif_alpha, lw=signif_lw)
            ax.text(res['composite_yr'][-1]+signif_text_loc_fix[0], res['composite_qs_signif'][i][-1]+signif_text_loc_fix[-1],
                    f'{qs_v*100:g}%', color=signif_color, alpha=signif_alpha, fontsize=signif_fontsize)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.axvline(x=0, ls=':', color='grey')
    ax.axhline(y=0, ls=':', color='grey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if title is not None:
        ax.set_title(title)

    if plot_lgd:
        ax.legend(frameon=False, loc='upper left')

    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def plot_volc_pdf(year_volc, anom_volc, anom_nonvolc, xs,
                  clr_volc=sns.xkcd_rgb['black'], clr_volc_signif=sns.xkcd_rgb['pale red'],
                  clr_nonvolc=sns.xkcd_rgb['grey'], clr_nonvolc_light=sns.xkcd_rgb['light grey'],
                  signif_qs=[0.8, 0.9, 0.95], signif_markers=['v', '^', 'd'], insignif_marker='o',
                  figsize=[8, 3], ax=None, plot_lgd=True, lgd_style=None, lgd_fs=10, lgd_ms=6,
                  ms_large=30, ms_small=15, qs_fs=15, yr_fs=None,
                  xlabel=None, ylabel=None, label_style=None, title=None, title_style=None,
                  xlim=None, ylim=None,
                  xticks=None, yticks=None,
                  signif_ratio_loc_x=0.02, signif_ratio_loc_y=0.95,
                  clr_style='signif', cmap_name='viridis_r', nclrs=1000,
                  clr_yr_range=[1000, 1999], clr_yr_step=100,
                 ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        n_volc = np.shape(anom_volc)[0]
        n_nonvolc = np.shape(anom_nonvolc)[0]

        kde_nonvolc = gaussian_kde(anom_nonvolc)
        ax.plot(xs, kde_nonvolc(xs), color=clr_nonvolc, label=f'KDE of non-volcanic years (n={n_nonvolc})')
        ax.fill_between(xs, 0, kde_nonvolc(xs), color=clr_nonvolc_light)

        # quantiles of nonvolc
        anom_nonvolc_qs = np.quantile(anom_nonvolc, signif_qs)
        for q, anom_q in zip(signif_qs, anom_nonvolc_qs):
            idx = np.argmin(np.abs(xs-anom_q))
            ax.vlines(xs[idx], 0, kde_nonvolc(xs)[idx], linestyle='-.', zorder=98, color=clr_nonvolc)
            ax.text(xs[idx], kde_nonvolc(xs)[idx], f'{q*100:g}%', zorder=101, color=clr_nonvolc, fontsize=qs_fs,
                    horizontalalignment='left', verticalalignment='bottom')

        anom_median = np.quantile(anom_nonvolc, [0.5])[0]
        idx = np.argmin(np.abs(xs-anom_median))
        ax.vlines(xs[idx], 0, kde_nonvolc(xs)[idx], linestyle='-.', zorder=98, color=clr_nonvolc)
        ax.text(xs[idx], kde_nonvolc(xs)[idx], '50%', zorder=101, color=clr_nonvolc, fontsize=qs_fs,
                horizontalalignment='left', verticalalignment='bottom')

        anom_volc_sorted = sorted(anom_volc)
        sort_idx = np.argsort(anom_volc)
        year_sorted = np.array(year_volc)[sort_idx]

        clr_list = []
        marker_list = []
        ms_list = []

        n_qs = np.size(signif_qs)
        n_signif = np.zeros(n_qs)

        for k, anom_v in enumerate(anom_volc_sorted):
            if clr_style == 'signif':
                if anom_v >= anom_nonvolc_qs[0]:
                    clr_list.append(clr_volc_signif)
                else:
                    clr_list.append(clr_volc)
            elif clr_style == 'time':
                # color the volcanic years according to time
                year = year_sorted[k]
                sns_cmap = sns.color_palette(palette=cmap_name, n_colors=nclrs)
                clr_ind = int((year-clr_yr_range[0])/(clr_yr_range[-1]-clr_yr_range[0])*nclrs)
                clr_list.append(sns_cmap[clr_ind])

            else:
                raise ValueError('Wrong `clr_style`: please choose between {"signif", "time"}.')

            loc_found = False
            # insignificant
            if (anom_v < anom_nonvolc_qs[0]) & (not loc_found):
                marker_list.append(insignif_marker)
                ms_list.append(ms_small)
                loc_found = True

            for i in range(n_qs-1):
                if (anom_v >= anom_nonvolc_qs[i]) & (anom_v < anom_nonvolc_qs[i+1]) & (not loc_found):
                    marker_list.append(signif_markers[i])
                    n_signif[i] += 1
                    ms_list.append(ms_large)
                    loc_found = True

            if (anom_v >= anom_nonvolc_qs[-1]) & (not loc_found):
                marker_list.append(signif_markers[-1])
                n_signif[-1] += 1
                ms_list.append(ms_large)
                loc_found = True

        i = 0
        kde_max = np.max(kde_nonvolc(xs))
        for yr, v in zip(year_sorted, anom_volc_sorted):
            lb = f'Volcanic events (n={n_volc})' if i==0 else None
            ax.vlines(v, 0, kde_max/n_volc*(i+1), color=clr_list[i], linestyle='-', zorder=99, label=lb, lw=1)
            ax.scatter(v, y=kde_max/n_volc*(i+1), color=clr_list[i], marker=marker_list[i], s=ms_list[i], zorder=100)
            ax.text(v, kde_max/n_volc*(i+1)*1.01, yr, color=clr_list[i], horizontalalignment='right', fontsize=yr_fs)
            i += 1

        n_signif_cum = np.copy(n_signif)
        for i in range(np.size(n_signif_cum)):
            n_signif_cum[i] = np.sum(n_signif_cum[i:])

        signif_ratio_str = ','.join([str(int(n)) for n in n_signif_cum])
        ax.text(signif_ratio_loc_x, signif_ratio_loc_y, f'Signif. ratio: ({signif_ratio_str})/{n_volc}', transform=ax.transAxes)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if xticks is not None:
            ax.set_xticks(xticks)

        if yticks is not None:
            ax.set_xticks(yticks)

        if xlabel is not None:    
            label_style = {} if label_style is None else label_style.copy()
            ax.set_xlabel(xlabel, **label_style)

        if ylabel is not None:    
            label_style = {} if label_style is None else label_style.copy()
            ax.set_ylabel(ylabel, **label_style)

        if title is not None:    
            title_kwargs = {'color': clr_volc_signif}
            title_style = {} if title_style is None else title_style.copy()
            ax.set_title(title, **title_kwargs)

        # legend
        if plot_lgd:
            lgd_kwargs = {'frameon': False, 'loc': 'lower left', 'bbox_to_anchor': (0, -1), 'fontsize': lgd_fs, 'ncol': 1, 'handletextpad': 0.1}
            lgd_style = {} if lgd_style is None else lgd_style.copy()
            lgd_kwargs.update(lgd_style)

            legend_elements = [
                Line2D([], [], marker=insignif_marker, markersize=lgd_ms, color=clr_volc,
                label=f'Volcanic events (< {signif_qs[0]*100:g}% non-volcanic years)', linestyle='None'),
            ]

            for i, q in enumerate(signif_qs):
                if i < np.size(signif_qs)-1:
                    legend_elements.append(
                        Line2D([], [], marker=signif_markers[i], markersize=lgd_ms, color=clr_volc,
                        label=f'Volcanic events (between {signif_qs[i]*100:g}-{signif_qs[i+1]*100:g}% non-volcanic years)', linestyle='None'),
                    )
                else:
                    legend_elements.append(
                        Line2D([], [], marker=signif_markers[i], markersize=lgd_ms, color=clr_volc,
                        label=f'Volcanic events (>{signif_qs[i]*100:g}% non-volcanic years)', linestyle='None'),
                    )

            legend_elements.append(
                Patch(facecolor=clr_nonvolc_light, edgecolor=clr_nonvolc, label=f'  Distribution of non-volcanic years')
            )

            ax.legend(handles=legend_elements, **lgd_kwargs)

        # colorbar
        if clr_style == 'time':
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cmap_obj = ListedColormap(sns_cmap.as_hex())
            clr_norm = Normalize(vmin=clr_yr_range[0], vmax=clr_yr_range[1]+1)
            cb = mpl.colorbar.ColorbarBase(
                cax, cmap=cmap_obj, orientation='vertical', norm=clr_norm,
            )
            cb.set_label('Year (CE)')


        if 'fig' in locals():
            return fig, ax
        else:
            return ax


def plot_volc_cdf(year_volc, anom_volc, anom_nonvolc, anom_nonvolc_draws, value_range,
                  nbin=2000, qs=[0.05, 0.5, 0.95], band_idx=[0, -1], line_idx=1, figsize=[5, 5], xlabel=None, ylabel='Cumulative Distribution Function',
                  lw_nonvolc_qs=1, lw_volc_nonvolc=1, lw_nonvolc_med=1, xlim=None, title=None, show_ratio_in_title=True,
                  clr_volc_signif=sns.xkcd_rgb['pale red'], clr_volc=sns.xkcd_rgb['black'],
                  clr_nonvolc=sns.xkcd_rgb['grey'], clr_nonvolc_qs=sns.xkcd_rgb['light grey'],
                  fs=15, ms=100, yr_base=2001, yr_label_x_adj=None, yr_label_y_adj=0, yr_lb_shift_ratio=1.8,
                  label_volc='Volcanic years', label_nonvolc='Non-volcanic years', plot_nonvolc=False,
                  label_nonvolc_qs='Randomly selected\nnon-volcanic years', plot_lgd=True, ax=None, lgd_style=None):

        kws = {'cumulative': True, 'density': True, 'histtype': 'step', 'range': value_range, 'bins': nbin, 'lw': lw_volc_nonvolc}
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        n, bins, patches = {}, {}, {}
        n['volc'], bins['volc'], patches['volc'] = ax.hist(anom_volc, label=label_volc, color=clr_volc, **kws, zorder=99)

        cdf_draw = []
        for anom_draw in anom_nonvolc_draws:
            cdf_anom_draw = cumfreq(anom_draw, numbins=nbin, defaultreallimits=value_range)
            cdf = cdf_anom_draw.cumcount / np.shape(anom_nonvolc_draws)[-1]
            cdf_draw.append(cdf)

        cdf_qs = utils.calc_cdf_qs(cdf_draw, qs)
        cdf_lb = utils.recover_cdf_from_locs(cdf_qs[qs[band_idx[0]]], nbin)
        cdf_ub = utils.recover_cdf_from_locs(cdf_qs[qs[band_idx[1]]], nbin)

        if plot_nonvolc:
            n['nonvolc'], bins['nonvolc'], patches['nonvolc'] = ax.hist(anom_nonvolc, label=label_nonvolc, color=clr_nonvolc, zorder=98, **kws)
        else:
            cdf_line = utils.recover_cdf_from_locs(cdf_qs[qs[line_idx]], nbin)
            lb_line = f'{label_nonvolc_qs} ({qs[line_idx]*100:g}%)'
            ax.fill_between(np.linspace(value_range[0], value_range[-1], nbin), cdf_line, cdf_line, color=clr_nonvolc, label=lb_line, lw=lw_nonvolc_med, zorder=98)

        x_values = np.linspace(value_range[0], value_range[1], nbin)
        ub_loc_dict = {}
        for k, v in cdf_qs[qs[-1]].items():
            ub_loc_dict[f'{k:.2f}'] = x_values[int(v)]

        for k in n.keys():
            patches[k][0].set_xy(patches[k][0].get_xy()[:-1])

        nsig = 0
        if yr_label_x_adj is None:
            yr_label_x_adj = -np.abs(value_range[0])/10

        anom_volc_list = []
        for i, yr in enumerate(year_volc):
            for j, b in enumerate(bins['volc']):
                if anom_volc[i] >= b and anom_volc[i] < bins['volc'][j+1]:
                    n_tmp = n['volc'][j]

            loc = f'{n_tmp:.2f}'
            if anom_volc[i] <= ub_loc_dict[loc]:
                signif = False
            else:
                signif = True
                nsig += 1

            clr = clr_volc_signif if signif else clr_volc
            ax.scatter(anom_volc[i], n_tmp, marker='^', color=clr, zorder=100, s=ms)
            if i>0 and f'{anom_volc[i]:.4f}' in anom_volc_list:
                # text for the same ranking
                ax.text(anom_volc[i]+yr_label_x_adj, n_tmp-1/len(anom_volc)/yr_lb_shift_ratio, yr%yr_base, color=clr, zorder=100, fontsize=fs)
            else:
                ax.text(anom_volc[i]+yr_label_x_adj, n_tmp+yr_label_y_adj, yr%yr_base, color=clr, zorder=100, fontsize=fs)
            anom_volc_list.append(f'{anom_volc[i]:.4f}')

        ax.scatter(None, None, marker='^', color=clr_volc, label='Insignificant events')
        ax.scatter(None, None, marker='^', color=clr_volc_signif, label=f'Significant events (> {qs[band_idx[1]]*100:g}%)')

        lb_qs = f'{label_nonvolc_qs} ({qs[band_idx[0]]*100:g}%-{qs[band_idx[1]]*100:g}%)'
        ax.fill_between(np.linspace(value_range[0], value_range[-1], nbin), cdf_lb, cdf_ub, color=clr_nonvolc_qs, label=lb_qs, lw=lw_nonvolc_qs)

        ax.set_ylim(0, 1.05)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(value_range[0], value_range[1])

        ax.set_ylabel(ylabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if plot_lgd:
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 1, 4, 2, 3]

            handles = [handles[idx] for idx in order]
            labels = [labels[idx] for idx in order]

            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles[0:2]]
            handles[0:2] = new_handles

            lgd_kwargs = {'loc': 'lower right', 'bbox_to_anchor': (2., 0), 'fontsize': 15}
            lgd_style = {} if lgd_style is None else lgd_style.copy()
            lgd_kwargs.update(lgd_style)

            ax.legend(handles, labels, **lgd_kwargs)

        if title is not None:
            ax.set_title(f'{title}', y=1.05)

        if show_ratio_in_title:
            nevents = np.size(year_volc)
            ratio_str = f'{nsig}/{nevents}'
            ax.text(0.02, 0.9, f'Signif. ratio: {ratio_str}', transform=ax.transAxes)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax

def make_lb(name, unit, wrap=False):
    if unit is not None:
        if wrap:
            lb = f'{name}\n[{unit}]'
        else:
            lb = f'{name} [{unit}]'
    else:
        lb = f'{name}'
    return lb

def plot_scatter_map(values, lats, lons, levels=None,
    figsize=(12, 6), projection='Robinson', transform=ccrs.PlateCarree(),
    ax=None, proj_args=None, central_longitude=0, stock_img=True, marker='o', ms=200, edge_clr='w', title=None, title_fs=20,
    cmap='Reds', vmin=None, vmax=None, clim=None, cmap_under=None, cmap_over=None, cbar=True, cbar_ticks=None, cbar_labels=None, cbar_orientation='vertical',
    cbar_pad=0.05, cbar_extend='neither', cbar_fraction=0.15, cbar_shrink=0.5,
    cbar_title=None, cbar_title_x=0.5, cbar_title_y=1.05, cbar_aspect=10,
    gridlines=True, gl_labels=False, gl_top_lbs=False, gl_right_lbs=False):

    proj_args = {} if proj_args is None else proj_args
    proj_args_default = {'central_longitude': central_longitude}
    proj_args_default.update(proj_args)
    projection = CartopySettings.projection_dict[projection](**proj_args_default)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=projection)

    if stock_img:
        ax.stock_img()

    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=title_fs)

    cmap = plt.get_cmap(cmap)
    if cmap_under is not None:
        cmap.set_under(cmap_under)
    if cmap_over is not None:
        cmap.set_over(cmap_over)

    if levels is not None:
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    else:
        norm = None

    im = ax.scatter(lons, lats, s=ms, c=values, zorder=99, norm=norm,
        edgecolor=edge_clr, marker=marker, cmap=cmap, vmin=vmin, vmax=vmax, transform=transform)

    if clim is not None:
        im.set_clim(clim)

    if cbar:
        cbar = plt.colorbar(im, ax=ax,
            orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=cbar_extend,
            fraction=cbar_fraction, shrink=cbar_shrink)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        if cbar_labels is not None:
            cbar.set_ticklabels(cbar_labels)
        if cbar_title:
            cbar.ax.set_title(cbar_title, x=cbar_title_x, y=cbar_title_y)

    if gridlines:
        gl = ax.gridlines(draw_labels=gl_labels)
        gl.top_labels = gl_top_lbs
        gl.right_labels = gl_right_lbs
        

    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def plot_ml_loss(train_loss, valid_loss, optim_epoch=None, train_label='train_loss', valid_label='valid_loss',
                 figsize=[8, 4], xlabel='Epoch', ylabel='Loss', xlim=None, ylim=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = np.arange(1, np.size(train_loss)+1)
    ax.plot(epochs, train_loss, label=train_label)
    ax.plot(epochs, valid_loss, label=valid_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if optim_epoch is not None:
        ax.axvline(x=optim_epoch, color='tab:grey', ls='--', label=f'epoch: {optim_epoch}')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def plot_ml_predict(res_dict, figsize=[8, 4], xlabel='Time', ylabel='Value', xlim=None, ylim=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    pred = res_dict['pred']
    truth = res_dict['truth']
    ax.plot(pred, label='prediction')
    ax.plot(truth, label='truth')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'r={res_dict["corr"]:.2f}, CE={res_dict["CE"]:.2f}')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def plot_eof(eof, pc, lat, lon, time, eof_title='EOF', pc_title='PC'):
    fig = plt.figure(figsize=[12, 10])
    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.1, hspace=0.2)

    ax = {}
    ax['mode'] = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
    ax['mode'].set_global()
    ax['mode'].add_feature(cfeature.LAND, color='lightgrey')
    ax['mode'].add_feature(cfeature.OCEAN, color='white')
    ax['mode'].coastlines()

    CS = ax['mode'].contourf(
        lon, lat, eof,
        levels=np.arange(-1, 1.1, 0.1),
        vmin=-1, vmax=1,
        cmap='RdBu_r',
        transform=ccrs.PlateCarree(),
        extend='neither',
    )
    ax['mode'].set_title(eof_title)
    cbar = fig.colorbar(CS, shrink=1.)
    # cbar.ax.set_title(r'[$^{\circ}$C]', y=1.05)
    cbar.ax.set_yticks(np.arange(-1, 1.1, 0.2))

    ax['pc'] = fig.add_subplot(gs[1, 0])
    ax['pc'].plot(time, pc)
    ax['pc'].set_title(pc_title)
    ax['pc'].set_xlabel('Year')
    ax['pc'].set_ylabel(pc_title)
    ax['pc'].grid(ls='--')
    ax['pc'].spines.top.set_visible(False)
    ax['pc'].spines.right.set_visible(False)

    return fig, ax