from .GraphEstimation import dMat
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


class Graph:
    ''' The object for a graph '''

    def __init__(self, lonlat, temp, proxy):
        '''
        Args:
            lonlat (numpy.array): the location array in shape of (num_gridpoint + num_proxy, 2), where the 2nd dimension is aranged as [longitude, latitude]
            temp (numpy.array): the temperature array in shape of (num_time, num_gridpoint)
            proxy (numpy.array): the proxy array in shape of (num_time, num_proxy)
        '''
        self.lonlat = lonlat
        self.temp = temp
        self.proxy = proxy

    def calc_distance(self):
        ''' Calculate the distance matrix

        Attributes:
            D (numpy.array): the great circle distance matrix
        '''
        self.D = dMat(self.lonlat)

    def calc_adj(self, distance=1000):
        ''' Calculate the adjacency matrix for a neighborhood graph

        Args:
            distance (float): the distance in unit of [km] for neighbor searching


        Attributes:
            adj (numpy.array): the adjacency matrix in shape of (num_temperature + num_proxy, num_temperature + num_proxy)

        '''
        ind_P = range(self.temp.shape[1], self.temp.shape[1]+self.proxy.shape[1])
        adj = (self.D <= distance).astype(int)              # replace 0s or 1s
        adj[np.ix_(ind_P, ind_P)] = np.eye(len(ind_P)) # set pp to 1s

        self.adj = adj
        self.distance = distance

    def get_neighbor_locs(self, idx):
        n_gridpts = self.temp.shape[1]
        i = idx + n_gridpts

        target_lon, target_lat = self.lonlat[i]
        neighbor_lons = []
        neighbor_lats = []
        neighbor_idx = []
        for j, m in enumerate(self.adj[i, :]):
            if m == 1:
                if list(self.lonlat[j]) != [target_lon, target_lat]:
                    neighbor_lon, neighbor_lat = self.lonlat[j]
                    neighbor_lons.append(neighbor_lon)
                    neighbor_lats.append(neighbor_lat)
                    neighbor_idx.append(j)

        return target_lon, target_lat, neighbor_lons, neighbor_lats, neighbor_idx

    def plot_neighbors(self, idx, figsize=(4, 4), ms=50, title=None, neighbor_clr='r', target_clr='k', edge_clr='w', marker='o', cmap=None, norm=None):
        ''' Plot the location of the neighbors according to the adjacency matrix

        Parameters
        ----------

        idx : int
            the index of the target proxy

        figsize : tuple
            the figure size

        title : str
            the title string

        ms : float
            the marker size

        neighbor_clr : str or color list
            the colors for the neighbors

        target_clr : str or color list
            the color for the target proxy

        edge_clr : str or color list
            the color for the edge of the markers

        marker : str
            the marker symbol

        Returns
        ----------

        adj : array
            the adjacency matrix in shape of (num_temperature + num_proxy, num_temperature + num_proxy)

        '''
        target_lon, target_lat, neighbor_lons, neighbor_lats, _ = self.get_neighbor_locs(idx)

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection=ccrs.Orthographic(central_longitude=target_lon, central_latitude=target_lat))
        ax.set_global()
        ax.stock_img()

        if title is None:
            ax.set_title(f'Neighbors of the target proxy (r={self.distance} km)')
        else:
            ax.set_title(title)


        transform=ccrs.PlateCarree()
        if cmap is not None:
            ax.scatter(neighbor_lons, neighbor_lats, marker=marker, s=ms, c=neighbor_clr, edgecolor=edge_clr,transform=transform, cmap=cmap, norm=norm)
        else:
            ax.scatter(neighbor_lons, neighbor_lats, marker=marker, s=ms, c=neighbor_clr, edgecolor=edge_clr,transform=transform)
        ax.scatter(target_lon, target_lat, marker=marker, s=ms, c=target_clr, edgecolor=edge_clr, transform=transform)

        return fig, ax


    def plot_neighbors_corr(self, idx, time_idx_range=None, figsize=(4, 4), ms=50, title=None,
        cmap='RdBu_r', target_clr='k', edge_clr='w', marker='o',  levels=np.linspace(-1, 1, 21),
        cbar_pad=0.1, cbar_orientation='horizontal', cbar_aspect=10, cbar_labels=[-1, -0.5, 0, 0.5, 1],
        cbar_fraction=0.15, cbar_shrink=0.5, cbar_title='Correlation', plot_cbar=True):
        ''' Plot the location of the neighbors according to the adjacency matrix

        Parameters
        ----------

        idx : int
            the index of the target proxy

        time_idx_range : list
            the list of time indices for correlation calculation

        figsize : tuple
            the figure size

        title : str
            the title string

        ms : float
            the marker size

        neighbor_clr : str or color list
            the colors for the neighbors

        target_clr : str or color list
            the color for the target proxy

        edge_clr : str or color list
            the color for the edge of the markers

        marker : str
            the marker symbol

        plot_cbar : bool
            if True, plot the colorbar

        Returns
        ----------

        adj : array
            the adjacency matrix in shape of (num_temperature + num_proxy, num_temperature + num_proxy)

        '''

        _, _, _, _, neighbor_idx = self.get_neighbor_locs(idx)
        target_value = self.proxy[:, idx]
        corrs = []
        # print('np.shape(target_value):', np.shape(target_value))
        for i in neighbor_idx:
            neighbor_value = self.temp[:, i]
            if time_idx_range is None:
                time_idx_range = ~np.isnan(neighbor_value)
            # print('np.shape(neighbor_value):', np.shape(neighbor_value))
            corr = np.corrcoef(neighbor_value[time_idx_range], target_value[time_idx_range])[1, 0]
            corrs.append(corr)

        # print(corrs)

        cmap = plt.get_cmap(cmap)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig, ax = self.plot_neighbors(idx, figsize=figsize, ms=ms, title=title, neighbor_clr=corrs, target_clr=target_clr, edge_clr=edge_clr, marker=marker, cmap=cmap, norm=norm)

        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        if plot_cbar:
            cbar = fig.colorbar(smap, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend='neither', fraction=cbar_fraction, shrink=cbar_shrink)
            cbar.ax.set_title(cbar_title)
            cbar.set_ticks(cbar_labels)

        return fig, ax
