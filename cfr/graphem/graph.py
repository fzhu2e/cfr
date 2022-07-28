from .GraphEstimation import dMat
from .GraphEstimation import graph_sparsity
from .GraphEstimation import graph_greedy_search

import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


class Graph:
    ''' The object for a graph '''

    def __init__(self, lonlat, field, proxy):
        '''
        Args:
            lonlat (numpy.array): the location array with shape (num_grid + num_proxy, 2), where the 2nd dimension is aranged as [longitude, latitude]
            field (numpy.array): the climate field with shape (num_time, num_grid)
            proxy (numpy.array): the proxy matrix with shape (num_time, num_proxy)
        '''
        self.lonlat = lonlat
        self.field = field
        self.proxy = proxy

    def calc_distance(self):
        ''' Calculate the distance matrix

        Attributes:
            dist_mat (numpy.array): the great circle distance matrix
        '''
        
        self.dist_mat = dMat(self.lonlat)
        

    def neigh_adj(self, cutoff_radius = None, n_neighbors = None):
        ''' Calculate the adjacency matrix for a neighborhood graph

        The neighborhood is controlled either/and by the cutoff radius or the 
        number of neigbors.  If both are specified, the beighborhood is 
        comprised of the `n_neighbors` closest points within a distance of 
        `cutoff_radius`. An error is raised if neither method is specified.

        Parameters
        ----------
        cutoff_radius : float
            the great circle distance in unit of [km] for neighbor search.
            Default: None
        n_neighbors : int
            number of closest neighbors (Default: None)

        Returns
        -------
        adj (numpy.array): the adjacency matrix (num_grid + num_proxy, num_grid + num_proxy)
        '''
        
        if not hasattr(self, 'dist_mat'):
            self.calc_distance()
            
        # make distance matrix upper triangular
        D = self.dist_mat.copy()
        D[np.tril_indices_from(D)] = np.nan
        p = D.shape[0] # matrix size
        
        adj   = np.zeros((p,p))   
        ind_F = range(self.field.shape[1])
        ind_P = range(self.field.shape[1], p)
        
        if cutoff_radius is not None and n_neighbors is not None:        
            D[D>cutoff_radius] = np.Inf
            for j in range(p):       
                idx = np.argsort(D[j,:])
                nonnan = np.sum(~np.isnan(D[j,idx]))
                adj[j,idx[:np.min([nonnan,n_neighbors])]] = 1

        elif cutoff_radius is not None and n_neighbors is None:   
            adj = (self.dist_mat <= cutoff_radius).astype(int)  # Booleans as 0s or 1s
            
        elif cutoff_radius is None and n_neighbors is not None:
            for j in range(p):       
                idx = np.argsort(D[j,:])
                nonnan = np.sum(~np.isnan(D[j,idx]))
                adj[j,idx[:np.min([nonnan,n_neighbors])]] = 1
        else:
            print('No rule was specified to create the neighborhood')
        
        # symmetrize adjacency matrix
        adj = np.where(adj,adj,adj.T)  # h/t https://stackoverflow.com/a/58718913/4984978
        
        # set PP block to identity
        adj[np.ix_(ind_P, ind_P)] = np.eye(len(ind_P)) 
        # restore diagnonal over ind_F
        for j in ind_F:   
            adj[j,j] = 1
        
        self.adj = adj
        self.cutoff_radius = cutoff_radius
        self.n_neighbors = n_neighbors
        self.sparsity = graph_sparsity(adj, ind_F, ind_P)
        
    def glasso_adj(self, target_FF, target_FP):
        ''' Calculates the adjacency matrix using the Graphical LASSO. 
            Uses a QUIC solver and a greedy approach to find a graph 
            with a given target sparsity in the FF and FP parts of 
            the adjacency matrix. The PP part is assumed diagonal.

        Parameters
        ----------
        
        target_FF: float in [0,100] 
            Target sparsity of the in-field part of the graph (percentage)
        target_FP: 
            Target sparsity of the field/proxy part of the graph (percentage)
            
        Returns
        -------
        adj : matrix, (p1+p2) x (p1+p2)
            Estimated adjacency matrix (graph) of the total data matrix
        sparsity:  float x 3
            Sparsity of the estimated graph
            
        References
        ----------
        Hsieh et al., "QUIC: Quadratic Approximation for Sparse Inverse
    		Covariance Estimation", Journal of Machine Learning Research 15 (2014) 2911-2947.     
        
        Guillot, Rajaratnam & Emile-Geay. "Statistical paleoclimate reconstructions 
            via Markov random fields" Ann. Appl. Stat. 9 (1) 324 - 352, March 2015.    

        '''
        adj, sp = graph_greedy_search(self.field, self.proxy, target_FF, target_FP)      
        self.adj = adj
        self.sparsity = sp
        
        
    def plot_adj(self,figsize=(6, 6), clr='C0', ax=None, title=None):
        ''' Plot the adjacency matrix for a neighborhood graph

        Args:
            figsize (tuple) : figure size (optional)
            clr : color of the rectangular patches and text delineating regions of the matrix   


        Attributes:
            adj (numpy.array): the adjacency matrix in shape of (num_grid + num_proxy, num_grid + num_proxy)

        '''
        num_tot = self.adj.shape[1]
        num_grid = self.field.shape[1]
        num_proxy = self.proxy.shape[1]
        
        assert num_tot == num_grid+num_proxy, "matrix dimensions do not add up"
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize) # in inches

        ax.imshow(self.adj,cmap="Greys",interpolation="none")
        ax.set_xlabel('Index')
        
        # TODO : better label positions, using rcParams
        
        # plot climate-climate part of the graph
        ax.add_patch(plt.Rectangle((0, 0), num_grid, num_grid, alpha = 0.3,
                                   fc=clr, # face color
                                   ec='none'))  # edge color
        ax.annotate('in-field', color=clr,
                    xy=(num_grid/2, 0),xycoords='data',
                    xytext=(num_grid/2/num_tot, 1.02), textcoords='axes fraction')
        
        # plot climate-proxy part of the graph
        fac= 0.75
        ax.add_patch(plt.Rectangle((num_grid, 0), num_proxy-2, num_grid,
                                   fc='none', ec=clr, linewidth=2))
        ax.annotate('field-proxy', color=clr,
                    xy=(num_grid+fac*num_proxy/2, 0),xycoords='data',
                    xytext=((num_grid+fac*num_proxy/2)/num_tot, 1.02), textcoords='axes fraction')
        
        # plot proxy-proxy part of the graph
        ax.add_patch(plt.Rectangle((num_grid+1, num_grid+1), num_proxy-2, num_proxy,
                                   linewidth=2,ls='--',
                                   fc='none', ec=clr))
                    
        ax.text(1.05*num_tot,num_grid+1.4*num_proxy/2, s='proxy-proxy', 
                rotation='vertical', color = clr)

        if title is not None:
            ax.set_title(title, y=1.1)
        
        if 'fig' in locals():
            return fig, ax
        else:
            return ax
        

    def get_neighbor_locs(self, idx):
        n_gridpts = self.field.shape[1]
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

    def plot_neighbors(self, idx, figsize=(4, 4), ms=50, title=None, neighbor_clr='r',
                       target_clr='k', edge_clr='w', marker='o', cmap=None, norm=None, ax=None):
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
            the adjacency matrix in shape of (num_grid + num_proxy, num_grid + num_proxy)

        '''
        target_lon, target_lat, neighbor_lons, neighbor_lats, _ = self.get_neighbor_locs(idx)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot(projection=ccrs.Orthographic(central_longitude=target_lon, central_latitude=target_lat))

        ax.set_global()
        ax.stock_img()

        if title is None:
            if hasattr(self, 'cutoff_radius'):
                ax.set_title(f'Neighbors (r={self.cutoff_radius} km)')
            else:
                ax.set_title(f'Neighbors of the target proxy')
        else:
            ax.set_title(title)


        transform=ccrs.PlateCarree()
        if cmap is not None:
            ax.scatter(neighbor_lons, neighbor_lats, marker=marker, s=ms, c=neighbor_clr, edgecolor=edge_clr,transform=transform, cmap=cmap, norm=norm)
        else:
            ax.scatter(neighbor_lons, neighbor_lats, marker=marker, s=ms, c=neighbor_clr, edgecolor=edge_clr,transform=transform)
        ax.scatter(target_lon, target_lat, marker=marker, s=ms, c=target_clr, edgecolor=edge_clr, transform=transform)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax


    def plot_neighbors_corr(self, idx, time_idx_range=None, figsize=(5, 5), ms=50, title=None,
        cmap='RdBu_r', target_clr='k', edge_clr='w', marker='o',  levels=np.linspace(-1, 1, 21),
        cbar_pad=0.1, cbar_orientation='horizontal', cbar_aspect=10, cbar_labels=[-1, -0.5, 0, 0.5, 1],
        cbar_fraction=0.15, cbar_shrink=0.5, cbar_title='Correlation', plot_cbar=True, ax=None):
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
            the adjacency matrix in shape of (num_grid + num_proxy, num_grid + num_proxy)

        '''
        _, _, _, _, neighbor_idx = self.get_neighbor_locs(idx)
        target_value = self.proxy[:, idx]
        corrs = []
        # print('np.shape(target_value):', np.shape(target_value))
        for i in neighbor_idx:
            neighbor_value = self.field[:, i]
            if time_idx_range is None:
                time_idx_range = ~np.isnan(neighbor_value)
            # print('np.shape(neighbor_value):', np.shape(neighbor_value))
            corr = np.corrcoef(neighbor_value[time_idx_range], target_value[time_idx_range])[1, 0]
            corrs.append(corr)
        # print(corrs)

        cmap = plt.get_cmap(cmap)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        if ax is None:
            fig, ax = self.plot_neighbors(
                idx, figsize=figsize, ms=ms, title=title, neighbor_clr=corrs, target_clr=target_clr,
                edge_clr=edge_clr, marker=marker, cmap=cmap, norm=norm)
        else:
            ax = self.plot_neighbors(
                idx, figsize=figsize, ms=ms, title=title, neighbor_clr=corrs, target_clr=target_clr,
                edge_clr=edge_clr, marker=marker, cmap=cmap, norm=norm, ax=ax)

        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        if plot_cbar:
            cbar = plt.colorbar(smap, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend='neither', fraction=cbar_fraction, shrink=cbar_shrink)
            cbar.ax.set_title(cbar_title)
            cbar.set_ticks(cbar_labels)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax


