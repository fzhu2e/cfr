import numpy as np
from . import QUIC

def graph_greedy_search(field_inst, proxy_inst, target_FF, target_FP, target_PP = 0.0, N=30, maxit=500):
    '''
    Solves the graphical lasso problem using the QUIC algorithm.
    Uses a greedy approach to find a graph with a given target sparsity in the
    FF and FP parts of the adjacency matrix. The PP part is assumed diagonal.
               
    Parameters
    ----------
    field_inst: matrix, n x p1 (time x space) 
        Climate field over the instrumental period. Cannot have missing values. 
    proxy_inst: matrix, n x p2 (time x space)
        Proxy field over the instrumental period. Cannot have missing values. 
    target_FF: float in [0,100] 
        Target sparsity of the in-field part of the graph (percentage)
    target_FP: float in [0,100] 
        Target sparsity of the field/proxy part of the graph (percentage)
    target_PP: float in [0,100] 
        Target sparsity of the proxy/proxy part of the graph 
        (this is a dummy argument for back-compatibility ; internally set to 0)    
    N: int
        Number of graphs to consider (Default = 30)
    maxit: int
        maximum number of iterations of the algorithm (Default = 500)
        
    Returns
    -------
    adj_prec: matrix, (p1+p2) x (p1+p2)
        Estimated adjacency matrix (graph) of the total data matrix
    sparsity_prec: float
        Sparsity of the estimated graph
        
    References
    ----------
    Hsieh et al., "QUIC: Quadratic Approximation for Sparse Inverse
		Covariance Estimation", Journal of Machine Learning Research 15 (2014) 2911-2947.     
    
    D. Guillot. B.Rajaratnam. J.Emile-Geay. "Statistical paleoclimate reconstructions via Markov random fields." 
       Ann. Appl. Stat. 9 (1) 324 - 352, March 2015.    
    '''
    # TODO: ask Dominique to rewrite this for FF, FP search only.
    
    if np.isnan(field_inst).any() or np.isnan(proxy_inst).any():
        raise ValueError("Error: this method requires the data matrices to be gap-free")
        
    print("Solving graphical LASSO using greedy search")

    X = np.hstack((field_inst, proxy_inst))
    ind_F = range(field_inst.shape[1])
    ind_P = range(field_inst.shape[1],field_inst.shape[1]+proxy_inst.shape[1])
    S = np.corrcoef(X.T)
    p = S.shape[0]
    pen = np.zeros((p,p))   # Penality matrix
    adj = np.zeros((p,p))
    adj_prec = np.zeros((p,p))

    rho_max = (S - np.diag(np.diag(S))).max()
    rho_min = 0.1*rho_max

    O = np.eye(p)
    W = np.eye(p)

    rhos = np.linspace(rho_min, rho_max, N)

    c_FF = N-1
    c_FP = N-1
    c_PP = N-1

    it = 0
    stop = 0

    visited = np.array([])   # Visited positions
    sparsity = [0.0,0.0,0.0]
    

    print("Iter    FF      FP      PP\n")
    
    while (stop == 0) & (it < maxit):
        sparsity_prec = np.copy(sparsity)
        adj_prec = np.copy(adj)
        # Build penalty matrix
        pen[np.ix_(ind_F,ind_F)] = rhos[c_FF]
        pen[np.ix_(ind_F,ind_P)] = rhos[c_FP]
        pen[np.ix_(ind_P,ind_F)] = rhos[c_FP]
        pen[np.ix_(ind_P,ind_P)] = rhos[c_PP]

        [W, O] = QUIC.QUIC(S, pen)
        # Compute sparsity of the different parts (TODO)
        [sparsity_FF, sparsity_FP, sparsity_PP, adj] = sp_level_3(O, ind_F, ind_P)
        sparsity = [sparsity_FF, sparsity_FP, sparsity_PP]

        if (sparsity_FF < target_FF):
            c_FF = max(0,c_FF-1)

        if (sparsity_FP < target_FP):
            c_FP = max(0,c_FP-1);

        if (sparsity_PP < target_PP):
            c_PP = max(0,c_PP-1)

        it = it + 1
        # Print current iteration information
        print("%1.3d  %6.3f  %6.3f  %6.3f" % (it, sparsity_FF, sparsity_FP, sparsity_PP))
        c_point = c_FF + c_FP*N + c_PP*N**2

        if (c_point in visited):
            stop = 1
        else:
            visited = np.append(visited, c_point)
    
    # for consistency with neighborhood graphs, set PP to 1s
    adj_prec[np.ix_(ind_P, ind_P)] = np.eye(len(ind_P))
    # print(type(adj_prec), type(sparsity_prec))

    return adj_prec, sparsity_prec


from numpy import cos, sin, arcsin, sqrt, radians
import itertools


def neighbor_graph(lonlat, ind_F, ind_P, cutoff_radius = 1000):
    '''
    Constructs a neighborhood graph. Vertices are connected 
    if and only if they are at less than a given cutoff radius.
    
    Parameters:
    ----------
    lonlat: matrix, p x 2
        Matrix where each row contains the (longitude, latititude) of a location.
    ind_F: vector
        Vector containing field gridpoint indices
    ind_P: vector 
        Vector containing the indices of the proxies
    cutoff_radius: float, positive
        Radius of the neighborhood graph
        
    Returns: 
    --------
    adj: matrix
        Adjacency matrix of the neighborhood graph
    sparsity: float
        Sparsity level of the adjacency matrix
    '''
    print("Estimating graph using neighborhood method")

    D = dMat(lonlat)
    adj = (D <= cutoff_radius).astype(np.int)           # replace 0s or 1s
    adj[np.ix_(ind_P, ind_P)] = np.eye(len(ind_P)) # set pp to 1s
    sparsity = graph_sparsity(adj, ind_F, ind_P)

    return [adj, sparsity]


def great_circle_distance(lon1, lat1, lon2, lat2):
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
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * arcsin(sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km

def dMat(lonlat):
    '''
    Computes a matrix containing the distance between every pair of points 
    
    Parameters:
    -----------
    lonlat: matrix, p x 2
        Matrix where each row contains the (longitude, latititude) of a location.
        
    Returns:
    --------
    D: matrix, p x p
        Matrix whose (i,j)-th entry contains the great circle distance (in km) between location i 
        and location j.
    '''
    N = lonlat.shape[0]
    D = np.zeros((N,N))
    for pair in itertools.product(range(N),repeat=2):
        D[pair] = great_circle_distance(lonlat[pair[0],0],lonlat[pair[0],1],lonlat[pair[1],0],lonlat[pair[1],1])

    return D

def graph_sparsity(adj, ind_F, ind_P):
    '''
    Computes the sparsity level of the graph specified by an adjacency matrix in the 
    in-field, field/proxy, and proxy/proxy parts of the matrix.

    Parameters:
    -----------
    adj: matrix
        Adjacency matrix of the graph
    ind_F: vector
        Vector containing the indices of the field locations
    ind_P: vector
        Vector containing the indices of the proxies
        
    Returns:
    --------
    sparsity_FF: float
        Sparsity level of the in-field part of the graph
    sparsity_FP: float 
        Sparsity level of the field/proxy part of the graph
    sparsity_PP: float
        Sparsity level of the proxy/proxy part of the graph
    '''
    n_T = len(ind_F)
    n_P = len(ind_P)
    sparsity_FF = (adj[np.ix_(ind_F, ind_F)] - np.eye(n_T)).sum()/(n_T**2-n_T)  # n**0.5 - n 
    sparsity_FP = adj[np.ix_(ind_F, ind_P)].sum()/(n_T*n_P)
    sparsity_PP = 0.0

    return [sparsity_FF, sparsity_FP, sparsity_PP]

def adj_matrix(S, tol=1e-3):
    '''
    Computes the adjacency matrix of the graph associated to a matrix
    via thresholding

    Parameters:
    -----------
    S: matrix
        Matrix to compute the graph of
    tol: thresholding value (Default = 1e-3)

    Returns:
    --------
    adj: matrix
        Adjacency matrix of the graph associated to S.
    '''
    adj = (abs(S) > tol).astype(int)
    adj = adj - np.diag(np.diag(adj))
    return adj

def sp_level_3(O, ind_F, ind_P):
    '''
    Computes the sparsity level (percentage of nonzero entries) 
    of a matrix in the in-field, 
    field/proxy, and proxy/proxy part of the matrix O.

    Parameters
    ----------
    O: matrix
        Matrix to compute the sparsity level of
    ind_F: vector of int
        Indices of the field locations
    ind_P: vector of int
        Indices of the proxy locations
        
    Returns
    -------
    level_FF: sparsity level of the in-field block of the matrix
    level_FP: sparsity level of the field/proxy block of the matrix
    level_PP: sparsity level of the proxy/proxy block of the matrix
    adj: adjacency matrix associated to the matrix O
    '''
    p = O.shape[0]
    d = np.diag(O)
    dinv = np.sqrt(d)**(-1)
    D = np.diag(dinv)
    O_scaled = D.dot(O).dot(D)
    adj = adj_matrix(O_scaled)
    p_FF = len(ind_F)
    p_PP = len(ind_P)

    adj_FF = adj[np.ix_(ind_F, ind_F)]
    level_FF = adj_FF.sum().astype(float) / (p_FF*(p_FF-1))*100

    adj_FP = adj[np.ix_(ind_F, ind_P)]
    level_FP = adj_FP.sum().astype(float) / (p_FF*p_PP)*100

    adj_PP = adj[np.ix_(ind_P, ind_P)]
    level_PP = adj_PP.sum().astype(float) / (p_PP*(p_PP-1))*100

    adj = adj + np.eye(p)
    return [level_FF, level_FP, level_PP, adj]
