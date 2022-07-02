import numpy as np
from . import QUIC

def graph_greedy_search(temp_inst, proxy_inst, target_TT, target_TP, target_PP, N=30, maxit=500):
	'''
	Greedy approach to estimate the conditional independence struture of the field using the graphical lasso. 
	Attemps to find a graph with a given target sparsity in the TT, TP, and PP parts of the adjacency matrix. 
	For more details, see D. Guillot. B.Rajaratnam. J.Emile-Geay. "Statistical paleoclimate reconstructions via Markov random fields." 
	   Ann. Appl. Stat. 9 (1) 324 - 352, March 2015.
			   
	Parameters:
	temp_inst: matrix, n x p1(time x space) 
		Temperature field over the instrumental period. Cannot have missing values. 
	proxy_inst: matrix, n x p2 (time x space)
		Proxy field over the instrumental period. Cannot have missing values. 
	target_TT: float in [0,100] 
		Target sparsity of the temperature/temperature part of the graph (percentage)
	target_TP: 
		Target sparsity of the temperature/proxy part of the graph (percentage)
	target_PP: 
		Target sparsity of the proxy/proxy part of the graph (percentage)
	N: int
		Number of graphs to consider (Default = 30)
	maxit: int
		maximum number of iterations of the algorithm (Default = 500)
		
	Returns:
	-------
	adj_prec: matrix, (p1+p2) x (p1+p2)
		Estimated adjacency matrix of the graph of the temperature+proxy field
	sparsity_prec, float
		Sparsity of the estimated graph
	'''
	print("Estimating graph using greedy search")

	X = np.hstack((temp_inst, proxy_inst))
	ind_T = range(temp_inst.shape[1])
	ind_P = range(temp_inst.shape[1],temp_inst.shape[1]+proxy_inst.shape[1])
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

	c_TT = N-1
	c_TP = N-1
	c_PP = N-1

	it = 0
	stop = 0

	visited = np.array([])   # Visited positions
	sparsity = [0.0,0.0,0.0]

	print("Iter    TT      TP      PP\n")
	
	while (stop == 0) & (it < maxit):

		sparsity_prec = np.copy(sparsity)
		adj_prec = np.copy(adj)
		# Build penalty matrix
		pen[np.ix_(ind_T,ind_T)] = rhos[c_TT]
		pen[np.ix_(ind_T,ind_P)] = rhos[c_TP]
		pen[np.ix_(ind_P,ind_T)] = rhos[c_TP]
		pen[np.ix_(ind_P,ind_P)] = rhos[c_PP]

		[W, O] = QUIC.QUIC(S, pen)
		# Compute sparsity of the different parts (TODO)
		[sparsity_TT, sparsity_TP, sparsity_PP, adj] = sp_level_3(O, ind_T, ind_P)
		sparsity = [sparsity_TT, sparsity_TP, sparsity_PP]

		if (sparsity_TT < target_TT):
			c_TT = max(0,c_TT-1)

		if (sparsity_TP < target_TP):
			c_TP = max(0,c_TP-1);

		if (sparsity_PP < target_PP):
			c_PP = max(0,c_PP-1)

		it = it + 1
		# Print current iteration information
		print("%1.3d  %6.3f  %6.3f  %6.3f" % (it, sparsity_TT, sparsity_TP, sparsity_PP))
		c_point = c_TT + c_TP*N + c_PP*N**2

		if (c_point in visited):
			stop = 1
		else:
			visited = np.append(visited, c_point)

	return [adj_prec,sparsity_prec]


from numpy import cos, sin, arcsin, sqrt, radians
import itertools


def neighbor_graph(lonlat, ind_T, ind_P, distance = 1000):
	'''
	Constructs a neighborhood graph. Vertices are connected 
	if and only if they are at less than a given distance.
	
	Parameters:
	----------
	lonlat: matrix, p x 2
		Matrix where each row contains the (longitude, latititude) of a location.
	ind_T: vector
		Vector containing the indices of the temperature locations
	ind_P: vector 
		Vector containing the indices of the proxies
	distance: float, positive
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
	adj = (D <= distance).astype(np.int)           # replace 0s or 1s
	adj[np.ix_(ind_P, ind_P)] = np.eye(len(ind_P)) # set pp to 1s
	sparsity = graph_sparsity(adj, ind_T, ind_P)

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

def graph_sparsity(adj, ind_T, ind_P):
	'''
	Computes the sparsity level of the graph specified by an adjacency matrix in the 
	temperature/temperature, temperature/proxy, and proxy/proxy parts of the matrix.

	Parameters:
	-----------
	adj: matrix
		Adjacency matrix of the graph
	ind_T: vector
		Vector containing the indices of the temperature locations
	ind_P: vector
		Vector containing the indices of the proxies
		
	Returns:
	--------
	sparsity_TT: float
		Sparsity level of the temperature/temperature part of the graph
	sparsity_TP: float 
		Sparsity level of the temperature/proxy part of the graph
	sparsity_PP: float
		Sparsity level of the proxy/proxy part of the graph
	'''
	n_T = len(ind_T)
	n_P = len(ind_P)
	sparsity_TT = (adj[np.ix_(ind_T, ind_T)] - np.eye(n_T)).sum()/(n_T**2-n_T)  # n**0.5 - n 
	sparsity_TP = adj[np.ix_(ind_T, ind_P)].sum()/(n_T*n_P)
	sparsity_PP = 0.0

	return [sparsity_TT, sparsity_TP, sparsity_PP]

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

def sp_level_3(O, ind_T, ind_P):
	'''
	Computes the sparsity level (percentage of nonzero entries) 
	of a matrix in the temperature/temperature, 
	temperature/proxy, and proxy/proxy part of the matrix O.

	Parameters
	----------
	O: matrix
		Matrix to compute the sparsity level of
	ind_T: vector of int
		Indices of the temperature locations
	ind_P: vector of int
		Indices of the proxy locations
		
	Returns
	-------
	level_TT: sparsity level of the temperature/temperature block of the matrix
	level_TP: sparsity level of the temperature/proxy block of the matrix
	level_PP: sparsity level of the proxy/proxy block of the matrix
	adj: adjacency matrix associated to the matrix O
	'''
	p = O.shape[0]
	d = np.diag(O)
	dinv = np.sqrt(d)**(-1)
	D = np.diag(dinv)
	O_scaled = D.dot(O).dot(D)
	adj = adj_matrix(O_scaled)
	p_TT = len(ind_T)
	p_PP = len(ind_P)

	adj_TT = adj[np.ix_(ind_T, ind_T)]
	level_TT = adj_TT.sum().astype(float) / (p_TT*(p_TT-1))*100

	adj_TP = adj[np.ix_(ind_T, ind_P)]
	level_TP = adj_TP.sum().astype(float) / (p_TT*p_PP)*100

	adj_PP = adj[np.ix_(ind_P, ind_P)]
	level_PP = adj_PP.sum().astype(float) / (p_PP*(p_PP-1))*100

	adj = adj + np.eye(p)
	return [level_TT, level_TP, level_PP, adj]
