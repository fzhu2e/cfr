import numpy as np
from scipy import linalg
import math
from . import QUIC
from . import iridge
from .GraphEstimation import graph_greedy_search
from .GraphEstimation import neighbor_graph

class GraphEM(object):
	''' The class for the GraphEM solver

	Implementation of the GraphEM algorithm

	Reference: D. Guillot. B.Rajaratnam. J.Emile-Geay. "Statistical paleoclimate reconstructions via Markov random fields."
	           Ann. Appl. Stat. 9 (1) 324 - 352, March 2015.

	Args:
	    tol (float): convergence tolerence of EM algorithm (default: 5e-3)
	    temp_r (numpy.ndarray): reconstructed temperature field
	    proxy_r (numpy.ndarray): reconstructed proxy field
	    calib (numpy.ndarray): index of calibration period
	    Sigma (numpy.ndarray): covariance matrix of the multivariate normal model
	    mu (numpy.ndarray): mean of the multivariate normal model

	'''
	def __init__(self, tol=5e-3):
		''' Initializes the GraphEM object'''

		self.tol = tol
		self.temp_r = []
		self.proxy_r = []
		self.calib = []
		self.Sigma = []
		self.mu = []

	def fit(self, temp, proxy, calib, graph=[], lonlat=[], sp_TT=3.0, sp_TP=3.0, sp_PP=3.0, N_graph=30, C0=[], M0=[], maxit=200,
		bootstrap=False, N_boot=20, distance=1000, graph_method='neighborhood', estimate_graph=True, save_graphs=False):
		'''
		Estimates the parameters of the GraphEM model and reconstruct the missing values of the temperature
		and proxy fields.

		Parameters
		----------
		temp: matrix (time x space)
			Temperature field. Missing values stored as "np.na".
		proxy: matrix (time x space)
			Proxy data. Missing values stored as "np.na". The time dimension should be
						the same as the temperature field.
		calib: vector
			Vector of indices representing the calibration period.
		graph: matrix
			Adjacency matrix of the temperature+proxy field. (Default = [], estimated via graph_greedy_search)
		lonlat: matrix ((number of temperature location + number of proxies) x 2). Default = []
			Matrix containing the (longitude, latitude) of the temperature and proxy locations. Only
			used if graph_method = 'neighborhood'.
		sp_TT: float
			Target sparsity of the temperature/temperature part of the inverse covariance matrix. Only used
			   when the graph is estimated by glasso. Default (3.0%)
		sp_TP: float
			Target sparsity of the temperature/proxy part of the inverse covariance matrix. Only used
			   when the graph is estimated by glasso. Default (3.0%)
		sp_PP: float
			Target sparsity of the proxy/proxy part of the inverse covariance matrix. Only used
			   when the graph is estimated by glasso. Default (3.0%)
		N_graph: int
			Number of graphs to consider in the graph_greedy_search method (Default = 30). Only used
		         if the graph is estimated using glasso.
		C0: matrix
			Initial estimate of the covariance matrix of the temperature+proxy field. (Default = []).
		M0: vector
			Initial estimate of the mean vector of the temperature+proxy field. (Default = []).
		bootstrap: boolean
			Indicates whether or not to produce multiple estimates of the reconstructed values via bootstrapping. (Default = False)
		N_boot: int
			Number of bootstrap samples to use if using the bootstrap method.
		distance: float
			Radius of the neighborhood graph. Only used if graph_method = 'neighborhood'. (Default = 1000 km).
		graph_method: "neighborhood" or "glasso"
			Method to use to estimate the graph. Used only if graph = [] or if estimate_graph = True.
		save_graphs: boolean
			Indicates whether or not to save all the graphs return by graph_greedy_search (Default = False).

		'''

		if ~bootstrap:
			self.temp = temp
			self.proxy = proxy
			self.calib = calib
			self.lonlat = lonlat
			self.graph = graph
			self.N_boot = N_boot

			self.graph_method = graph_method
			self.distance = distance
			self.sp_TT = sp_TT
			self.sp_TP = sp_TP
			self.sp_PP = sp_PP
			self.N_graph = N_graph

		ind_T = range(temp.shape[1])
		ind_P = range(temp.shape[1], temp.shape[1]+proxy.shape[1])
		X = np.hstack((temp, proxy))

		if bootstrap:
			self.bootstrap(N_boot = N_boot, save_graphs = save_graphs)
			self.temp_r = self.temp_r_all.mean(axis=0)
		else:
			if estimate_graph:
				if graph_method == "neighborhood":
					if len(lonlat) == 0:
						print("Error: you need to specify the longitude/latitude of temperature locations and proxies (lonlat)")
						return
					else:
						[self.graph, self.sparsity] = neighbor_graph(lonlat, ind_T, ind_P, self.distance)

				elif graph_method == "glasso":
                    # Test if there are any missing values in the calibration period
					if np.any(np.isnan(X[calib,:])):  # Use iridge to reconstruct missing values on the calibration period
						print("Reconstructing values over calibration period using RegEM iridge")
						Xridge, __, __ = self.EM(X[calib,:], [], C0, M0, maxit, use_iridge = True)
						[self.graph, self.sparsity] = graph_greedy_search(Xridge[:,ind_T], Xridge[:,ind_P], sp_TT, sp_TP, sp_PP, N_graph)
					else:
						[self.graph, self.sparsity] = graph_greedy_search(temp[calib,:], proxy[calib,:], sp_TT, sp_TP, sp_PP, N_graph)
				else:
					return "Error: graph can't be generated! Please choose a graph option."

			else:
				if graph == []:
					return "Error: you need to specify a graph if estimate_graph = False."

			[X,C,M] = self.EM(X, self.graph, C0, M0, maxit)
			self.temp_r = X[:,ind_T]
			self.proxy_r = X[:,ind_P]
			self.Sigma = C
			self.mu = M

	def EM(self, X, graph, C0 = [], M0 = [], maxit = 200, use_iridge = False):
		'''
		Expectation-Maximization (EM) algorithm with Gaussian Graphical Model

		Parameters
		----------
		X: matrix (n x p), time x space
			Temperature + Proxy matrix with missing values (np.na)
		graph: matrix (p x p)
			Adjacency matrix of the graph
		C0: matrix (p x p) 
			Initial covariance matrix of the field (Default = [], uses sample covariance matrix)
		M0: vector (p) 
			Initial mean of the field (Default = [], uses sample mean)
		maxit: int
			Maximum number of iteration of the algorithm (Default = 200)
		use_iridge: boolean
			If True, uses Ridge regularization to perform regression (Default = False)
			
		Returns
		-------
		X: matrix (n x p)
			Field with reconstructed values
		C: matrix (p x p) 
			Estimated covariance matrix of the field
		M: vector (p)
			Estimate mean vector of the field
		'''
		
       # Start EM algorithm
		print("Running GraphEM:\n")
		[n,p] = X.shape
        
        # Find unique lines
		indmis = np.isnan(X)
		nmis = len(find(indmis))
		(pattern, pattern_ix) = unique_rows(indmis)
		s = pattern.shape
		if len(s) == 1:
			nb_pattern =1
			pattern = pattern.reshape((1,s))
		else:
			nb_pattern = s[0]
			
		if M0 == []:
			# Center data
			Xmask = np.ma.masked_array(X,indmis)
			M = Xmask.mean(axis = 0)
			M = M.filled(np.nan)
		else: 
			M = M0
			
		X = X-M
		X[indmis] = 0.0

		if C0 == []:
			if use_iridge:
				C = X.T.dot(X)/(n-1)
			else:
				C = X.T.dot(X)/(n-1) + 0.1*np.eye(p)
		else:
			C = C0

		it = 0
		rdXmis = np.inf

		Xmis = np.zeros((n,p))

		print("Iter     dXmis     rdXmis\n")
        
		while ((it < maxit) & (rdXmis > self.tol)):
			it = it + 1
			CovRes = np.zeros((p,p))
			D = np.sqrt(np.diag(C))
			D[abs(D) < 1e-3] = 1.0  # Do not scale constant variables
			X = X/D
			C = (C/D)/(D[:,None])  # Correlation matrix
			for i1 in range(nb_pattern):
				pm = pattern[i1,:].sum() # Length of the patterm
				if pm > 0:
					avlr = find(~pattern[i1,:])
					misr = find(pattern[i1,:])
					if use_iridge:
						B, S, __, __ = iridge.iridge(C, avlr, misr, n-1)
					else: 
						B, S = ols(C, avlr, misr)
					ind_obs = find(pattern_ix == i1)
					mp = len(ind_obs)  # Number of rows matching current pattern
					Xmis[np.ix_(ind_obs,misr)] = X[np.ix_(ind_obs,avlr)].dot(B)
					CovRes[np.ix_(misr,misr)] = CovRes[np.ix_(misr,misr)] + mp*S
			# Return to original scaling
			X = X*D
			Xmis = Xmis * D
			C = (C*D)*(D[:,None])
			CovRes = CovRes*D*(D[:,None])
			dXmis = np.linalg.norm(Xmis[indmis] - X[indmis]) / np.sqrt(nmis)
			nXmis_pre  = np.linalg.norm((X+M)[indmis]) / np.sqrt(nmis)
			if nXmis_pre < 1e-16:
				rdXmis   = np.inf
			else:
				rdXmis   = dXmis / nXmis_pre
			X[indmis]  = Xmis[indmis]

			Mup = X.mean(axis=0)
			X = X-Mup
			M = M + Mup
				
			# Re-estimate C
			if use_iridge:
				C = (X.T.dot(X) + CovRes)/(n-1)
			else:
				C = self.fit_Sigma(np.cov(X.T) + CovRes/(n-1), self.graph)
			print("%1.3d     %1.4f     %1.4f" % (it, dXmis, rdXmis))
		X = X + M

		return [X, C, M]
            
	def fit_Sigma(self, S, graph):
		''' Estimates the covariance matrix of the field using the provided graph
		
		Parameters
		----------
		S: matrix
			Sample covariance matrix
		graph: matrix
			Adjacency matrix of the graph
			
		Returns
		-------
		C: matrix
			Estimated covariance matrix
		
		'''
		C = fitggm(S, graph)
		return C
        
	def bootstrap(self, N_boot=20, blocksize=2, n_proc=4, save_graphs = False):
		''' Block bootstrap method for the GraphEM algorithm
		
		Parameters
		----------
		N_boot: int
			Number of bootstrap samples to use (Default = 20)
		blocksize: int
			Size of the blocks used when constructing bootstrap samples (Default = 2)
		n_proc: int
			Number of processors available for distributed computing (not currently implemented). 
		save_graphs: boolean
			Indicated whether or not to save all the graphs estimated via graph_greedy_search (Default = False)
			
		Returns
		-------
		self.temp_r_all: matrix (sample x time x space)
			Matrix containing the reconstructed temperature field for each bootstrap sample
		
		'''
		temp = np.copy(self.temp)
		proxy = np.copy(self.proxy)

		[n,pt] = self.temp.shape
		verif = np.setdiff1d(range(n),self.calib)
		
		np.random.seed()
		temp_r_all = np.zeros((N_boot,n,pt))
		
		# Build bootstrap indices
		n_calib = len(self.calib)
		n_verif = len(verif)
		
		min_calib = min(self.calib)
		max_calib = max(self.calib)
		min_verif = min(verif)
		max_verif = max(verif)
		
		nb_blocks_calib = np.int(math.ceil(np.double(n_calib)/blocksize))
		nb_blocks_verif = np.int(math.ceil(np.double(n_verif)/blocksize))
		
		last_block_size_calib = np.int(n_calib - (nb_blocks_calib-1)*blocksize)
		last_block_size_verif = np.int(n_verif - (nb_blocks_verif-1)*blocksize)
		
		bootindices = []

		for i1 in range(N_boot):
			# Pick indices for the beginning of the blocks
			block_tmp_calib = np.random.choice(range(min_calib, max_calib-blocksize+1), nb_blocks_calib, replace=True)
			block_tmp_verif = np.random.choice(range(min_verif, max_verif-blocksize+1), nb_blocks_verif, replace=True)
			
			# Construct blocks
			block_tmp_c = [range(i,i+blocksize) for i in block_tmp_calib[:-1]]
			block_tmp_v = [range(i,i+blocksize) for i in block_tmp_verif[:-1]]
			# Add last blocks (that may be of a different size if the calib/verif periods are not divisible by blocksize)
			block_tmp_c.extend([range(block_tmp_calib[-1],block_tmp_calib[-1]+last_block_size_calib)])
			block_tmp_v.extend([range(block_tmp_verif[-1],block_tmp_verif[-1]+last_block_size_verif)])
		
			block_c = [i for subblock in block_tmp_c for i in subblock]        
			block_v = [i for subblock in block_tmp_v for i in subblock]        
			
			block = np.zeros(n, dtype=np.int)
			block[self.calib] = block_c
			block[verif] = block_v

			bootindices.append(block)
		
		self.bootindices = bootindices
		G = np.eye(self.temp.shape[1]+self.proxy.shape[1])

		self.boot_graphs = []
			
		# Begin bootstrap
		for i1 in range(N_boot):
			self.fit(temp[bootindices[i1],:], proxy[bootindices[i1],:], self.calib, [], self.lonlat, self.sp_TT, self.sp_TP, self.sp_PP, N_graph = self.N_graph, bootstrap=False, distance = self.distance, graph_method = self.graph_method)
			if save_graphs:
				self.boot_graphs.append(self.graph)
			C0 = self.Sigma
			M0 = self.mu
			# Imputation step (no graph estimation)
			self.fit(temp, proxy, self.calib, G, self.lonlat, self.sp_TT, self.sp_TP, self.sp_PP, C0 = C0, M0 = M0, maxit=1, graph_method = self.graph_method, distance = self.distance, estimate_graph = False)
			temp_r_all[i1,:,:] = self.temp_r
		self.temp_r_all = temp_r_all
            

def adj_matrix(S, tol=1e-3):
	'''
	Estimates an adjacency matrix via thresholding
	
	Entries larger than 'tol' are set to 1, and to 0 otherwise. 
	Diagonal entries are set to 0. 
	
	Parameters
	----------
	S: matrix
		original matrix
	tol: float
		threshold (Default = 1e-3)
		
	Returns
	-------
	adj: matrix
		Adjacency matrix
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


def fitggm(S, graph, tol=5e-3, maxit=200):
	'''
	Estimates the covariance matrix of the Gaussian graphical model via 
	maximum likelihood
	
	Parameters
	----------
	S: matrix (p x p)
		Sample covariance matrix of the field
	graph: matrix (p x p)
		Adjacency matrix of the graph
		
	tol: float
		Convergence tolerence (Default = 5e-3)
	maxit: int
		Maximum number of iterations of the algorithm (Default = 200)
		
	Returns
	-------
	W: matrix (p x p)
		Estimated covariance matrix
	'''
	p = S.shape[0]
	pp = p*(p-1)/2
	beta = np.zeros((p,p))

	W = np.copy(S)
	adj = np.copy(graph)

	for it in range(maxit):
		W_old = np.copy(W)
		for i1 in range(p):
			ind = np.copy(adj[i1,:])
			ind[i1] = 0
			indices = find(ind)
			W11_star = W[np.ix_(indices,indices)]
			s12_star = S[indices,i1]
			if ~np.all(ind==0):
				beta_star = np.linalg.solve(W11_star,s12_star)
				beta[i1,:] = 0
				beta[i1,indices] = beta_star
			else:
				beta[i1,:] = 0
			W[:,i1] = W.dot(beta[i1,:])
			W[i1,:] = W[:,i1]
			W[i1,i1] = S[i1,i1]
		Var = W - W_old
		np.fill_diagonal(Var,0)
		Var = np.triu(Var)
		meanvariation = abs(Var).sum()/pp
		W_old2 = np.copy(W_old)
		np.fill_diagonal(W_old2,0)
		W_old2 = np.triu(W_old2)
		mean_old = abs(W_old2).sum()/pp
		if meanvariation <= tol * mean_old:
			# Compute inverse covariance if desired
			#for i1 in range(p):
			#    O22 = 1/(S[i1,i1]-W[i1,:].dot(beta[i1,:]))
			#    Omega[i1,:] = -1*beta[i1,:]*O22
			#    Omega[:,i1] = Omega[i1,:]
			#    Omega[i1,i1] = O22
			return W
	print("Warning: fitggm has reached its maximum number of iterations.")
	return W


def ols(Sigma, xind, yind):
	'''
	Computes regression coefficients in a multivariate Gaussian model
	
	Parameters
	----------
	Sigma: matrix (p x p) 
		Covariance matrix of the field
	xind: vector of int 
		Indices of x
	yind: vector of int
		Indices of y
		
	Returns
	-------
	B: matrix
		Regression coefficients
	S: matrix
		Covariance of the residuals
	'''
	B = linalg.inv(Sigma[np.ix_(xind,xind)]).dot(Sigma[np.ix_(xind,yind)])
	S = Sigma[np.ix_(yind,yind)] - Sigma[np.ix_(yind, xind)].dot(B)
	return [B, S]


def unique_rows(a):
	'''
	Returns the unique rows of a matrix
	
	Parameters
	----------
	a: matrix
	
	Returns
	-------
	unique_a: matrix
		Matrix containing the unique rows of a
	idx: Vector indicating the unique rows of a
	'''
	b = np.ascontiguousarray(a)
	(uniq,idx) = np.unique(b.view(np.dtype((np.void, b.dtype.itemsize*b.shape[1]))),return_inverse = True)
	unique_a = uniq.view(b.dtype).reshape(-1, b.shape[1])
	return (unique_a,idx)
    
class verif_stats:
	'''
	Class to compute verification statistics of a reconstructed temperature field
	
	Attributes
	----------
	T_hat: matrix (n x p (no bootstrap) or N x n x p (bootstrap)) 
		Reconstructed temperature field
	T: matrix(n x p)
		Target temperature field
	calib: vector of int
		Indices of the calibration period
		
	Usage
	-----
	Statistics are computed at creation of the object and are stored in 
	the MSE, RE, CE, and R2 attributes. 
	
	If T_hat is an N x n x p matrix 
	(i.e., N reconstructions are provided), the attribute 
	"XXX_all" is a matrix (the i-th row is the value of the statistic for the 
	i-th reconstruction), "XXX" contains the value of the statistic averaged over 
	all reconstructions, and "XXX_std" contains the standard deviation of the 
	statistics ("XXX" = MSE, RE, CE, or R2).  
	
	Methods
	-------
	_MSE: mean squared error
	_RE: Reduction of error
	_CE: Coefficient of efficiency
	_R2: Coefficient of determination (R^2)
	stats: computes all the statistics
	__str__: displays the MSE, RE, CE, and R2
	'''
	def __init__(self, temp_r, temp_target, calib):
		'''
		Initializes the verif_stats object
		
		Parameters
		----------
		temp_r: matrix (n x p (no bootstrap) or N x n x p (bootstrap))
			Reconstructed temperature field
		temp_target: matrix (n x p)
			Target temperature field
		calib: vector of int
			Indices of the calibration period
		'''
		self.T_hat = temp_r
		self.T = temp_target
		self.calib = calib
		self.stats()

	def _MSE(self, T, T_hat, calib):
		'''
		Computes the mean squared error (MSE) of the reconstruction 
		'''
		[n,p] = T.shape
		verif = np.setdiff1d(range(n),calib)
		Tv = T[verif,:]
		Tv_hat = T_hat[verif,:]
		return ((Tv - Tv_hat)**2).mean(axis=0)
		
	def _RE(self, T, T_hat, calib):
		'''
		Computes the reduction of error (RE) of the reconstruction
		'''
		[n,p] = T.shape
		verif = np.setdiff1d(range(n),calib)
		Tv = T[verif,:]
		Tv_hat = T_hat[verif,:]
		
		Tbar_calib = T[calib,:].mean(axis=0)
		
		RE = 1 - ((Tv-Tv_hat)**2).sum(axis=0) / ((Tv-Tbar_calib)**2).sum(axis=0)
		return RE
		
	def _CE(self, T, T_hat, calib):
		'''
		Computes the coefficient of efficiency (CE) of the reconstruction
		'''
		[n,p] = T.shape
		verif = np.setdiff1d(range(n),calib)
		Tv = T[verif,:]
		Tv_hat = T_hat[verif,:]
		
		Tbar_verif = T[verif,:].mean(axis=0)
		
		CE = 1 - ((Tv-Tv_hat)**2).sum(axis=0) / ((Tv-Tbar_verif)**2).sum(axis=0)
		return CE

	def _R2(self, T, T_hat, calib):
		'''
		Computes the coefficient of determination (R^2) of the reconstruction
		'''
		[n,p] = T.shape
		verif = np.setdiff1d(range(n),calib)
		Tv = T[verif,:]
		Tv_hat = T_hat[verif,:]
		R2 = np.zeros(p)
		
		for i1 in range(p):
			rho = np.corrcoef(Tv[:,i1],Tv_hat[:,i1])  
			R2[i1]=rho[0,1]**2
			
		return R2

	def stats(self):
		'''
		Computes the MSE, RE, CE, and R^2 statistics. 
		'''
		if len(self.T_hat.shape) == 3:
			# bootstrap 
			self.MSE_all = np.zeros((self.T_hat.shape[0],self.T_hat.shape[2]))
			self.RE_all = np.zeros((self.T_hat.shape[0],self.T_hat.shape[2]))
			self.CE_all = np.zeros((self.T_hat.shape[0],self.T_hat.shape[2]))
			self.R2_all = np.zeros((self.T_hat.shape[0],self.T_hat.shape[2]))
			
			for i1 in range(self.T_hat.shape[0]):
				self.MSE_all[i1,:] = self._MSE(self.T,self.T_hat[i1,:,:],self.calib)
				self.RE_all[i1,:] = self._RE(self.T,self.T_hat[i1,:,:],self.calib)
				self.CE_all[i1,:] = self._CE(self.T,self.T_hat[i1,:,:],self.calib)
				self.R2_all[i1,:] = self._R2(self.T,self.T_hat[i1,:,:],self.calib)
		
			self.MSE = self.MSE_all.mean(axis=0)
			self.RE = self.RE_all.mean(axis=0)
			self.CE = self.CE_all.mean(axis=0)
			self.R2 = self.R2_all.mean(axis=0)
			
			self.MSE_std = self.MSE_all.std(axis=0)
			self.RE_std = self.RE_all.std(axis=0)
			self.CE_std = self.CE_all.std(axis=0)
			self.R2_std = self.R2_all.std(axis=0)
			
		else:
			self.MSE = self._MSE(self.T,self.T_hat,self.calib)
			self.RE = self._RE(self.T,self.T_hat,self.calib)
			self.CE = self._CE(self.T,self.T_hat,self.calib)
			self.R2 = self._R2(self.T,self.T_hat,self.calib)
		
	def __str__(self):
		'''
		Displays the calculated values of MSE, RE, CE, and R2
		'''
		return "Mean MSE = %1.4f, Mean RE = %1.4f, Mean CE = %1.4f, Mean R2 = %1.4f" % (self.MSE.mean(), self.RE.mean(), self.CE.mean(), self.R2.mean())    
		
def find(condition):
	'''
	Code to replace the old matplotlib.mlab.find function
	(Return the indices where some condition is true)
	'''
	res, = np.nonzero(np.ravel(condition))
	return res  
