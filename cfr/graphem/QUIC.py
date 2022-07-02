# compile C++ file with (Linux-compatible): g++ -shared QUIC.cpp -o quiclib.so -fPIC -llapack -lblas -lstdc++

import numpy as np
import os
from ctypes import CDLL, c_double, c_int


lib_path = os.path.dirname(os.path.abspath(__file__))
quiclib = CDLL(os.path.join(lib_path,'quiclib.so'))

def QUIC(Si, rho):
	'''
	Solves the graphical lasso problem using QUIC.
	(Wrapper for the C++ code QUIC.cpp.)
	Reference: Hsieh et al., "QUIC: Quadratic Approximation for Sparse Inverse
		Covariance Estimation", Journal of Machine Learning Research 15 (2014) 2911-2947. 
	Note: In order to use the code, the C++ code QUIC.cpp needs to be compiled first. 

	Parameters:
	-----------
	Si: matrix
		Sample covariance matrix
	rho: float
		Regularization parameter
	Returns:
	--------
	Sigma: matrix
		Estimated covariance matrix
	Omega: matrix
		Estimated inverse covariance matrix
	'''
	d1 = Si.shape[0]
	d = d1**2
	Si = Si.reshape(d)
	rho = rho.reshape(d)
	Sigma = np.zeros(d)
	Omega = np.zeros(d)
	S = (c_double*d)()
	rhomat = (c_double*d)()
	O = (c_double*d)()
	W = (c_double*d)()
	Id = np.eye(d1).reshape(d)
	for i1 in range(d):
		S[i1] = (c_double)(Si[i1])
		rhomat[i1] = (c_double)(rho[i1])
		W[i1] = (c_double)(Id[i1])
		O[i1] = (c_double)(Id[i1])
		
	n = (c_int*1)(d1)
	npath = (c_int*1)(1)
	path = (c_int*1)(0)
	tol = (c_double*1)(1e-3)
	msg = (c_int*1)(0)
	maxIter = (c_int*1)(1000)
	opt = (c_double*1)(0)
	cputime = (c_double*1)(0)
	it = (c_int*1)(0)
	dGap = (c_double*1)(0)

	quiclib.QUIC('d',n,S,rhomat,npath,path,tol,msg,maxIter,O,W,opt,cputime,it,dGap)

	for i1 in range(d):
		Sigma[i1] = W[i1]
		Omega[i1] = O[i1]
	Sigma = Sigma.reshape((d1,d1))
	Omega = Omega.reshape((d1,d1))
	return [Sigma, Omega]
    
