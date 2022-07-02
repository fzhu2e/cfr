import numpy as np
from scipy.sparse.linalg import eigs
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
#####

'''
Collections of functions to perform Ridge regression, with 
generalized cross-validation to choose the Ridge regularization 
parameter.
'''

def gcvfctn (h, d, fc2, trS0, dof0):
	'''
		h:  regularization parameter,
		d:  column vector of eigenvalues of cov(X),
	  fc2:  row sum of squared Fourier coefficients, fc2=sum(F.^2, 2),
	 trS0:  trace(S0) = Frobenius norm of generic part of residual matrix,
	 dof0:  degrees of freedom in estimate of residual covariance
			matrix when regularization parameter is set to zero
	'''

	filfac = (h**2) / (d + h**2)
	filfac = filfac.reshape(len(filfac),1)
	A = dof0 + sum(filfac)
	g = (sum((filfac**2) * fc2) + trS0 ) / A**2
	return g

#####

def gcvridge(F, d, trS0, n, r, trSmin, minvarfrac = 0):
	'''
		F:  the matrix of Fourier coefficients,
		d:  column vector of eigenvalues of X'*X/n,
	 trS0:  trace(S0) = trace of generic part of residual 2nd moment matrix,
		n:  number of degrees of freedom for estimation of 2nd moments,
		r:  number of nonzero eigenvalues of X'*X/n,
	trSmin:  minimum of trace(S_h) to construct approximate lower bound
			on regularization parameter h (to prevent GCV from choosing
			too small a regularization parameter).
	'''
	# Ensure d is in double
	d = np.atleast_1d(d)

	# Check sizes of input matrices
	if len(d) < r:
		print("All nonzero eigenvalues must be given.")
		return

	if minvarfrac < 0  or minvarfrac > 1:
		print("error: minvarfrac must be in [0,1].")
		return

	p = F.shape[0]

	if p < r:
		print("F must have at least as many rows as there are nonzero eigenvalues d")
		return

	# Row sum of squared Fourier coefficients

	#fc2 = np.sum(F**2, axis = 1)
	fc2 = F**2
	fc2 = np.reshape(fc2,(len(fc2),1))         # Reshape

	# Accuracy of regularization parameter
	h_tol  = 0.2/(np.sqrt(n))

	# Heuristic upper bound on regularization parameter
	varfrac = d.cumsum(axis=0)/d.sum(axis=0)

	if minvarfrac > varfrac.min():
		func = interp1d(varfrac, d)
		d_max = func(minvarfrac)
		h_max = np.sqrt(d_max)
	else:
		h_max = np.sqrt(d.max()) / h_tol

	# Heuristic lower bound on regularization parameter

	if trS0 > trSmin:
		h_min = np.finfo(float).eps**0.5
	else:
		rtsvd = np.zeros((r, 1))
		rtsvd[r-1] = trS0

		for j in range(r-2,0,-1):
			rtsvd[j] = rtsvd[j+1] + fc2[j+1]

		rmin = (abs(rtsvd - trSmin)).argmin()
		h_min = (max(d[rmin], d.min()/n))** 0.5

	# find minimizer of GCV function
	if h_min < h_max:
		fun = lambda x: gcvfctn(x,d[0:r], fc2[0:r], trS0, n-r)
		soln = minimize_scalar(fun, bounds = (h_min, h_max), method='bounded')
		h_opt  = soln['x']
	else:
		print("Upper bound on regularization parameter smaller than lower bound.")
		h_opt  = h_min

	return h_opt

#####

def peigs (A, rmax):
	'''
	'''

	# check number of input arguments
	[m,n] = A.shape

	if rmax > min(m,n):
		rmax  = min(m,n)  # rank cannot exceed size of A

	if rmax < min(m, n)/10.0:
		d,V = eigs(A, rmax)
	else:
		d,V = np.linalg.eig(A)

	# ensure eigenvalues are monotonically decreasing
	I = np.argsort(d)
	I = I [::-1]
	d = d[I]
	V = V[:,I]

	# estimate number of positive eigenvalues of A
	d_min = d.max() * max(m,n) * np.finfo(float).eps

	r = (d > d_min).sum()

	# discard eigenpairs with eigenvalues that are close to or less than zero
	d = d[0:r]
	V = V[:,0:r]                # V = V(:, 1:r)

	return [V,d,r]

#####

def iridge(C, xind, yind, dof, relvar_res = 5e-2):
	'''
	'''
	Cxx = C[np.ix_(xind, xind)]
	Cyy = C[np.ix_(yind, yind)]
	Cxy = C[np.ix_(xind, yind)]

	px = Cxx.shape[0]
	py = Cyy.shape[0]

	# eigendecomposition of Cxx
	rmax = min(dof, px)     # maximum possible rank of Cxx

	[V, d, r] = peigs(Cxx, rmax)
	V = V.real # In case Cxx is not perfectly symmetric numerically
	d = d.real

	# Fourier coefficients
	Vt = V.conj().T
	F = (np.tile((1/(d**0.5)).reshape(len(d),1),px) * Vt).dot(Cxy)

	if dof > r:
		S0  = Cyy - F.conj().T.dot(F)
	else:
		S0  = np.zeros((py, py))

	# approximate minimum squared residual
	trSmin = (relvar_res)* (np.diag(Cyy))

	# initialize output
	h   = np.zeros((py, 1))
	B   = np.zeros((px, py))

	S  = np.zeros((py, py))
	peff = np.zeros((py, 1))

	for k in range (py):
		
		h[k] = gcvridge(F[:,k], d, S0[k,k], dof, r, trSmin[k], relvar_res)
		# k-th column of matrix of regression coefficients
		B[:,k]  = V.dot( (d**0.5 / (d + h[k]**2) * F[:,k]))

		# if S_out     assemble estimate of covariance matrix of residuals
		for j in range(k+1): # k starts at 0
			diagS  = ( h[j]**2 * h[k]**2 ) / ( (d + h[j]**2) * (d + h[k]**2))            
			S[j,k] = S0[j,k] + (F[:,j].conj().T).dot(diagS * F[:,k])
			S[k,j] = S[j,k]
		# if peff_out  effective number of adjusted parameters in this column
		# of B: peff = trace(Mxx_h Cxx)
		peff[k]  = sum(d / (d + h[k]**2))

	return [B,S,h,peff]


####

# Testing
def test():
	C = np.array([[1,2,3],[4,5,6],[7,8,9]])
	print(iridge(C, [0,1], [2], 2))
#[array([[ 0.24669516],[ 0.6739837 ]]), array([[ 2.76176439]]), array([[ 1.68022783]]), array([[ 0.69601762]])]
'''
>> [B,S,h,peff]=iridge(Caa, Cmm, Cam,2,[])

B =

    0.2467
    0.6740


S =

    2.7618


h =

    1.6802


peff =

    0.6960

'''

#    A = np.array([[1,2,3,4,5],[6,7,8,9,0],[1,3,5,7,9],[9,7,4,2,3],[1,2,3,4,5]])
#    print(iridge(A,[0,1,2],[3,4],2))

#[array([[ 0.00403285,  0.00353936],[ 0.02064954, -0.00077429],[ 0.00474236,  0.00794145]]), 
#      array([[ 25.16699546, -22.16105325],[-22.16105325,  50.74469669]]), 
#      array([[ 24.77492727], [ 24.78841235]]), 
#      array([[ 0.02078152], [ 0.02075934]])]

'''
>> [B,S,h,peff]=iridge(Caa, Cmm, Cam,2,[])

B =

    0.0040    0.0035
    0.0206   -0.0008
    0.0047    0.0079


S =

   25.1676  -22.1609
  -22.1609   50.7449


h =

   24.7914  # 24.7749
   24.8003  # 24.7884


peff =

    0.0208
    0.0207

'''
