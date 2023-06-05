import numpy as np
import cupy as cp # 4.9x speedup
from scipy.linalg import eigvalsh_tridiagonal
from tqdm import trange

rng = np.random.default_rng()


class GOEsampler:
	def __init__(self, N):
		self.N = N

	def sample(self):
		""" Sample a random matrix from the Gaussian Orthogonal Ensemble """
		gausmat = cp.random.normal(size=(self.N, self.N))
		return (gausmat + gausmat.T) / 2
	
	@staticmethod
	def eigenvalues(mat):
		""" Eigenvalues of a matrix, sorted in ascending order """
		return cp.linalg.eigvalsh(mat)

	@property
	def size(self):
		""" Number of eigenvalues """
		return self.N


class LatticeSampler:
	"""
	Generate a matrix representation of a d-dimensional lattice.
	"""
	def __init__(self, d, n, W, t, w0=10, torus=True):
		self.d = d
		self.n = n
		self.W = W
		self.t = t
		self.w0 = w0
		self.torus = torus
	
	def sample(self):
		d, n, t, torus = self.d, self.n, self.t, self.torus
		N = n**d
		m = cp.diag(self.w0 + rng.normal(scale=self.W, size=N))
		
		# connect across each dimension
		for dim in range(d):
			# connect qubit i to qubit i+n**d and i-n**d
			m += cp.diag(cp.ones(N-n**dim)*t, n**dim)
			m += cp.diag(cp.ones(N-n**dim)*t, -n**dim)
			if torus:
				m += cp.diag(cp.ones(n**dim)*t, N-n**dim)
				m += cp.diag(cp.ones(n**dim)*t, -N+n**dim)
		return m
	
	@staticmethod
	def eigenvalues(mat):
		""" Eigenvalues of a matrix, sorted in ascending order """
		return cp.linalg.eigvalsh(mat)

	@property
	def size(self):
		""" Number of eigenvalues """
		return self.n ** self.d
	

class Betasampler:
	""" Sampler for the beta-Hermite ensemble with tridiagonal matrices from Dumitriu, Edelman (2018). """
	def __init__(self, N, beta=1):
		self.N = N
		self.beta = beta
	
	def sample(self):
		""" Sample a random matrix from the beta-Hermite ensemble """
		return np.random.randn(self.N),\
				np.random.chisquare(np.arange(self.N-1, 0, -1)*self.beta)/np.sqrt(2)
	
	@staticmethod
	def eigenvalues(mat):
		""" Eigenvalues of a matrix, sorted in ascending order """
		return eigvalsh_tridiagonal(mat[0], mat[1])

	@property
	def size(self):
		""" Number of eigenvalues """
		return self.N


class MatrixStats:
	def __init__(self, sampler):
		self.sampler = sampler
		self.eigenvalues_ = []

	def collect(self, n_points=None):
		""" Collect samples from the ensemble """
		if n_points is None:
			# default to sample size used by Shklovskii, Shapiro
			# https://doi.org/10.1103/PhysRevB.47.11487
			n_points = 10**5
		n_samples = max(n_points // self.sampler.size, 1)
		for i in trange(n_samples):
			self.eigenvalues_.append(self.sampler.eigenvalues((self.sampler.sample())))

	def eigenvalues(self):
		""" Calculate the eigenvalue distribution of the matrices sampled from the ensemble with collect() """
		return cp.array(self.eigenvalues_).get()
	
	def spacings(self, selector=None):
		""" Calculate the spacing distribution of the matrices sampled from the ensemble with collect() """
		e = self.eigenvalues()
		if selector:
			mask = np.array([selector(sample) for sample in e])
		else:
			mask = np.ones(e.shape, dtype=bool)
		mask = mask[0]
		return np.diff(e[:, mask], axis=1)

	def s(self, selector=None):
		spacings_ = self.spacings(selector)
		return spacings_ / np.mean(spacings_)
	
	def d2correlations(self, selector=None):
		""" Calculate the correlations between adjacent spacings
			as in Oganesyan and Huse
			https://doi.org/10.1103/PhysRevB.75.155111 """
		delta = self.spacings(selector)
		dn, dnm1 = delta[:, :-1], delta[:, 1:]
		r = np.minimum(dn, dnm1) / np.maximum(dn, dnm1)
		return r
	
def PalHuseSelector(eigenvalues):
	""" Select eigenvalues in the middle third of the spectrum,
	 	as done by Pal and Huse in 10.1103/PhysRevB.82.174411 """
	idx = np.arange(eigenvalues.size)
	l = eigenvalues.size
	mask = (idx > l/3) & (idx < 2*l/3)
	return mask