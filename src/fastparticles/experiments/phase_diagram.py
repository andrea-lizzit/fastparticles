import matplotlib.pyplot as plt
import numpy as np
import scipy
import pyamg
from ndlattice import mdlattice
from tqdm import tqdm

rng = np.random.default_rng()

def IPR(psi):
	"""
	calculate the inverse participation ratio
	"""
	return np.sum(np.abs(psi)**4)

def solve(E, lattice):
	N = lattice.shape[0]
	A = lattice - np.diag(E*np.ones(N))
	b = np.zeros(N)
	x = scipy.linalg.solve(A, b) # could use solveh_banded
	print(np.max(x), np.min(x))
	return IPR(x)

def solve_multigrid(E, lattice):
	N = lattice.shape[0]
	d = len(lattice.shape)
	A = scipy.sparse.csr_array(lattice - np.diag(E*np.ones(N)))
	b = np.zeros(N)
	x = pyamg.solve(A, b)
	return IPR(x)
	# convert the matrix to a Poisson equation
	# Agg = pyamg.aggregation.standard_aggregation(A)
	# B = np.ones((N, 1))
	# n = int(np.cbrt(N))
	# stencil = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
	# A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
	ml = pyamg.smoothed_aggregation_solver(A, max_coarse=100)
	print(ml)

	residuals = []
	b = np.random.rand(A.shape[0])
	x0 = rng.uniform(size=A.shape[0])

	x = ml.solve(b=b, x0=x0, tol=1e-10, residuals=residuals)
	return IPR(x)


# m = mdlattice(d, n, W, t)

d, n, t = 3, 17, 1
# calculate the IPR for a grid of E and W
E = np.linspace(-10, 10, 10)
W = np.linspace(0, 20, 10)
IPRgrid = np.zeros((len(E), len(W)))

tbar = tqdm(total=len(E)*len(W))
for i, e in enumerate(E):
	for j, w in enumerate(W):
		m = mdlattice(d, n, w, t)
		IPRgrid[i, j] = solve(e, m)
		tbar.update(1)

np.save('IPRgrid.npy', IPRgrid)

# plot the results
fig, ax = plt.subplots()
im = ax.matshow(IPRgrid, extent=[0, 20, -10, 10])
ax.set_xlabel('W')
ax.set_ylabel('E')
fig.colorbar(im)
plt.show()