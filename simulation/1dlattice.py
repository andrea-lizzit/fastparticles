import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal
from tqdm import trange

n = 1000
nrep = 10
beta = 2
window = np.arange(n//4, 3*n//4)
ds = np.zeros((nrep, len(window)-1))
W = 0.1

for ii in trange(nrep):
	l = eigvalsh_tridiagonal(W*np.random.uniform(low=-0.5, high=0.5, size=n), np.ones(n-1))
	# l = eigvalsh_tridiagonal(np.random.randn(n), np.sqrt(np.random.chisquare(np.arange(n-1, 0, -1)*beta)/2))
	l_center = l[window]
	d = np.diff(l_center)/beta/np.pi*np.sqrt(2*beta*n-l_center[:-1]**2)
	ds[ii, :] = d

# plot histogram
# bins start from 0, end at 2.5, and step size is 0.05
plt.hist(ds.flat, bins=np.arange(0, 0.6, 0.005), density=True)
plt.show()

w, v = eigh_tridiagonal(W*np.random.uniform(low=-0.5, high=0.5, size=n), np.ones(n-1))
plt.plot(v[-3, :])
plt.show()