import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
import matplotlib.pyplot as plt
from tqdm import trange

n = 50
nrep = 10000
beta = 2
ds = np.zeros((nrep, n//2-1))

for ii in trange(nrep):
	l = eigvalsh_tridiagonal(np.random.randn(n), np.random.normal(size=n-1))
	# l = eigvalsh_tridiagonal(np.random.randn(n), np.sqrt(np.random.chisquare(np.arange(n-1, 0, -1)*beta)/2))
	l_center = l[n//4:3*n//4]
	d = np.diff(l_center)/beta/np.pi*np.sqrt(2*beta*n-l_center[:-1]**2)
	ds[ii, :] = d

# plot histogram
# bins start from 0, end at 2.5, and step size is 0.05
plt.hist(ds.flat, bins=np.arange(0, 1, 0.02), density=True)
plt.show()