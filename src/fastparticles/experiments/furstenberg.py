import math

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

rng = np.random.default_rng()
ENABLE_TRANSFER_MATRIX = False
ENABLE_CHECK = False

def psin_transfer_matrix(E, e, psi0, psi1, all=False):
	T = np.eye(2)
	for i in range(len(e)):
		Ti = np.array([[E-W*e[i], -1], [1, 0]])
		T = np.matmul(Ti, T)
		if all:
			yield np.matmul(T, np.array([psi0, psi1]))[0]
	if not all:
		return np.matmul(T, np.array([psi0, psi1]))[0]

def psin_direct(E, e, psi0, psi1, all=False):
	psim1, psim2 = psi1, psi0
	for ee in e:
		psim1, psim2 = (E-W*ee)*psim1 - psim2, psim1
		if all:
			yield psim1
	if not all:
		return psim1

def psin(E, e, psi0, psi1, all=False):
	if ENABLE_CHECK and all:
		raise ValueError("Invalid combination of flags. (ENABLE_CHECK and all")
	if ENABLE_CHECK:
		res = psin_transfer_matrix(E, e, psi0, psi1)
		res2 = psin_direct(E, e, psi0, psi1)
		if math.fabs(res - res2) > 1e-6:
			print("Error: ", res, res2)
			raise RuntimeError()
		return res
	if ENABLE_TRANSFER_MATRIX:
		res = psin_transfer_matrix(E, e, psi0, psi1)
	else:
		res = psin_direct(E, e, psi0, psi1)
	return res

W = .15
k = np.pi*0.06
n = 10000
E = 2*np.cos(k)
e = W*rng.uniform(low=-0.5, high=0.5, size=n)
psi = np.array(list(psin_direct(E, e, 1, np.exp(1j*k), all=True)))
plt.plot(np.real(psi))
plt.plot(np.imag(psi))
plt.show()

# psisum = 0
# for i in trange(100):
# 	e = rng.uniform(low=-0.5, high=0.5, size=100000)
# 	psi0 = 1
# 	psi1 = 1
# 	psisum += psin(0, e, psi0, psi1)
# psisum /= 100
# print(psisum)