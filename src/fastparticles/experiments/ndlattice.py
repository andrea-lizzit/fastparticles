from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm, trange

output_dir = Path('out/multidimensional/')

wignerdyson = lambda x, p: np.pi/2 * x * np.exp(-np.pi/4 * np.square(x))

def mdlattice(d, n, W, t, w0=10, torus=True):
	"""
	Generate a matrix representation of a d-dimensional lattice.
	"""
	N = n**d
	rng = np.random.default_rng()
	m = np.diag(w0 + rng.uniform(low=-0.5*W, high=0.5*W, size=N))
	
	# connect across each dimension
	for dim in range(d):
		# connect qubit i to qubit i+n**d and i-n**d
		m += np.diag(np.ones(N-n**dim)*t, n**dim)
		m += np.diag(np.ones(N-n**dim)*t, -n**dim)
		if torus:
			m += np.diag(np.ones(n**dim)*t, N-n**dim)
			m += np.diag(np.ones(n**dim)*t, -N+n**dim)
	return m

def second_excitation_energies(lattice, K):
	"""
	Compute the second excitation energies of a lattice of oscillators.
	"""
	# get the eigenvalues and eigenvectors
	w, V = LA.eigh(lattice)
	# eigenvectors in v[:, i]
	# compute the second excitation energies
	N = len(w)
	energies = np.empty(N*(N+1)//2)
	# compute the quartic perturbation term
	# refer to eq. 14 of report 1
	W = np.square(np.abs(V)) # intermediate matrix
	De_quartic = 4*K*np.matmul(W, W.T)
	out_idx = 0
	for i in range(N):
		for j in range(i, N):
			e_unperturbed = w[i] + w[j]
			De_quadratic = (1+K)*e_unperturbed
			energies[out_idx] = e_unperturbed + De_quadratic + De_quartic[i, j]
			out_idx += 1
	return sorted(energies)

def second_excitation_energies_spins(lattice):
	""" Compute the second excitation energies of a lattice of spins. """
	N = len(lattice)
	energies = np.empty(N*(N-1)//2 + N)
	idx = 0
	for i in range(N):
		for j in range(i, N):
			# construct the state as tensor product of the states of the systems
			# i.e. exciting spins i and j
			state = np.zeros(N)
			state[i] = 1
			state[j] = 1
			# compute the energy of the state
			energy = np.matmul(state, np.matmul(lattice, state))
			energies[idx] = energy
			idx += 1
	return sorted(energies)

if __name__ == '__main__':
	import matplotlib
	matplotlib.use("pgf")
	matplotlib.rcParams.update(
		{
			"pgf.texsystem": "pdflatex",
			"font.family": "serif",
			"text.usetex": True,
			"pgf.rcfonts": False,
			"axes.unicode_minus": False,
		}
	)

	font = {'family' : 'sans',
			'weight' : 'normal',
			'size'   : 10}
	matplotlib.rc('font', **font)

	n_it = 10
	e_l = []
	s_l = []
	d, n, w0 = 3, 17, 100
	W, t = 20, 1
	N = n**d
	K = 0
	EXCITATIONS = 1
	TYPE = 'oscillators'
	for it in trange(n_it):
		if EXCITATIONS == 1:
			m = mdlattice(d, n, W, t, w0)
			ev = np.linalg.eigvalsh(m)
			e_l.append(ev)
			s_l.append(s:=np.diff(ev))
			if np.any(s < 0):
				raise RuntimeError("Negative spacing found! Eigenenergies were not sorted.")
		elif EXCITATIONS == 2:
			m = mdlattice(d, n, W, t)
			if TYPE == 'oscillators':
				e_l.append(second_excitation_energies(m, K))
				s_l.append(s:=np.diff(second_excitation_energies(m, K)))
			elif TYPE == 'spins':
				e_l.append(second_excitation_energies_spins(m))
				s_l.append(s:=np.diff(second_excitation_energies_spins(m)))
			else:
				raise ValueError("Invalid TYPE")
		else:
			raise ValueError("Invalid EXCITATIONS")


	energies = np.concatenate(e_l)
	spacings = np.concatenate(s_l)
	s_scaled = spacings / np.mean(spacings)
	heights, borders, _ = plt.hist(s_scaled.flat, bins=100, density=True, range=(0, 5), histtype='step')
	centers = borders[:-1] + np.diff(borders) / 2
	plt.xlabel(r"s")
	plt.ylabel("density")
	# put a legend with the values of the parameters
	plt.legend([f"rc1; n={n}, W={W}, t={t}, K={K}\ntype={TYPE}, excitations={EXCITATIONS}"])
	plt.savefig(output_dir / f"rc1-{EXCITATIONS}spacings_{d}_{n}x{n}_W{W}_t{t}_k{K}_{TYPE}.png")
	plt.savefig(output_dir / f"rc1-{EXCITATIONS}spacings_{d}_{n}x{n}_W{W}_t{t}_k{K}_{TYPE}.pgf")
	# plt.show()

	# m = mdlattice(d, n, W)
	# w, v = LA.eigh(m)
	# phi = v[:, np.floor(len(v)*0.1).astype(np.int32)].reshape((n,)*d)
	# # plot the 3d function phi as a scatterplot
	# x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(x, y, z, s=phi**2*5e3, c=phi, cmap='viridis', linewidth=0.5)
	# plt.show()
