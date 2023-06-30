import itertools
import numpy as np
import cupy as cp
import scipy
from tqdm import tqdm

mempool = cp.get_default_memory_pool()

class HarmonicOscillator:
    def __call__(self, excitations):
        return excitations


class ManyBodyBosons:
    """ Study the many body level statistics of the system. """

    def __init__(self, n_excitations=None):
        self.n_excitations = n_excitations

    def __call__(self, stats):
        e, V = stats.eigenvalues(), stats.eigenvectors()
        K = self.n_excitations
        if K is None:
            # Default to a half filled system
            # as in Oganesyan and Huse,
            # https://doi.org/10.1103/PhysRevB.75.155111
            K = len(e[0])//2
        N = e.shape[1]
        # Consider all possible nex-particle excitations
        # Their number is the binomial coefficient (K+N-1, K)
        num_comb = scipy.special.comb(N+K-1, K, exact=True)
        n_e_energies = np.zeros((e.shape[0], num_comb))
        for i, combination in enumerate(tqdm(itertools.combinations(np.arange(N+K-1), N-1), total=num_comb)):
            # Convert the combination to the number of excitations in each mode
            assert len(combination) == N-1
            c = [-1] + list(combination) + [K+N-1]
            b = np.diff(c) - 1
            # Calculate the energy of the excited state
            E = self.get_energy(e, b, V)
            n_e_energies[:, i] = E
        mbstats = stats.__class__(stats.sampler)
        free_bytes = mempool.free_bytes()
        if n_e_energies.nbytes < free_bytes:
            mbstats.eigenvalues_ = cp.array(n_e_energies)
            mbstats.eigenvalues_.sort()
            mbstats.eigenvalues_ = mbstats.eigenvalues_.get()
        else:
            print("Warning: not enough memory on GPU to sort the eigenvalues. Falling back to CPU.")
            mbstats.eigenvalues_ = n_e_energies
            mbstats.eigenvalues_.sort()
        return mbstats
 

class ManyBodyLevels(ManyBodyBosons):
    """ Study the many body level statistics of the system. """

    def __init__(self, n_excitations=None, K=0):
        self.n_excitations = n_excitations
        self.K = K

    def get_energy(self, e, b, V=None):
        if self.K != 0 and V is None:
            raise ValueError("V must be specified when K is not zero")
        N = e.shape[1]
        # Calculate the energy of the excited state
        E = np.sum(e * b, axis=1)
        # add perturbation
        # E += self.K * np.sum(e * (b*(b-1)), axis=1) # this perturbation is wrong, because it is in the b basis instead of the a basis
        E += self.K * np.sum((V**4) @ (b*(b-1)), axis=1)
        Vi2 = np.sum(V**2, axis=1)
        for (alpha, beta) in itertools.combinations(range(N), 2):
            E += self.K * 4 * Vi2[:, alpha] * Vi2[:, beta] * b[alpha] * b[beta]
        return E
    

class CrossDiagonal(ManyBodyBosons):
    """ Study the many body level statistics of the system. """

    def __init__(self, n_excitations=None, K=0):
        self.n_excitations = n_excitations
        self.K = K

    def get_energy(self, e, b, V=None):
        if self.K != 0 and V is None:
            raise ValueError("V must be specified when K is not zero")
        N = e.shape[1]
        # Calculate the energy of the excited state
        E = np.sum(e * b, axis=1)
        # add perturbation
        # check if there is space on GPU
        free_bytes = mempool.free_bytes()
        if self.K != 0 and V.nbytes + b.nbytes < free_bytes:
            cV, cb = cp.asarray(V), cp.asarray(b)
            E += self.K * cp.einsum('ria,rja,rka,rla,a->r', cV, cV, cV, cV, cb*(cb-1) - cb**2).get()
            E += self.K * 2 * cp.einsum('ria,rjb,rka,rlb,a,b->r', cV, cV, cV, cV, cb, cb).get()
            return E
        print("Warning: not enough memory on GPU to calculate the perturbation. Falling back to CPU.")
        E += self.K * np.einsum('ria,rja,rka,rla,a->r', V, V, V, V, b*(b-1) - b**2)
        E += self.K * 2 * np.einsum('ria,rjb,rka,rlb,a,b->r', V, V, V, V, b, b)
        return E
    

class RandomCrossDiagonal(ManyBodyBosons):
    """ Study the many body level statistics of the system. """

    def __init__(self, n_excitations=None, K=0):
        self.n_excitations = n_excitations
        self.K = K

    def get_energy(self, e, b, V=None):
        if self.K != 0 and V is None:
            raise ValueError("V must be specified when K is not zero")
        N = e.shape[1]
        # Calculate the energy of the excited state
        E = np.sum(e * b, axis=1)
        # add perturbation
        # check if there is space on GPU
        free_bytes = mempool.free_bytes()
        if self.K != 0 and V.nbytes + b.nbytes < free_bytes:
            cV, cb = cp.asarray(V), cp.asarray(b)
            cR = cp.random.rand(e.shape[0], N, N, N, N)
            E += self.K * cp.einsum('rijkl, ria,rja,rka,rla,a->r', cR, cV, cV, cV, cV, cb*(cb-1) - cb**2).get()
            E += self.K * 2 * cp.einsum('rijkl, ria,rjb,rka,rlb,a,b->r', cR, cV, cV, cV, cV, cb, cb).get()
            return E
        print("Warning: not enough memory on GPU to calculate the perturbation. Falling back to CPU.")
        R = np.random.rand(N, N, N, N)
        E += self.K * np.einsum('ijkl,ria,rja,rka,rla,a->r', R, V, V, V, V, b*(b-1) - b**2)
        E += self.K * 2 * np.einsum('ijkl,ria,rjb,rka,rlb,a,b->r', R, V, V, V, V, b, b)
        return E
