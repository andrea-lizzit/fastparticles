import itertools
import numpy as np
import scipy
from tqdm import tqdm

class ManyBodyLevels:
    """ Study the many body level statistics of the system. """
    def __init__(self, n_excitations=None):
        self.n_excitations = n_excitations
    def __call__(self, stats):
        e = stats.eigenvalues()
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
            E = np.sum(e * b, axis=1)
            n_e_energies[:, i] = E
        mbstats = stats.__class__(stats.sampler)
        mbstats.eigenvalues_ = np.sort(n_e_energies, axis=1)
        return mbstats