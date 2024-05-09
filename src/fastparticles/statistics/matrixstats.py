import cupy as cp
import numpy as np
from typing import Union
from fastparticles.operators import Operator
from tqdm import trange

class MatrixStats:
    def __init__(self, sampler: Operator):
        self.sampler = sampler
        self.eigenvalues_ = []
        self.eigenvectors_ = []

    @classmethod
    def from_data(cls, eigenvalues, eigenvectors=None):
        stats = cls(None)
        stats.eigenvalues_ = np.reshape(eigenvalues, (1, 1, -1))
        stats.eigenvalues_ = np.sort(stats.eigenvalues_, axis=2)
        if eigenvectors is not None:
            raise NotImplementedError("eigenvectors not implemented")
        return stats

    def collect(self, n_points : Union[int, None] = None, n_realizations=None, eigenvectors=False):
        """ Collect samples from the ensemble """
        if n_realizations is not None and n_points is not None:
            raise ValueError("Only one of n_points and n_realizations can be specified")
        if n_points is None:
            # default to sample size used by Shklovskii, Shapiro
            # https://doi.org/10.1103/PhysRevB.47.11487
            n_points = 10**5
        if n_realizations:
            n_samples = n_realizations
        else:
            n_samples = (n_points + self.sampler.size-2) // (self.sampler.size-1)
        for _ in trange(n_samples):
            if eigenvectors:
                w, v = cp.linalg.eigh(self.sampler.sample())
                self.eigenvalues_.append(w)
                self.eigenvectors_.append(v)
            else:
                self.eigenvalues_.append(cp.linalg.eigvalsh(self.sampler.sample()))

    def eigenvalues(self):
        """ Calculate the eigenvalue distribution of the matrices sampled from the ensemble with collect() """
        if type(self.eigenvalues_) == list:
            if type(self.eigenvalues_[0]) == np.ndarray:
                return np.array(self.eigenvalues_)
            elif type(self.eigenvalues_[0]) == cp.ndarray:
                return cp.asnumpy(cp.array(self.eigenvalues_))
        elif type(self.eigenvalues_) == cp.ndarray:
            return cp.asnumpy(self.eigenvalues_)
        return self.eigenvalues_

    def eigenvectors(self):
        if len(self.eigenvectors_) != len(self.eigenvalues_):
            raise ValueError("No eigenvectors collected")
        if type(self.eigenvectors_) == list and type(self.eigenvectors_[0]) == np.ndarray:
            return np.stack(self.eigenvectors_)
        if type(self.eigenvectors_) == list and type(self.eigenvectors_[0]) == cp.ndarray:
            return cp.asnumpy(cp.stack(self.eigenvectors_))
        return self.eigenvectors_
    
    def spacings(self, selector=None):
        """ Calculate the spacing distribution of the matrices sampled from the ensemble with collect() """
        e = self.eigenvalues()
        if selector:
            mask = np.array([selector(sample) for sample in e])
        else:
            mask = np.ones(e.shape, dtype=bool)
        mask = mask[0]
        res = np.diff(e[:, mask], axis=1)
        if np.any(res < 0):
            print("Warning: the eigenvalues are not sorted. This might also mean that there are degenerate eigenvalues, when they are not expected.")
            raise ValueError("Eigenvalues are not sorted.")
        return res

    # @cache
    def s(self, selector=None):
        spacings_ = self.spacings(selector)
        return spacings_ / np.mean(spacings_, axis=1, keepdims=True)
    
    # @cache
    def d2correlations(self, selector=None):
        """ Calculate the correlations between adjacent spacings
            as in Oganesyan and Huse
            https://doi.org/10.1103/PhysRevB.75.155111 """
        delta = self.spacings(selector)
        dn, dnm1 = delta[:, :-1], delta[:, 1:]
        r = np.minimum(dn, dnm1) / np.maximum(dn, dnm1)
        return r

class XXZ(MatrixStats):
    def __init__(self, phi, J1, J2, systemspec):
        self.systemspec = systemspec
        self.phi = phi
        self.J1 = J1
        self.J2 = J2
        self.eigenvalues_ = []
        self.eigenvectors_ = []

    def matrix_(self):
        mat = cp.zeros((self.systemspec.N, self.systemspec.N), dtype=cp.complex64)
        phase = np.exp(1j*self.phi/self.systemspec.n)
        for i in range(self.systemspec.n-1):
            mat += 1/2 * phase * operators.exchange(i+1, i, self.systemspec)
            mat += 1/2 * np.conj(phase) * operators.exchange(i, i+1, self.systemspec)
            mat += self.J1 * operators.Z(i, self.systemspec) * operators.Z(i+1, self.systemspec)
        if self.J2 != 0:
            for i in range(self.systemspec.n-2):
                mat += self.J2 * operators.Z(i, self.systemspec) * operators.Z(i+2, self.systemspec)
        return mat

    def sample(self):
        raise RuntimeError("This matrix is not stochastic, cannot sample.")

    def collect(self, eigenvectors=False):
        """ Collect samples from the ensemble """
        mat = self.matrix_()
        if eigenvectors:
            raise NotImplementedError("eigenvectors not implemented due to issues with sorting")
            w, v = cp.linalg.eigh(mat)
            self.eigenvalues_.append(w)
            self.eigenvectors_.append(v)
        else:
            self.eigenvalues_.append(cp.sort(cp.linalg.eigvalsh(mat)))


def PalHuseSelector(eigenvalues, min_size=10):
    """ Select eigenvalues in the middle third of the spectrum,
         as done by Pal and Huse in 10.1103/PhysRevB.82.174411 """

    idx = np.arange(eigenvalues.size)
    l = eigenvalues.size - 1
    start, end = l/3, 2*l/3
    start, end = min(start, (l - min_size)/2), max(end, (l + min_size)/2)
    mask = (idx > start) & (idx < 2*end)    
    return mask