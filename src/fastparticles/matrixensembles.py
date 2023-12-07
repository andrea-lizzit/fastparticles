import numpy as np
import cupy as cp # 4.9x speedup
from scipy.linalg import eigvalsh_tridiagonal
from tqdm import trange
from .studies import ManyBodyLevels
from .operators import operators
from typing import Union
from functools import cache
import math


rng = np.random.default_rng()


class GOEsampler:
    def __init__(self, N):
        self.N = N

    def sample(self):
        """ Sample a random matrix from the Gaussian Orthogonal Ensemble """
        gausmat = cp.random.normal(size=(self.N, self.N))
        return (gausmat + gausmat.T) / 2
    
    def eigenvalues(self, mat):
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
    
    def eigenvalues(self, mat):
        """ Eigenvalues of a matrix, sorted in ascending order """
        return cp.linalg.eigvalsh(mat)

    def eig(self, mat):
        return cp.linalg.eigh(mat)

    @property
    def size(self):
        """ Number of eigenvalues """
        return self.n ** self.d


class OganesyanHuseSampler:
    """
    Generate a matrix representation of a 1-dimensional lattice with nearest- and second-neighbour hopping,
    as described in 10.1103/PhysRevB.75.155111
    """
    def __init__(self, n, W, t=1, V=2, e=None):
        self.d = 1
        self.n = n
        self.W = W
        self.t = t
        self.V =  V
        if e:
            self.e = e
        else:
            self.e = n // 2

    def sample(self):
        """ Sample a matrix from the ensemble. """
        w = rng.normal(scale=self.W, size=(self.n,))
        systemspec = operators.FermionSystemSpec(self.n, self.e)
        mat = cp.zeros((systemspec.N, systemspec.N))
        for i in range(systemspec.n):
            i1 = (i+1) % systemspec.n
            i2 = (i+2) % systemspec.n

            ni = operators.exchange(i, i, systemspec)
            ni1 = operators.exchange(i1, i1, systemspec)
            I = cp.identity(systemspec.N)
            mat += w[i] * ni
            mat += self.V * (ni - I*0.5) * (ni1 - I*0.5)
            mat += operators.exchange(i, i1, systemspec) + operators.exchange(i1, i, systemspec) + operators.exchange(i, i2, systemspec) + operators.exchange(i2, i, systemspec)
        return mat
    
    def eigenvalues(self, mat):
        """ Eigenvalues of a matrix, sorted in ascending order """
        return cp.sort(cp.linalg.eigvalsh(mat))
    
    @property
    def size(self):
        """ Number of eigenvalues """
        return operators.SystemSpec(self.n, self.e).N

class BosonOHSampler:
    """
    Generate a matrix representation of a 1-dimensional lattice with nearest- and second-neighbour hopping,
    as described in 10.1103/PhysRevB.75.155111, but with bosons
    """
    def __init__(self, n, W, t=1, V=2, e=None):
        self.d = 1
        self.n = n
        self.W = W
        self.t = t
        self.V =  V
        if e:
            self.e = e
        else:
            self.e = n // 2

    def sample(self):
        """ Sample a matrix from the ensemble. """
        w = rng.normal(scale=self.W, size=(self.n,))
        systemspec = operators.BosonSystemSpec(self.n, self.e)
        mat = cp.zeros((systemspec.N, systemspec.N))
        for i in range(systemspec.n):
            i1 = (i+1) % systemspec.n
            i2 = (i+2) % systemspec.n

            ni = operators.boson_exchange(i, i, systemspec)
            ni1 = operators.boson_exchange(i1, i1, systemspec)
            I = cp.identity(systemspec.N)
            mat += w[i] * ni
            mat += self.V * (ni - I*0.5) * (ni1 - I*0.5)
            mat += operators.boson_exchange(i, i1, systemspec) + operators.boson_exchange(i1, i, systemspec) + operators.boson_exchange(i, i2, systemspec) + operators.boson_exchange(i2, i, systemspec)
        return mat
    
    def eigenvalues(self, mat):
        """ Eigenvalues of a matrix, sorted in ascending order """
        return cp.sort(cp.linalg.eigvalsh(mat))
    
    @property
    def size(self):
        """ Number of eigenvalues """
        return operators.BosonSystemSpec(self.n, self.e).N
 
class BosonChainSampler:
    """ Similar to LatticeSampler with d=1 for bosons. The calculation is exact. """
    def __init__(self, n, W, t, K=0, e=None, w0=10, rng=None):
        self.n = n
        self.W = W
        self.t = t
        self.w0 = w0
        self.e = e if e else n//2
        self.K = K
        self.rng = rng if rng else np.random.default_rng()

    def sample(self):
        systemspec = operators.BosonSystemSpec(self.n, self.e)
        w = self.rng.normal(scale=self.W, size=(self.n,))
        m = cp.zeros((systemspec.N, systemspec.N))
        for i in range(self.n):
            i1 = (i+1) % self.n
            m += self.t*operators.boson_exchange(i, i1, systemspec) + self.t*operators.boson_exchange(i1, i, systemspec)
            m += (self.w0 + w[i]) * operators.boson_exchange(i, i, systemspec)
            m += self.K*operators.boson_a4(i, systemspec)
        return m

    def eigenvalues(self, mat):
        return cp.linalg.eigvalsh(mat)

    def eig(self, mat):
        return cp.linalg.eigh(mat)

    @property
    def size(self):
        return operators.BosonSystemSpec(self.n, self.e).N
    
class NNBosonXXZ:
    def __init__(self, n, W, t, J1, J2, e=None, w0=100, rng=None):
        self.n = n
        self.W = W
        self.t = t
        self.w0 = w0
        self.e = e if e else n//2
        self.J1 = J1
        self.J2 = J2
        self.rng = rng if rng else np.random.default_rng()

    def sample(self):
        systemspec = operators.BosonSystemSpec(self.n, self.e)
        w = self.rng.uniform(low=-self.W/2, high=self.W/2, size=(self.n,))
        m = cp.zeros((systemspec.N, systemspec.N))
        for i in range(self.n):
            i1 = (i+1) % self.n
            i2 = (i+2) % self.n
            m += self.t*operators.boson_exchange(i, i1, systemspec) + self.t*operators.boson_exchange(i1, i, systemspec)
            m += (self.w0 + w[i]) * operators.boson_exchange(i, i, systemspec)
            m += self.J1 * operators.exchange(i, i, systemspec) * operators.exchange(i1, i1, systemspec)
            m += self.J2 * operators.exchange(i, i, systemspec) * operators.exchange(i2, i2, systemspec)
        return m

    def eigenvalues(self, mat):
        return cp.linalg.eigvalsh(mat)

    def eig(self, mat):
        return cp.linalg.eigh(mat)

    @property
    def size(self):
        return operators.BosonSystemSpec(self.n, self.e).N

class CrossoverSampler:
    """
    Similar to OganesyanHuseSampler with V=0, but for bosons
    """
    def __init__(self, n, W, t=1, V=2, e=None):
        self.d = 1
        self.n = n
        self.W = W
        self.t = t
        self.V =  V
        if e:
            self.e = e
        else:
            self.e = n // 2

    def sample(self):
        """ Sample a matrix from the ensemble. """
        w = rng.normal(scale=self.W, size=(self.n,))
        mat = cp.diag(w)
        for i in range(self.n):
            i1 = (i+1) % self.n
            i2 = (i+2) % self.n
            mat[i, i1] = self.t
            mat[i1, i] = self.t
            mat[i, i2] = self.t
            mat[i2, i] = self.t
        return mat
    
    def eigenvalues(self, mat):
        """ Eigenvalues of a matrix, sorted in ascending order """
        return cp.sort(cp.linalg.eigvalsh(mat))
    
    @property
    def size(self):
        """ Number of eigenvalues """
        return operators.SystemSpec(self.n, self.e).N



class OODSampler:
    """
    Generate a matrix representation of a 1-dimensional lattice with out of diagonal terms
    """
    def __init__(self, n, W, t=1, V=2, e=None):
        self.d = 1
        self.n = n
        self.W = W
        self.t = t
        self.V =  V
        if e:
            self.e = e
        else:
            self.e = n // 2

    def sample(self):
        """ Sample a matrix from the ensemble. """
        w = rng.normal(scale=self.W, size=(self.n,))
        systemspec = operators.SystemSpec(self.n, self.e)
        mat = cp.zeros((systemspec.N, systemspec.N))
        for i in range(systemspec.n):
            i1 = (i+1) % systemspec.n
            ni = operators.exchange(i, i, systemspec)
            I = cp.identity(systemspec.N)
            mat += w[i] * ni
            mat += self.t * operators.exchange(i, i1, systemspec) + operators.exchange(i1, i, systemspec)
        # add 2 particle out of diagonal terms
        r = self.V * cp.random.randn(systemspec.N, systemspec.N)
        # make it symmetric
        r = (r + r.T) / np.sqrt(2)
        # make the diagonal zero
        r -= cp.diag(cp.diag(r))
        return mat + r
    
    def eigenvalues(self, mat):
        """ Eigenvalues of a matrix, sorted in ascending order """
        return cp.linalg.eigvalsh(mat).sort()
    
    @property
    def size(self):
        """ Number of eigenvalues """
        return operators.SystemSpec(self.n, self.e).N


class Betasampler:
    """ Sampler for the beta-Hermite ensemble with tridiagonal matrices
        from Dumitriu, Edelman https://doi.org/10.1063/1.1507823 """
    def __init__(self, N, beta=1):
        self.N = N
        self.beta = beta
    
    def sample(self):
        """ Sample a random matrix from the beta-Hermite ensemble """
        return np.random.randn(self.N),\
                np.random.chisquare(np.arange(self.N-1, 0, -1)*self.beta)/np.sqrt(2)
    
    def eigenvalues(self, mat):
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
                w, v = self.sampler.eig(self.sampler.sample())
                self.eigenvalues_.append(w)
                self.eigenvectors_.append(v)
            else:
                self.eigenvalues_.append(self.sampler.eigenvalues(self.sampler.sample()))

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
