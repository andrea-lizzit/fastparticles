from tqdm import trange
from typing import Union
import cupy as cp
from .pcoperators import BPCOperator, BExchange, B4Operator, FExchange, FPCOperator
from .operators import Operator, ScalarOperator
from fastparticles.hilbert.pcspace import BosonHilbertSpace, FermionHilbertSpace
import numpy as np
from functools import reduce

rng = cp.random.default_rng()

class BosonChainSampler(BPCOperator):
    """Simulates a chain of bosons with diagonal interaction. 
    
    :param W: disorder strength
    :type W: float
    :param t: hopping strength
    :type t: float
    :param K: interaction strength
    :type K: float, optional
    :param w0: on-site energy, defaults to 100
    :type w0: float, optional
    """

    def __init__(self, hs: BosonHilbertSpace, W, t, K=0, w0=100, rng=np.random.default_rng()):
        """Constructor method
        """
        super().__init__(hs)
        self.n = hs.N
        self.W = W
        self.t = t
        self.w0 = w0
        self.e = hs.e
        self.K = K
        self.t_part = None
        self.K_part = None
        self.rng = rng if rng else np.random.default_rng()

    def matrix(self):
        w = self.rng.standard_normal(size=(self.n,)) * self.W
        m = self._t_part()
        for i in range(self.n):
            m += (self.w0 + w[i]).item() * BExchange(self.hs, i, i)
        self.m = m + self.t * self._t_part() + self.K * self._K_part()
        return self.m.matrix()
    
    def _t_part(self):
        if self.t_part:
            return self.t_part
        self.t_part = reduce(lambda a,b: a+b, [BExchange(self.hs, i, (i+1)%self.n) + BExchange(self.hs, (i+1)%self.n, i) for i in range(self.n)])
        return self.t_part
    
    def _K_part(self):
        if self.K_part:
            return self.K_part
        self.K_part = reduce(lambda a,b: a+b, [B4Operator(self.hs, i) for i in range(self.n)])
        return self.K_part
    
    def __call__(self, psi, phi):
        print("Untested code")
        i, j = self.hs.index(psi), self.hs.index(phi)
        s1, s2 = cp.zeros((self.hs.dim, self.hs.dim), dtype=cp.complex64), cp.zeros((self.hs.dim, self.hs.dim), dtype=cp.complex64)
        s1[i] = 1
        s2[j] = 1
        return cp.dot(cp.dot(s1, self.m), s2)
        

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


class OganesyanHuseSampler(FPCOperator):
    """
    Generate a matrix representation of a 1-dimensional lattice with nearest- and second-neighbour hopping,
    as described in 10.1103/PhysRevB.75.155111
    """
    def __init__(self, hs: FermionHilbertSpace, W, t=1, V=2):
        super().__init__(hs)
        self.d = 1
        self.n = hs.N
        self.W = W
        self.t = t
        self.V =  V
        self.e = hs.e
        self.hs = hs

    def matrix(self):
        """ Sample a matrix from the ensemble. """
        w = rng.standard_normal(size=(self.n,)) * self.W
        mat = ScalarOperator(self.hs, 0)
        for i in range(self.hs.N):
            i1 = (i+1) % self.hs.N
            i2 = (i+2) % self.hs.N

            ni = FExchange(self.hs, i, i)
            ni1 = FExchange(self.hs, i1, i1)
            print(-1 * FExchange(self.hs, i, i))
            mat += w[i].item() * ni
            mat += self.V * (ni - 0.5) * (ni1 - 0.5)
            mat += FExchange(self.hs, i, i1) + FExchange(self.hs, i1, i) + FExchange(self.hs, i, i2) + FExchange(self.hs, i2, i)
        return mat.matrix()

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
