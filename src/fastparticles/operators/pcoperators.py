import cupy as cp
import math
import itertools
from fastparticles.indexing import indexof, boson_indexof
from .operators import Operator
from fastparticles.hilbert.pcspace import BosonHilbertSpace, FermionHilbertSpace, SpinHilbertSpace
from fastparticles.indexing import BosonSystemSpec, FermionSystemSpec

def exchange(i, j, systemspec):
    """ Representation of the a_i^\dagger*a_j operator in an n-qubit system in the e-particle basis. """
    if systemspec.e > systemspec.n:
        raise ValueError("The number of particles cannot be higher than the number of qubits")
    mat = cp.zeros((systemspec.N, systemspec.N))
    available = list(range(systemspec.n))
    available.remove(i)
    if i != j:
        available.remove(j)
    for index in itertools.combinations(available, systemspec.e-1):
        before = indexof(tuple(sorted(index + (j,))), systemspec.n, systemspec.e)
        after = indexof(tuple(sorted(index + (i,))), systemspec.n, systemspec.e)
        mat[after, before] = 1
    return mat
    
def boson_exchange(i, j, systemspec):
    """ Representation of the a_i^\dagger*a_j operator in an n-oscillator system in the e-particle basis. """
    if systemspec.statistic != "boson":
        raise ValueError("Wrong particle statistic")
    mat = cp.zeros((systemspec.N, systemspec.N))
    available = list(range(systemspec.n))
    for index in itertools.combinations_with_replacement(available, systemspec.e-1):
        before = boson_indexof(tuple(sorted(index + (j,))), systemspec.n, systemspec.e)
        after = boson_indexof(tuple(sorted(index + (i,))), systemspec.n, systemspec.e)
        ni = index.count(i)
        nj = index.count(j)
        mat[after, before] = math.sqrt(ni+1)*math.sqrt(nj+1)
    return mat

def boson_a4(i, systemspec):
    """ Representation of the term a_i^\dagger a_i^\dagger a_i a_i."""
    if systemspec.statistic != "boson":
        raise ValueError("Wrong particle statistic")
    mat = cp.zeros((systemspec.N, systemspec.N))
    if systemspec.e == 1:
        return mat
    available = list(range(systemspec.n))
    for index in itertools.combinations_with_replacement(available, systemspec.e-2):
        state_idx = boson_indexof(tuple(sorted(index + (i,i))), systemspec.n, systemspec.e)
        c = index.count(i) + 2
        mat[state_idx, state_idx] = c * (c-1)
    return mat

class FPCOperator(Operator):
    def __init__(self, hs: FermionHilbertSpace):
        super().__init__(hs)
    def __call__(self, psi, phi):
        raise NotImplementedError("Not implemented yet")

class FExchange(FPCOperator):
    def __init__(self, hs: FermionHilbertSpace, i, j):
        super().__init__(hs)
        self.i = i
        self.j = j
    def matrix(self):
        return exchange(self.i, self.j, FermionSystemSpec(self.hs.N, self.hs.e))

class BPCOperator(Operator):
    def __init__(self, hs: BosonHilbertSpace):
        super().__init__(hs)
    
class BExchange(BPCOperator):
    def __init__(self, hs: BosonHilbertSpace, i, j):
        super().__init__(hs)
        self.i = i
        self.j = j
    def matrix(self):
        return boson_exchange(self.i, self.j, BosonSystemSpec(self.hs.N, self.hs.e))

class B4Operator(BPCOperator):
    def __init__(self, hs: BosonHilbertSpace, i):
        super().__init__(hs)
        self.i = i
    def matrix(self):
        return boson_a4(self.i, BosonSystemSpec(self.hs.N, self.hs.e))

class SPCOperator(Operator):
    def __init__(self, hs: SpinHilbertSpace):
        super().__init__(hs)
    def __call__(self, psi, phi):
        raise NotImplementedError("Not implemented yet")

class SExchange(SPCOperator):
    """ Representation of the \\sigma^+_i\\sigma^-_j operator in an n-spin system in the e-particle basis. """
    def __init__(self, hs: SpinHilbertSpace, i, j):
        super().__init__(hs)
        self.i = i
        self.j = j
    def matrix(self):
        return exchange(self.i, self.j, FermionSystemSpec(self.hs.N, self.hs.e))
    
class Sz(SPCOperator):
    """ Representation of the Pauli \\sigma^z_i operator in an n-spin system in the e-particle basis. """
    def __init__(self, hs: SpinHilbertSpace, i):
        super().__init__(hs)
        self.i = i
    def matrix(self):
        return Z(self.i, FermionSystemSpec(self.hs.N, self.hs.e))
    
def Z(i, systemspec):
    """ Representation of the Z_i operator in an n-qubit system in the e-particle basis. """
    if systemspec.e > systemspec.n:
        raise ValueError("The number of particles cannot be higher than the number of qubits")
    diag = cp.zeros(systemspec.N)
    available = list(range(systemspec.n))
    for index in itertools.combinations(available, systemspec.e):
        state_index = indexof(tuple(sorted(index)), systemspec.n, systemspec.e)
        diag[state_index] = 1 if i in index else -1
    return cp.diag(diag)