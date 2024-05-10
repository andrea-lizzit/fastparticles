import math
from copy import copy
from functools import cache
import itertools
import cupy as cp

class SystemSpec:
    def __init__(self, n, e):
        self.n = n
        self.e = e

    def remake(self, **kwargs):
        spec = copy(self)
        for key, value in kwargs.items():
            setattr(spec, key, value)
        return spec

def combr(n, e):
    return math.comb(n+e-1, e)

class BosonSystemSpec(SystemSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = combr(self.n, self.e)
        self.statistic = "boson"

class FermionSystemSpec(SystemSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = math.comb(self.n, self.e)
        self.statistic = "fermion"

@cache
def indexofr(qubits, n, e):
    """ Index in the basis of the state with excited qubits in the given positions. Recursive implementation: faster, but might run out of cache. """
    # disable when not debugging
    # it is a relatively expensive check in a time-critical piece of code
    # if not np.all(np.diff(qubits) >= 0):
    #     raise ValueError("Array is not sorted")
    if len(qubits) != e:
        raise ValueError("Wrong number of excited qubits")
    if n == e:
        return 0
    if e == 1:
        return qubits[0]
    k = qubits[0]
    count = math.comb(n, e) - math.comb(n - k, e)
    newqubits = tuple(q-k-1 for q in qubits[1:])
    return count + indexof(newqubits, n-k-1, e-1)

@cache
def indexof(qubits, n, e):
    """[summary]
    Index in the basis of the state with excited qubits in the given positions.
     
    ### Parameters
    1. qubits : tuple
        - *Must be sorted* in ascending order
    2. n: int
        - total number of qubits
    3. e: int
        - number of excited qubits
    """
    # disable when not debugging
    # it is a relatively expensive check in a time-critical piece of code
    # if not np.all(np.diff(qubits) >= 0):
    #     raise ValueError("Array is not sorted")
    count = 0
    while True:
        if len(qubits) != e:
            raise ValueError("Wrong number of excited qubits")
        if n == e:
            return count
        if e == 1:
            return count + qubits[0]
        k = qubits[0]
        count += math.comb(n, e) - math.comb(n - k, e)
        qubits = tuple(q-k-1 for q in qubits[1:])
        n = n-k-1
        e = e-1

def spin_indexof(qubits):
    """[summary]
    Index in the basis of the state.
     
    ### Parameters
    1. state : tuple
        - Tuple of length n where 1 represents spin up and 0 represents spin down
    2. n: int
        - total number of qubits
    """
    # convert tuple of ints into little-endian binary number
    state = 0
    for i, q in enumerate(qubits):
        state += q << i
    return state

@cache
def boson_indexof(exc, n, e):
    """[summary]
    Index in the basis of the state with excited qubits in the given positions.
     
    ### Parameters
    1. qubits : tuple
        - *Must be sorted* in ascending order
    2. n: int
        - total number of qubits
    3. e: int
        - number of excited qubits
    """
    count = 0
    while True:
        if len(exc) != e:
            raise ValueError("Wrong number of excitations")
        if e == 1:
            return count + exc[0]
        k = exc[0]
        count += combr(n, e) - combr(n-k, e)    
        exc = tuple(q-k for q in exc[1:])
        n = n-k
        e = e-1
        

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

def spin_exchange(i, j, n):
    """ Representation of the \\sigma^+_i*\\sigma_j^- / 4 operator in an n-spin system."""
    N = 2**n
    mat = cp.zeros((N, N))
    if i != j:
        M, m = max(i, j), min(i, j)
        for c in range(2**(n-2)):
            small_part = c & ((1 << m) - 1)
            rest = c - small_part
            mid_part = rest & ((1 << M-1) - 1)
            large_part = rest - mid_part
            before = (large_part << 2) + (mid_part << 1) + small_part
            # now there are 0 in position i and j
            after = before + (1 << i) + (0 << j)
            before = before + (0 << i) + (1 << j)
            mat[after, before] = 1
        return mat
    else:
        for c in range(2**(n-1)):
            small_part = c & ((1 << i) - 1)
            large_part = c - small_part
            state = (large_part << 1) + (1 << i) + small_part
            mat[state, state] = 1
        return mat
    
def spin_sigmam(i, n):
    """ Representation of the \\sigma^-_i operator in an n-spin system."""
    N = 2**n
    mat = cp.zeros((N, N))
    for c in range(2**(n-1)):
        small_part = c & ((1 << i) - 1)
        large_part = c - small_part
        before = (large_part << 1) + (1 << i) + small_part
        after = (large_part << 1) + (0 << i) + small_part
        mat[after, before] = 2
    return mat

def spin_sigmap(i, n):
    """ Representation of the \\sigma^+_i operator in an n-spin system."""
    N = 2**n
    mat = cp.zeros((N, N))
    for c in range(2**(n-1)):
        small_part = c & ((1 << i) - 1)
        large_part = c - small_part
        before = (large_part << 1) + (0 << i) + small_part
        after = (large_part << 1) + (1 << i) + small_part
        mat[after, before] = 2
    return mat


def spin_sigmaz(i, n):
    """ Representation of the \\sigma^z_i operator in an n-spin system."""
    stride = 1 << i
    n_strides = 2**(n - i - 1)
    d = cp.tile(cp.concatenate([-1*cp.ones(stride), cp.ones(stride)]), n_strides)
    return cp.diag(d)

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
    if systemspec.e == 1:
        return 0
    mat = cp.zeros((systemspec.N, systemspec.N))
    available = list(range(systemspec.n))
    for index in itertools.combinations_with_replacement(available, systemspec.e-2):
        state_idx = boson_indexof(tuple(sorted(index + (i,i))), systemspec.n, systemspec.e)
        c = index.count(i) + 2
        mat[state_idx, state_idx] = c * (c-1)
    return mat

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