import numpy as np
import math
from copy import copy
from collections import namedtuple


class SystemSpec:
    def __init__(self, n, e):
        self.n = n
        self.e = e
        self.N = math.comb(n, e)

    def remake(self, **kwargs):
        spec = copy(self)
        for key, value in kwargs:
            setattr(spec, key, value)
        return spec

def indexof(qubits, systemspec):
    """ Index in the basis of the state with excited qubits in the given positions. """
    if not sorted(excitations):
        raise ValueError("Array is not sorted")
    if len(qubits) != systemspec.e:
        raise ValueError("Wrong number of excited qubits")
    if systemspec.n == systemspec.e:
        return 0
    k = qubits[0]
    count = math.comb(systemspec.n, systemspec.e) - math.comb(systemspec.n - k, systemspec.e)
    return count + indexof(qubits[1:], systemspec.remake(n=systemspec.n-1))

    
def exchange(i, j, systemspec):
    """ Representation of the a_i^\dagger*a_j operator in an n-qubit system in the e-particle basis. """
    if e > n:
        raise ValueError("The number of particles cannot be higher than the number of qubits")
    mat = np.zeros((systemspec.N, systemspec.N))
    for index in np.ndindex(*[systemspec.n]*(systemspec.e-1)):
        if np.any(index == i) or np.any(index == j):
            continue
        before = sort(np.insert(index, e, j))
        after = sort(np.insert(index, e, i))
        mat[after, before] = 1
    return mat
    
