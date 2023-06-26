import numpy as np
import cupy as cp
import math
from copy import copy
from collections import namedtuple
import itertools
from functools import cache

class SystemSpec:
    def __init__(self, n, e):
        self.n = n
        self.e = e
        self.N = math.comb(n, e)

    def remake(self, **kwargs):
        spec = copy(self)
        for key, value in kwargs.items():
            setattr(spec, key, value)
        return spec

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
    """ Index in the basis of the state with excited qubits in the given positions. """
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
    
