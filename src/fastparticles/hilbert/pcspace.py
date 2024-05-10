from .abstract_hilbert import HilbertSpace
from fastparticles.indexing import indexof, boson_indexof, spin_indexof, combr
import math

class BosonHilbertSpace(HilbertSpace):
    def __init__(self, N, e):
        self.N = N
        self.e = e
    def index(self, state):
        exc = self.to_exc(state)
        return boson_indexof(exc, self.N, len(exc))
    def valid(self, state):
        return sum(state) == self.e
    def to_exc(self, state):
        exc = []
        for i, s in enumerate(state):
            exc.extend([i]*s)
        return tuple(exc)
    def _dim(self):
        return combr(self.N, self.e)
    def __eq__(self, other):
        if not isinstance(other, BosonHilbertSpace):
            return False
        return self.N == other.N and self.e == other.e
    
class FermionHilbertSpace(HilbertSpace):
    def __init__(self, N, e):
        self.N = N
        self.e = e
    def index(self, state):
        return indexof(state, self.N, self.e)
    def valid(self, state):
        return sum(state) == self.e and all(s <= 1 for s in state)
    def to_exc(self, state):
        return tuple(i for i, s in enumerate(state) if s == 1)
    def _dim(self):
        return math.comb(self.N, self.e)
    def __eq__(self, other):
        if not isinstance(other, FermionHilbertSpace):
            return False
        return self.N == other.N and self.e == other.e
    
def SpinHilbertSpace(*args):
    if len(args) == 2:
        return FSpinHilbertSpace(*args)
    if len(args) == 3:
        return FPSpinHilbertSpace(*args)
    raise ValueError("Invalid number of arguments")

class FPSpinHilbertSpace(FermionHilbertSpace):
    def __init__(self, N, e, localdim):
        if localdim != 2:
            raise NotImplementedError("Higher spins not implemented yet.")
        super().__init__(N, e)
    def __eq__(self, other):
        if not isinstance(other, FPSpinHilbertSpace):
            return False
        return self.N == other.N and self.e == other.e and self.localdim == other.localdim

class FSpinHilbertSpace(HilbertSpace):
    def __init__(self, N, localdim):
        if localdim != 2:
            raise NotImplementedError("Higher spins not implemented yet.")
        self.N = N
    def index(self, state):
        return spin_indexof(state, self.N)
    def valid(self, state):
        return self.index(state) < 2**self.N
    def _dim(self):
        return 2**self.N
    def __eq__(self, other):
        if not isinstance(other, FSpinHilbertSpace):
            return False
        return self.N == other.N
    
class LinHilbertSpace:
    def build_tables(self):
        pass

    def __init__(self):
        self.build_tables()

    def index(self, state):
        s1, s2 = state[:self.dim // 2], state[self.dim // 2:]
        return self.Ta[s1] + self.Tb[s2]