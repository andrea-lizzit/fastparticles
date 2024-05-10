from fastparticles.hilbert import FSpinHilbertSpace
from .operators import Operator
from fastparticles.indexing import spin_exchange, spin_sigmam, spin_sigmap, spin_sigmaz

class SOperator(Operator):
    def __init__(self, hs: FSpinHilbertSpace):
        super().__init__(hs)
    def __call__(self, psi, phi):
        raise NotImplementedError("Not implemented yet")

class SExchange(SOperator):
    def __init__(self, hs: FSpinHilbertSpace, i, j):
        super().__init__(hs)
        self.i = i
        self.j = j
    def matrix(self):
        return spin_exchange(self.i, self.j, self.hs.N)
    
class Sz(SOperator):
    def __init__(self, hs: FSpinHilbertSpace, i):
        super().__init__(hs)
        self.i = i
    def matrix(self):
        return spin_sigmaz(self.i, self.hs.N)

class Sp(SOperator):
    def __init__(self, hs: FSpinHilbertSpace, i):
        super().__init__(hs)
        self.i = i
    def matrix(self):
        return spin_sigmap(self.i, self.hs.N)
    
class Sm(SOperator):
    def __init__(self, hs: FSpinHilbertSpace, i):
        super().__init__(hs)
        self.i = i
    def matrix(self):
        return spin_sigmam(self.i, self.hs.N)
    
class Sx(SOperator):
    def __init__(self, hs: FSpinHilbertSpace, i):
        super().__init__(hs)
        self.i = i
    def matrix(self):
        return 0.5*(spin_sigmap(self.i, self.hs.N) + spin_sigmam(self.i, self.hs.N))

class Sy(SOperator):
    def __init__(self, hs: FSpinHilbertSpace, i):
        super().__init__(hs)
        self.i = i
    def matrix(self):
        return -0.5j*(spin_sigmap(self.i, self.hs.N) - spin_sigmam(self.i, self.hs.N))