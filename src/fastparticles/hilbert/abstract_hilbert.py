from abc import ABC, abstractmethod

class Symmetry(ABC):
    @abstractmethod
    def subconf(self, conf):
        pass

class HilbertSpace(ABC):
    @abstractmethod
    def index(self, state):
        pass
    @abstractmethod
    def valid(self, state):
        pass
    @abstractmethod
    def __eq__(self, other):
        pass
    @property
    def dim(self):
        return self._dim()
    @abstractmethod
    def _dim(self):
        pass