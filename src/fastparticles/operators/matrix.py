from ..hilbert.pcspace import BosonHilbertSpace, FermionHilbertSpace, SpinHilbertSpace
from . import Opeartor, matrix
from plum import dispatch

@matrix.dispatch
def matrix(op: Operator, hs: BosonHilbertSpace):
	pass