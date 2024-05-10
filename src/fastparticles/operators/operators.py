from fastparticles.hilbert.abstract_hilbert import HilbertSpace
from abc import ABC, abstractmethod
import cupy as cp
from functools import reduce
import numbers

class Operator(ABC):
	def __init__(self, hs: HilbertSpace):
		self.hs = hs
	@abstractmethod
	def matrix(self):
		pass
	def __call__(self, psi, phi):
		pass
	def __add__(self, op):
		if isinstance(op, Operator):
			return OperatorSum(self.hs, [self, op])
		if isinstance(op, numbers.Number):
			op = ScalarOperator(self.hs, op)
			return OperatorSum(self.hs, [self, op])
		raise ValueError(f"Operator addition not supported for type {type(op)}")
	def __radd__(self, op):
		return self + op
	def __rmul__(self, op):
		return self * op
	def __mul__(self, op):
		if isinstance(op, Operator):
			return OperatorProduct(self.hs, [self, op])
		if isinstance(op, numbers.Number):
			op = ScalarOperator(self.hs, op)
			return OperatorProduct(self.hs, [self, op])
		raise ValueError(f"Operator multiplication not supported for type {type(op)}")
	def __sub__(self, op):
		return self + (-op)
	def __rsub__(self, op):
		return -self + op
	def __neg__(self):
		return -1 * self
	def __truediv__(self, op):
		if isinstance(op, numbers.Number):
			return self * (1/op)
		raise ValueError(f"Operator division not defined for operators")
	
	@property
	def size(self):
		""" Defined for backward compatibility. """
		return self.hs.dim
	def sample(self):
		""" Defined for backward compatibility. """
		return self.matrix()
	
class OperatorSum(Operator):
	def __init__(self, hs: HilbertSpace, ops: list[Operator]):
		super().__init__(hs)
		self.ops = ops
	def matrix(self):
		return reduce(cp.add, [op.matrix() for op in self.ops])
	def __call__(self, psi, phi):
		return sum(op(psi, phi) for op in self.ops)
	
class OperatorProduct(Operator):
	def __init__(self, hs: HilbertSpace, ops: list[Operator]):
		super().__init__(hs)
		self.ops = ops
	def matrix(self):
		return reduce(cp.matmul, [op.matrix() for op in self.ops])
	def __call__(self, psi, phi):
		raise NotImplementedError("OperatorProduct does not support __call__")
	
class ScalarOperator(Operator):
	def __init__(self, hs: HilbertSpace, scalar):
		super().__init__(hs)
		self.scalar = scalar
	def matrix(self):
		return cp.eye(self.hs.dim) * self.scalar
	def __call__(self, psi, phi):
		if psi == phi:
			return self.scalar
		return 0