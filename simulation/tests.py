import unittest
import numpy as np
from studies import ManyBodyLevels
from matrixensembles import MatrixStats, OganesyanHuseSampler
from operators import SystemSpec, indexof, exchange
import itertools
import random


def simplestats(r=0):
	stats = MatrixStats(None)
	arr = np.reshape(np.arange(10, dtype=np.float64), (1, 10))
	rng = np.random.default_rng()
	arr += rng.normal(scale=r, size=arr.shape)
	stats.eigenvalues_ = arr
	return stats


class TestManyBody(unittest.TestCase):
    
	def test_combinations_1(self):
		stats = simplestats()
		mbstats = ManyBodyLevels(1)(stats)
		self.assertIsNone(np.testing.assert_array_equal(stats.eigenvalues(), mbstats.eigenvalues()))
	
	def test_combinations_2(self):
		stats = simplestats(0.01)
		mbstats = ManyBodyLevels(2)(stats)
		e = stats.eigenvalues()
		N, K = e.shape[1], 2
		n_comb = np.math.factorial(N+K-1)//(np.math.factorial(K)*np.math.factorial(N-1))
		mbe = np.zeros([e.shape[0], n_comb])
		idx = 0
		for i in range(10):
			for j in range(i, 10):
				mbe[:, idx] = e[:, i]+e[:, j]
				idx += 1
		mbe = np.sort(mbe, axis=1)
		self.assertIsNone(np.testing.assert_array_equal(mbe, mbstats.eigenvalues()))


class TestSpins(unittest.TestCase):

	def test_basis_index(self):
		systemspec = SystemSpec(4, 2)
		self.assertEqual(indexof((0, 1), systemspec.n, systemspec.e), 0)
		self.assertEqual(indexof((0, 2), systemspec.n, systemspec.e), 1)
		self.assertEqual(indexof((1, 3), systemspec.n, systemspec.e), 4)

		systemspec = SystemSpec(5, 4)
		for i, c in enumerate(itertools.combinations(range(systemspec.n), systemspec.e)):
			self.assertEqual(indexof(c, systemspec.n, systemspec.e), i)

		systemspec = SystemSpec(6, 3)
		for i, c in enumerate(itertools.combinations(range(systemspec.n), systemspec.e)):
			self.assertEqual(indexof(c, systemspec.n, systemspec.e), i)

	def test_exchange(self):
		systemspec = SystemSpec(10, 3)
		mat = exchange(9, 6, systemspec).get()
		v, w = np.zeros((systemspec.N,)), np.zeros((systemspec.N,))
		v[indexof((3,4,6), systemspec.n, systemspec.e)] = 1
		w[indexof((3,4,9), systemspec.n, systemspec.e)] = 1
		self.assertIsNone(np.testing.assert_array_equal(mat@v, w))

	def test_exchange_auto(self):
		systemspec = SystemSpec(10, 3)
		for i, j in np.ndindex(systemspec.n, systemspec.n):
			mat = exchange(i, j, systemspec).get()
			choices = list(range(systemspec.n))
			choices.remove(i)
			if i != j:
				choices.remove(j)
			k, l = random.sample(choices, k=2)
			v, w = np.zeros((systemspec.N,)), np.zeros((systemspec.N,))
			v[indexof(tuple(sorted((j, k, l))), systemspec.n, systemspec.e)] = 1
			w[indexof(tuple(sorted((i, k, l))), systemspec.n, systemspec.e)] = 1
			self.assertIsNone(np.testing.assert_array_equal(mat@v, w))


class TestSampler(unittest.TestCase):

	def test_oganesyanhusesampler(self):
		n, e = 13, 2
		sampler = OganesyanHuseSampler(n, 0, e=e)
		systemspec = SystemSpec(n, e)
		H = sampler.sample().get()
		for m in range(n):
			m1 = (m+1) % n
			if m1 < m:
				m, m1 = m1, m
			with self.subTest(m=m, m1=m1):
				state = np.zeros((systemspec.N,))
				state[indexof((m, m1), systemspec.n, systemspec.e)] = 1
				exp = state @ H @ state
				reference = sampler.V*(0.25*systemspec.n-1)
				self.assertAlmostEqual(exp, reference, delta=1e-10)


if __name__=="__main__":
	unittest.main()