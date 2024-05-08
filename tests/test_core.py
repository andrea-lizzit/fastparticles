import unittest
import numpy as np
from fastparticles.studies import ManyBodyLevels
from fastparticles.matrixensembles import MatrixStats, OganesyanHuseSampler, BosonChainSampler
from fastparticles.operators.operators import FermionSystemSpec, BosonSystemSpec, boson_exchange, indexof, exchange, boson_indexof, boson_a4
import itertools
import random
import math
from tqdm import trange

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
		n_comb = math.factorial(N+K-1)//(math.factorial(K)*math.factorial(N-1))
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
		systemspec = FermionSystemSpec(4, 2)
		self.assertEqual(indexof((0, 1), systemspec.n, systemspec.e), 0)
		self.assertEqual(indexof((0, 2), systemspec.n, systemspec.e), 1)
		self.assertEqual(indexof((1, 3), systemspec.n, systemspec.e), 4)

		systemspec = FermionSystemSpec(5, 4)
		for i, c in enumerate(itertools.combinations(range(systemspec.n), systemspec.e)):
			self.assertEqual(indexof(c, systemspec.n, systemspec.e), i)

		systemspec = FermionSystemSpec(6, 3)
		for i, c in enumerate(itertools.combinations(range(systemspec.n), systemspec.e)):
			self.assertEqual(indexof(c, systemspec.n, systemspec.e), i)

	def test_exchange(self):
		systemspec = FermionSystemSpec(10, 3)
		mat = exchange(9, 6, systemspec).get()
		v, w = np.zeros((systemspec.N,)), np.zeros((systemspec.N,))
		v[indexof((3,4,6), systemspec.n, systemspec.e)] = 1
		w[indexof((3,4,9), systemspec.n, systemspec.e)] = 1
		self.assertIsNone(np.testing.assert_array_equal(mat@v, w))

	def test_exchange_auto(self):
		systemspec = FermionSystemSpec(10, 3)
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

class TestBosonExchange(unittest.TestCase):
	def test_exchange(self):
		systemspec = BosonSystemSpec(10, 3)
		for i, j in np.ndindex(systemspec.n, systemspec.n):
			mat = boson_exchange(i, j, systemspec).get()
			choices = list(range(systemspec.n))
			k, l = random.sample(choices, k=2)
			v, w = np.zeros((systemspec.N,)), np.zeros((systemspec.N,))
			v[boson_indexof(tuple(sorted((j, k, l))), systemspec.n, systemspec.e)] = 1
			w[boson_indexof(tuple(sorted((i, k, l))), systemspec.n, systemspec.e)] = 1
			a = mat@v
			a /= np.linalg.norm(a)
			b = w / np.linalg.norm(w)
			self.assertTrue(np.allclose(a, b))

class TestBosonOperators(unittest.TestCase):
	def test_quartic(self):
		systemspec = BosonSystemSpec(10, 6)
		for test_iter in range(20):
			state = random.choices(range(systemspec.n), k=systemspec.e)
			state = tuple(sorted(state))
			state_coord = np.zeros((systemspec.N,))
			state_coord[boson_indexof(state, systemspec.n, systemspec.e)] = 1
			i = random.randint(0, systemspec.n-1)
			num_exc = state.count(i)
			print(f"state={state}, i={i}, num_exc={num_exc}")
			val = num_exc * (num_exc-1)
			operator = boson_a4(i, systemspec).get()
			self.assertIsNone(np.testing.assert_allclose(operator @ state_coord, state_coord * val))
			

class TestManyBodyStudy(unittest.TestCase):
	def test_manybody(self):
		for n in trange(3, 8):
			for e in range(1, n // 2):
				with self.subTest(n=n, e=e):
					W, t = 1, 1
					sampler = BosonChainSampler(n, W, t, K=0, e=1, rng=np.random.default_rng(42))
					stats = MatrixStats(sampler)
					stats.collect(1)
					study = ManyBodyLevels(e)
					manybody = study(stats)
					sampler = BosonChainSampler(n, W, t, K=0, e=e, rng=np.random.default_rng(42))
					stats = MatrixStats(sampler)
					stats.collect(1)
					res = np.testing.assert_allclose(stats.eigenvalues().flatten(), manybody.eigenvalues().flatten())
					self.assertIsNone(res)

class TestSampler(unittest.TestCase):

	def test_oganesyanhusesampler(self):
		n, e = 13, 2
		sampler = OganesyanHuseSampler(n, 0, e=e)
		systemspec = FermionSystemSpec(n, e)
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

class TestBosons(unittest.TestCase):
	def test_bosonchain(self):
		n = 10
		sampler = BosonChainSampler(n, 1, 2, e=1)
		matrixstats = MatrixStats(sampler)
		matrixstats.collect(10)
		v = matrixstats.eigenvalues()
		for e in range(2, 2):
			with self.subTest(e=e):
				sampler2 = BosonChainSampler(n, 1, 2, e=e)
				matrixstats2 = MatrixStats(sampler2)
				matrixstats2.collect(10)
				v2 = matrixstats2.eigenvalues()
				v2_independent = np.zeros_like(v2)
				for i, c in enumerate(itertools.combinations_with_replacement(range(n), e)):
					v2_independent[:, i] = np.sum(v[:, c], axis=1)
				v2_independent = np.sort(v2_independent, axis=1)
				self.assertIsNone(np.testing.assert_array_almost_equal(v2, v2_independent, decimal=10))



if __name__=="__main__":
	unittest.main()