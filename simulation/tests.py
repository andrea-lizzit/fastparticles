import unittest
import numpy as np
from studies import ManyBodyLevels
from matrixensembles import MatrixStats

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

if __name__=="__main__":
	unittest.main()