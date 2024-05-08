import numpy as np

def find_peak(hist, bin_edges):
	i = np.argmax(hist)
	return (bin_edges[i] + bin_edges[i+1]) / 2

def peak_position(distribution, *params):
	hist, bin_edges = np.histogram(distribution, np.linspace(0, 2, 25))
	return find_peak(hist, bin_edges)

def search_w(distribution_f, range, tol=None, rtol=None, target=0.45, invert=False):
	""" Logarithmic search algorithm.
	
	Returns: the value of the parameter which produces a peak at target, within a tolerance tol. """
	if tol == None and rtol == None:
		raise ValueError("Must specify one of tol and rtol.")
	
	if tol:
		if range[1] - range[0] < tol:
			return (range[0] + range[1]) / 2
	if rtol:
		if (range[1] - range[0]) * 2 / (range[0] + range[1]) < rtol:
			return (range[0] + range[1]) / 2
	mid = np.mean(range)
	distribution = distribution_f(mid)
	val = np.mean(distribution)
	print(f"[{val:.2f}, {range[0]:.5f}, {range[1]:.5f}]", end='\r')
	condition = (val > target) ^ invert
	if condition:
		return search_w(distribution_f, (mid, range[1]), tol=tol, rtol=rtol)
	else:
		return search_w(distribution_f, (range[0], mid), tol=tol, rtol=rtol)
	
