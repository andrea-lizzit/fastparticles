import itertools
import matplotlib.pyplot as plt
import numpy as np

def poissonweights(heights, norm=1):
	sigma = np.sqrt(heights/norm)
	sigma[sigma==0] = 1/norm
	return 1/sigma

def paramstring(params, type='file'):
	if type == 'plot':
		return " ".join([f"{k}={v}" for k, v in params.items()])
	elif type == 'file':
		return "-".join([f"{k}{v}" for k, v in params.items()])
	elif type=='metadata':
		metadata = {}
		for k, v in params.items():
			metadata[k] = str(v)
		return metadata
	else:
		raise ValueError(f"Unknown type {type}")

def errorhist(ax, values, **kwargs):
	heights, borders = np.histogram(values, **kwargs)
	norm = np.sum(heights) * np.diff(borders)[0]
	heights = heights / norm
	weights = poissonweights(heights, norm=norm)
	centers = borders[:-1] + np.diff(borders) / 2

	ax.set_ylabel("density")

	ax.errorbar(
		centers,
		heights,
		yerr = 1/weights,
		marker = '.',
		markersize = 0,
		drawstyle = 'steps-mid',
	)

	# add a grid of very light gray color
	ax.grid(color='0.95', linestyle='-', linewidth=1)

def bosonmatplot(systemspec, mat=None):
	w = np.linalg.eigvalsh(mat)
	print("E: ", " ".join([f"{ww:.4g}" for ww in w.flatten()]))
	print("S: ", " ".join([f"{ww:.4g}" for ww in np.diff(w).flatten()]), "\n")
	available = list(range(systemspec.n))
	states_i = itertools.combinations_with_replacement(available, systemspec.e)
	for row, state_i in zip(mat, states_i):
		state = np.zeros(systemspec.n, dtype=int)
		for i in state_i:
			state[i] += 1
		print(" ".join([f"{v:.4g}" for v in row]), " ; ", " ".join([f"{v:d}" for v in state]))