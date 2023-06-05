import numpy as np
import scipy
import pyamg
import matplotlib.pyplot as plt
from ndlattice import mdlattice

# ------------------------------------------------------------------
# Step 2: setup up the system using pyamg.gallery
# ------------------------------------------------------------------
n = 20
stencil = pyamg.gallery.diffusion_stencil_2d(type='FE', epsilon=0.001, theta=np.pi / 3)
A = pyamg.gallery.stencil_grid(stencil, (n, n), format='csr')
N = A.shape[0]
print(A.toarray().shape)
A = scipy.sparse.csr_array(A)
print(A.toarray().shape)
b = np.random.rand(A.shape[0])                     # pick a random right hand side
pyamg.solve(A, b)
# b = np.zeros(N)
# ------------------------------------------------------------------
# Step 3: setup of the multigrid hierarchy
# ------------------------------------------------------------------
ml = pyamg.smoothed_aggregation_solver(A)   # construct the multigrid hierarchy

# ------------------------------------------------------------------
# Step 4: solve the system
# ------------------------------------------------------------------
res1 = []
x = ml.solve(b, tol=1e-12, residuals=res1)  # solve Ax=b to a tolerance of 1e-12

# ------------------------------------------------------------------
# Step 5: print details
# ------------------------------------------------------------------
print("\n")
print("Details: Default AMG")
print("--------------------")
print(ml)      