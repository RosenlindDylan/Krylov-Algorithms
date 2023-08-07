import numpy as np

import numpy as np
import math
import scipy.sparse as sp
from copy import deepcopy

np.set_printoptions(precision=3,suppress=True)

# method to generate a random sparse matrix A
def generate_sparse_positive_definite_matrix(size, sparsity):
    # Generate a random dense covariance matrix
    dense_cov = np.random.rand(size, size)
    
    # Make the matrix symmetric
    dense_cov = (dense_cov + dense_cov.T) / 2.0
    # Set elements below the diagonal to zero for symmetry
    dense_cov = np.triu(dense_cov)
    
    # Set a percentage of elements to zero for sparsity
    threshold = np.percentile(dense_cov[dense_cov > 0], sparsity)
    dense_cov[dense_cov < threshold] = 0
    
    # Convert dense matrix to sparse matrix
    sparse_cov = sp.csr_matrix(dense_cov)
    
    return sparse_cov

matrix_size = 10
sparsity_percent = 70
sparse_cov_matrix = generate_sparse_positive_definite_matrix(matrix_size, sparsity_percent)
A = deepcopy(sparse_cov_matrix.toarray())

# b should have entries generated uniformly between [0,1]
def generateB(size):
    return np.random.rand(size)

# we'll make x a random vector similar to b for now
def generateX(size):
    return np.random.rand(size)

b = generateB(10)
x = generateX(10)

print('initial guess is ' + str(x))

print('soln vector is ' + str(b))


# Initial guess
p0 = np.zeros((10))

# Place holders for the residual r and A(d)
Ad = np.zeros((10))

tolerance = 1e-10
max_it = 100

it = 0 # iteration counter
diff = 1.0
tol_hist_jac = []

print(' this is x ' + str(x))
print(' this is A ' + str(A))


r = b - (A @ x)

nx = 101                  # number of points in the x direction
ny = 101                  # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = -0.5, 0.5    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx - 1)        # grid spacing in the x direction
dy = ly / (ny - 1)  

def A(v, dx, dy):    
    Av = -((v[:-2]-2.0*v[1:-1]+v[2:])/dx**2 
       + (v[:-2]-2.0*v[1:-1]+v[2:])/dy**2)
    
    return Av

# Initial residual r0 and initial search direction d0
r[1:-1] = -b[1:-1] - A(x, dx, dy)
d = r.copy()

while (diff > tolerance):
    if it > max_it:
        print('\nSolution did not converged within the maximum'
            ' number of iterations'
            f'\nLast l2_diff was: {diff:.5e}')
        print('the final guess vector is ' + str(x))
        break

    # Laplacian of the search direction.
    Ad[1:-1] = A(d, dx, dy)
    # Magnitude of jump.
    alpha = np.sum(r*r) / np.sum(d*Ad)
    # Iterated solution
    xnew = x + alpha*d
    # Intermediate computation
    beta_denom = np.sum(r*r)
    # Update the residual.
    r = r - alpha*Ad
    # Compute beta
    beta = np.sum(r*r) / beta_denom
    # Update the search direction.
    d = r + beta*d
    
    diff = abs(np.linalg.norm(xnew) - np.linalg.norm(x))
    tol_hist_jac.append(diff)
    
    # Get ready for next iteration
    it += 1
    np.copyto(x, xnew)

else:
    print(f'\nThe solution converged after {it} iterations')
    print('the final guess vector is ' + str(x))