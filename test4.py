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
tol = 10**(-10)

print('initial guess is ' + str(x))

print('soln vector is ' + str(b))


# initial calculations
residual = b - (A @ x) #check if this multiplies correctly

# print('first estimation leaves the residual' + str(residual))

d = residual

deltaNew = np.linalg.norm(residual) #inner product
# print('and the initial delta value' + str(deltaNew))
deltaNaught = deltaNew


# iteration
i = 0
for i in range(5):
    q = (A @ d)

    # print('this is q value ' + str(q))

    alpha = deltaNew / (np.dot(d,q))
    # print('this is alpha value ' + str(alpha))

    #update guess
    x = x + (alpha * d)

    residual = residual - (alpha * q)

    deltaOld = deltaNew

    beta = deltaNew / deltaOld

    d = residual + (beta * d)

    i += 1
print(
    'process ended with residual' + str(residual)
    + ' \n and solution vector ' + str(d)
)