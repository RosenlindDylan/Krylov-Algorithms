import numpy as np

# this just prints the matrices better
np.set_printoptions(precision=3,suppress=True)

# b should have entries generated uniformly between [0,1] according to the paper
def generateB(size):
    return np.random.rand(size)

b = generateB(10)

# an example tolerance for error
tol = 10**(-10)

# A is a sparse positive definite symmetric matrix, later i'll write a method to generate random ones
A = np.array([[1.00000000, 0.03677389, 0.00000000, 0.97869604, 0.00000000, 0.47722024, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
 [0.03677389, 1.00000000, 0.00000000, 0.00000000, 0.84396722, 0.80661557, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
 [0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18467612, 0.00000000, 0.00000000, 0.00000000],
 [0.97869604, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.21125068, 0.00000000, 0.00000000],
 [0.00000000, 0.84396722, 0.00000000, 0.00000000, 1.00000000, 0.66669913, 0.00000000, 0.00000000, 0.87047764, 0.00000000],
 [0.47722024, 0.80661557, 0.00000000, 0.00000000, 0.66669913, 1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
 [0.00000000, 0.00000000, 0.18467612, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.74501799],
 [0.00000000, 0.00000000, 0.00000000, 0.21125068, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.72975303],
 [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.87047764, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.78071397],
 [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.74501799, 0.72975303, 0.78071397, 1.00000000]]
)

# initial guess is x_0 = 0
x = np.zeros(10)


def CGSolver(A, b, x, tol):
    # initial conditions
    residual = b - np.dot(A, x)
    v = residual
    innerResidual = np.linalg.norm(residual)
    
    # runs while the error is greater than specified tolerance
    while np.sqrt(innerResidual) > tol:
        w = np.dot(A, v) # this is the operation that would be done in the memristive array
        alpha = innerResidual / np.dot(v, w)

        x = x + np.dot(alpha, v)
        residual = residual - np.dot(alpha, w)
        innerResidualNew = np.dot(residual, residual)
        v = residual + (innerResidualNew/innerResidual)*v
        innerResidual = innerResidualNew
    return x

y = CGSolver(A,b,x,tol)

print("this is the other method solution vector" + str(y))
print("and checking gives" + str(A @ y))