import numpy as np

def CGSolver(A, b, x, tol,itMax):
    '''
        Parameters:
            A - (sparse) positive definite symmetrix matrix    
            b - solution vector (entries generated uniformly between [0,1] in LeGallo paper)
            x - initial guess vector, zero vector 
            tol - desired convergence tolerance
            itMax - maximum iterations
        Return:
            final guess vector (x) found once the tolerance or number of iterations is reached

        References:
            legallo
            https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            strang - files
    '''

    # initial conditions
    r = b - np.dot(A, x)
    v = r
    rho = np.linalg.norm(r)
    i = 0
    
    # runs while the error is greater than specified tolerance
    while np.sqrt(rho) > tol and i < itMax:
        w = np.dot(A, v) # this is the operation that would be done in the memristive array
        alpha = rho / np.dot(v, w)

        x = x + np.dot(alpha, v)
        r = r - np.dot(alpha, w)
        rhoNew = np.dot(r, r)
        v = r + (rhoNew/rho)*v
        rho = rhoNew
        i += 1
    return x