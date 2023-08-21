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
            Gallo, M. L., Sebastian, A., et. al (2018). Mixed-precision in-memory computing. 
                Nature Electronics. https://doi.org/10.1038/s41928-018-0054-8
            Shewchuk, J. (1994). An introduction to the conjugate gradient method without the agonizing pain.
                https://www.semanticscholar.org/paper/An-Introduction-to-the-Conjugate-Gradient-Method-Shewchuk/
            Mathematical Methods for Engineers II- Krylov Subspaces and Conjugate Gradients : Gilbert Strang 
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