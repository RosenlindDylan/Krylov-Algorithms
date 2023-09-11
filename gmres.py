import numpy as np

# this just prints the matrices better
np.set_printoptions(precision=3,suppress=True)

def GMRESSolver(A, b, x_0, itMax):
    '''
        Parameters:
            A - sparse matrix    
            b - solution vector (entries generated uniformly between [0,1] in LeGallo paper)
            x - initial guess vector, zero vector 
            itMax - maximum iterations
        Return:
            final guess vector (x) found once the number of iterations is reached
        References:
            Gallo, M. L., Sebastian, A., et. al (2018). Mixed-precision in-memory computing. 
                Nature Electronics. https://doi.org/10.1038/s41928-018-0054-8
            Saad, Y., &amp; Schultz, M. (n.d.). Iterative methods for sparse linear systems 
            - Stanford University. A GENERALIZED MINIMAL RESIDUAL ALGORITHM FOR SOLVING NON
              SYMMETRIC LINEAR SYSTEMS. https://web.stanford.edu/class/cme324/saad.pdf 
    '''
    # given r and initals beta and v_1
    # finding dim
    n = A.shape[0]
    H = np.zeros((itMax + 1, itMax)) # array of all h
    x = x_0.copy()
    Q = np.zeros((n, itMax + 1))

    r = b - np.dot(A, x_0)
    beta = np.linalg.norm(r)
    Q[:,0] = r / beta
    
    # v is q[k] ( updating x value)
    # w is y ( residual)
    # l is j

    for k in range(itMax):
        w = np.dot(A, Q[:,k]) # memristive array calculation

        for l in range(k + 1):
            H[l,k] = np.dot(Q[:,l],w) 
            w -= H[l,k] * Q[:,l]


        H[k+1,k] = np.linalg.norm(w)

        if (H[k + 1, k] != 0 and k != itMax - 1):
            Q[:, k + 1] = w / H[k + 1, k]

        b_vector = np.zeros(itMax + 1)
        b_vector[0] = beta

        min, _ = np.linalg.lstsq(H[:k + 2, :k + 1], b_vector[:k + 2], rcond=None)[:2]

        x = x_0 + np.dot(Q[:, :k + 1], min)


    return x

