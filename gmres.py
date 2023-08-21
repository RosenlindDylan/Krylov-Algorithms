import numpy as np

# this just prints the matrices better
np.set_printoptions(precision=3,suppress=True)

def GMRESSolver(A, b, x_0, max_iter):
    # given r and initals beta and v_1
    # finding dim
    n = A.shape[0]
    H = np.zeros((max_iter + 1, max_iter)) # array of all h
    x = x_0.copy()
    Q = np.zeros((n, max_iter + 1))

    r = b - np.dot(A, x_0)
    beta = np.linalg.norm(r)
    Q[:,0] = r / beta
    
    # v is q[k] ( updating x value)
    # w is y ( residual)
    # l is j

    for k in range(max_iter):
        w = np.dot(A, Q[:,k]) # memristive array calculation

        for l in range(k + 1):
            H[l,k] = np.dot(Q[:,l],w) 
            w -= H[l,k] * Q[:,l]


        H[k+1,k] = np.linalg.norm(w)

        if (H[k + 1, k] != 0 and k != max_iter - 1):
            Q[:, k + 1] = w / H[k + 1, k]

        b_vector = np.zeros(max_iter + 1)
        b_vector[0] = beta

        min, _ = np.linalg.lstsq(H[:k + 2, :k + 1], b_vector[:k + 2], rcond=None)[:2]

        x = x_0 + np.dot(Q[:, :k + 1], min)


    return x

