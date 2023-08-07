import numpy

class CGSolver:
    def __init__(self, A, b) -> None:
        numpy.set_printoptions(precision=3,suppress=True)

        # later these will be set within a different method so it isn't only dependent
        #   on initialization
        x = 0
        residual = 0
        delta = 0

    # this is the low precision in memory computing array
    # returns the product of the matrix A and the current guess x
    def right_solver(A,x):
        return numpy.multiply(A,x)


    # this represents the high precision computational unit
    # this updates the guess vector x and returns it for the right solver
    def left_solver(A,x,b):
        # for first time getting called
        if x == 0:
            residual = b - CGSolver.right_solver(A,x)
            z = 0
            rho = residual


        

        alpha = (numpy.norm(residual))**2


        pass

    def fullSolver(A,b,x):
        pass