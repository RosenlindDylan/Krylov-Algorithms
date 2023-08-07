import numpy
from cgsolver import CGSolver

A = numpy.array([[1,0,0],[0,1,0],[0,0,1]])

test = CGSolver(A)