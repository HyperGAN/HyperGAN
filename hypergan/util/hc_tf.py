#This is like ops.py, but for larger compositions of graph nodes.
#TODO: could use a better name
from hypergan.util.ops import *

#TODO can live elsewhere
def find_smallest_prime(x, y):
    for i in range(3,x-1):
        for j in range(1, y-1):
            if(x % (i) == 0 and y % (j) == 0 and x // i == y // j):
                #if(i==j):
                #    return 2,2
                return i,j
    return None,None
