#!/usr/bin/env python
import numpy as np

def h1(x):
        return (x+1)%5
def h2(x):
        return (3*x+1)%5

def minhash(data, hashfuncs):
    rows, cols, sigrows = len(data), len(data[0]), len(hashfuncs)

    # initialize signature matrix with maxint
    sigmatrix = []
    for i in range(sigrows):
        sigmatrix.append([np.inf] * cols)

    for r in range(rows):
        hashvalue = list(map(lambda x: x(r), hashfuncs))
        # if data != 0 and signature > hash value, replace signature with hash value
        for c in range(cols):
            if data[r][c] == 0:
                continue
            for i in range(sigrows):
                if sigmatrix[i][c] > hashvalue[i]:
                    sigmatrix[i][c] = hashvalue[i]

    return sigmatrix

if __name__ == '__main__':

    data = [[1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 0]]

    # print signature matrix
    print(minhash(data, [h1, h2]))