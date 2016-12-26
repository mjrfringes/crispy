#!/usr/bin/env python

import numpy as np

def Conv2d(bigarr, smallarr):
    
    """
    Function Conv2d
    
    Inputs:
    1. bigarr:    larger array
    2. smallarr:  smaller array

    Outputs: 
    1. outarr:    convolved array

    Equivalent to scipy.signal.convolve2d(bigarr, smallarr, mode='same'),
    but faster under some circumstances for reasons I don't understand.
    """

    outarr = np.zeros(bigarr.shape)
    dim1, dim2 = smallarr.shape

    for i in range(smallarr.shape[0]):
        for j in range(smallarr.shape[1]):
            if smallarr[i, j] == 0:
                continue
            
            ###############################################################
            # Line up the arrays with one another, tricky at the edges.
            ###############################################################

            i1 = i - dim1//2
            j1 = j - dim2//2
            
            if i1 == 0 and j1 == 0:
                outarr += bigarr*smallarr[i, j]
            elif i1 > 0 and j1 == 0:
                outarr[i1:] += bigarr[:-i1]*smallarr[i, j]
            elif i1 < 0 and j1 == 0:
                outarr[:i1] += bigarr[-i1:]*smallarr[i, j]

            elif i1 == 0 and j1 < 0:
                outarr[:, :j1] += bigarr[:, -j1:]*smallarr[i, j]
            elif i1 > 0 and j1 < 0:
                outarr[i1:, :j1] += bigarr[:-i1, -j1:]*smallarr[i, j]
            elif i1 < 0 and j1 < 0:
                outarr[:i1, :j1] += bigarr[-i1:, -j1:]*smallarr[i, j]

            elif i1 == 0 and j1 > 0:
                outarr[:, j1:] += bigarr[:, :-j1]*smallarr[i, j]
            elif i1 > 0 and j1 > 0:
                outarr[i1:, j1:] += bigarr[:-i1, :-j1]*smallarr[i, j]
            elif i1 < 0 and j1 > 0:
                outarr[:i1, j1:] += bigarr[-i1:, :-j1]*smallarr[i, j]

    return outarr
