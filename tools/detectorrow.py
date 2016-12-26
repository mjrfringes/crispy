#!/usr/bin/env python

import numpy as np

def DetectorRow(par, image, kernels, weights, jmin, npix, n):
    
    imageout = np.zeros(npix)

    for j in range(jmin, npix - jmin):
        
        ############################################################
        # Bilinear interpolation means that the kernel change is 
        # linear except at vertices, and the only interior vertex
        # is in the middle of the field.
        ############################################################
        
        if j == jmin or j == npix//2 + 1:
            kernel = np.zeros(kernels[0].shape)
            for k in range(len(kernels)):
                if weights[j, k] != 0:
                    kernel += weights[j, k]*kernels[k]
            dkernel = np.zeros(kernels[0].shape)
            for k in range(len(kernels)):
                if weights[j + 1, k] != 0:
                    dkernel += weights[j + 1, k]*kernels[k]
            dkernel -= kernel
        else:
            kernel += dkernel

        jmid = int(j/(npix - 1.)*image.shape[1])
        
        j1 = max(0, jmid - kernel.shape[1]//2)
        j2 = j1 + kernel.shape[1] #min(image.shape[1], j1 + kernel.shape[1])

        imageout[j] = np.sum(kernel*image[:, j1:j2])
        

    return imageout
