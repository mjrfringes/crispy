#!/usr/bin/env python

from scipy import signal
import scipy.interpolate
import scipy.ndimage

import numpy as np
import multiprocessing
from detectorrow import DetectorRow
from parallel_utils import Task, Consumer
import logging
import matplotlib.pyplot as plt
from detutils import frebin


def simpleDetector(par,image):
    '''
    Apply a simple, flux-conservative detector binning function
    '''
    pixperlenslet = par.pitch/par.pixsize
    logging.debug('pixperlenslet = %f' % pixperlenslet)
    npix = int(image.shape[0]/(par.pxprlens/pixperlenslet))
    logging.info('Final size of image = %dx%d' % (npix,npix))

    imageout = frebin(image,(npix,npix))
    
    return imageout


def Detector(par, image, kernels, locations, parallel=True,
             maxcpus=6):
    
    """
    Function Detector

    Convolve the input image with the appropriate kernels to include
    misalignments, focus errors, and other optical aberrations.  Also
    convolve with a top-hat pixel response function and rebin to the
    final pixel resolution.  Save time by only doing the convolution
    at the final pixel values.

    Inputs:
    1. par:        parameters class
    2. image:      image plane after dispersion, distortion
    3. kernels:    List of convolution kernels.  
    4. loc:        Locations of convolution kernels.
    5. parallel:      boolean--run in parallel?  Default True
    6. maxcpus:       maximum number of cpus to use.  Default 6, only used
                      if parallel=True

    Output:
    1. imageout:   image convolved with relevant kernels, pixel response,
                   resampled to proper pixel scale
 
    """

    ####################################################################
    # Pixel binning
    ####################################################################

    pixperlenslet = par.pitch/par.pixsize
    n = int(par.pxprlens/pixperlenslet)
    logging.info('pixperlenslet = %f' % pixperlenslet)
    logging.info('integer divider n = %d' % n)

    ####################################################################
    # Top-hat pixel response function.  Then shrink the kernel as much
    # as possible while preserving at least minfrac, 99.99%, of the flux.
    # In practice the loss will be smaller than this because it is 
    # capped at 0.01% in *any* of the kernels with the same final
    # dimensions.
    ####################################################################

    logging.info('Convolving individual kernels with pixel response function.')
    pixresponse = np.ones((n, n))*1./n**2
    minfrac = 1 - 1e-4

    for i in range(len(kernels)):
        kernels[i] = signal.convolve2d(kernels[i], pixresponse)
        kernels[i] /= np.sum(kernels[i])
    
#     logging.info('Shrink kernels to keep 99.99% of flux.')
#     nx, ny = kernels[0].shape
#     bestarea = nx*ny + 1
#     for i in range(nx//3):
#         for j in range(ny//3):
#             area = (nx - 2*i)*(ny - 2*j)
#             ok = [np.sum(kernel[i:-i, j:-j]) > minfrac for kernel in kernels]
#             if np.all(np.asarray(ok), axis=0) and area < bestarea:
#                 bestarea = area
#                 ibest, jbest = [i, j]
# 
    for i in range(len(kernels)):
#         kernels[i] = kernels[i][ibest:-ibest, jbest:-jbest]
        plt.imshow(kernels[i],interpolation='nearest')
        print np.sum(kernels[i])
        plt.show()

    npix = int(image.shape[0]/(par.pxprlens/pixperlenslet))
    logging.info('Final detector size is %dx%d' % (npix,npix))
    imin = int(kernels[0].shape[0]*npix/image.shape[0]) + 1
    jmin = int(kernels[0].shape[1]*npix/image.shape[1]) + 1

    imageout = np.zeros((npix, npix))
    loc = np.asarray(locations)
    kernel = np.zeros(kernels[0].shape)
    kernels = np.asarray(kernels)

    ####################################################################
    # Calculate weights for bilinear interpolation
    ####################################################################

    logging.info('Calculate weights for bilinear interpolation between kernels.')

    xfrac = np.linspace(0, npix, npix)/npix
    yfrac, xfrac = np.meshgrid(xfrac, xfrac)

    allweights = np.ones((npix, npix, len(loc)))*1/0.25
    for i in range(len(loc)):
        allweights[:, :, i] *= (np.abs(xfrac - loc[i, 0]) < 0.5)
        allweights[:, :, i] *= (np.abs(yfrac - loc[i, 1]) < 0.5)

    for i in range(npix):
        for k in range(len(loc)):
            if xfrac[i, 0] > loc[k, 0]:
                allweights[i, :, k] *= loc[k, 0] + 0.5 - xfrac[i, :]
            else:
                allweights[i, :, k] *= xfrac[i, :] - (loc[k, 0] - 0.5)

    for j in range(npix):
        for k in range(len(loc)):
            if yfrac[0, j] > loc[k, 1]:
                allweights[:, j, k] *= loc[k, 1] + 0.5 - yfrac[:, j]
            else:
                allweights[:, j, k] *= yfrac[:, j] - (loc[k, 1] - 0.5)
    #plt.imshow(allweights[:,:,0],interpolation='nearest')
    #plt.show()
    logging.info('Compute detector pixel value row by row.')
    if not parallel:
        for i in range(imin, npix - imin):
            
            imid = int(xfrac[i, 0]*image.shape[0])
            i1 = max(0, imid - kernels[0].shape[0]//2)
            i2 = min(image.shape[0], i1 + kernels[0].shape[0])
            imout = DetectorRow(par, image[i1:i2],
                                        kernels, allweights[i], jmin, npix, n)
            imageout[i] = imout

    else:     
        logging.info('Starting parallel computing for detector map')
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        ncpus = min(multiprocessing.cpu_count(), maxcpus)
        consumers = [ Consumer(tasks, results)
                      for i in range(ncpus) ]
        for w in consumers:
            w.start()
        for i in range(imin, npix - imin):
            
            imid = int(xfrac[i, 0]*image.shape[0])
            i1 = max(0, imid - kernels[0].shape[0]//2)
            i2 = min(image.shape[0], i1 + kernels[0].shape[0])

            tasks.put(Task(i, DetectorRow,
                           (par, image[i1:i2],
                            kernels, allweights[i], jmin, npix, n)))
        for i in range(ncpus):
            tasks.put(None)
            
        for i in range(imin, npix - imin):
            index, result = results.get()
            imout1= result
            imageout[index] = imout1

    return imageout
