#!/usr/bin/env python

from scipy import signal
import scipy.interpolate
import scipy.ndimage

import numpy as np
import multiprocessing
import logging
import matplotlib.pyplot as plt
from tools.detutils import frebin
import logging as log

def rebinDetector(par,finalFrame,clip=False):
    '''
    Rebins the dense detector map with the correct scaling while conserving flux.
    This also works with non-integer ratios.
    
    Parameters
    ----------
    par :   Parameter instance
    finalFrame : 2D ndarray
            Dense detector map to be rebinned.
    
    Returns
    -------
    detectorFrame : 2D array
            Return the detector frame with correct pixel scale.
    
    '''
    detpixprlenslet = par.pitch/par.pixsize 
    log.info('Number of detector pixels per lenslet: %f' % detpixprlenslet)
        
    newShape = (finalFrame.shape[0]//(par.pxperdetpix),finalFrame.shape[1]//(par.pxperdetpix))
    log.info('Rebinning final detector. Image has dimensions %dx%d' % newShape)
    detectorFrame = frebin(finalFrame,newShape) 
    
    if clip:
        i = int(detectorFrame.shape[0]*(1. - 1./np.sqrt(2.))/2.)
        detectorFrame = detectorFrame[i:-i, i:-i]

    return detectorFrame



def readDetector(par,IFSimage,inttime=100,append_header=False):
    '''
    Read noise, CIC, dark current; NO TRAPS
    Input is IFSimage in average photons per second
    Quantum efficiency considerations are already taken care of when
    generating IFSimage images
    '''
    if append_header:
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*'*60), end=True)
        par.hdr.append(('comment', '*'*22 + ' Detector readout ' + '*'*20), end=True)
        par.hdr.append(('comment', '*'*60), end=True)    
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('RN',par.RN,'Read noise (electrons/read)'), end=True) 
        par.hdr.append(('CIC',par.CIC,'Clock-induced charge'), end=True) 
        par.hdr.append(('DARK',par.dark,'Dark current'), end=True) 
        par.hdr.append(('Traps',par.Traps,'Use traps? T/F'), end=True) 
        
        
    ### thoughts on implementing the EMGain:
    # This requires an inverse cumulative probability density which depends
    # on the number of incoming electrons in the well, with a max of 32.
    # Suggestion is to pre-compute the 32 required functions, save them
    # then apply them to the array, for example using np.vectorize
    # Another way would be to make lists of coordinates for all pixels with the same
    # values, and call this icdf a maximum of 32 times; after the random numbers
    # are generated, put them back in their right place on the detector.
    ###
    
    return np.random.poisson(IFSimage.data*inttime+par.dark*inttime+par.CIC)+np.random.poisson(par.RN,IFSimage.data.shape)