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

