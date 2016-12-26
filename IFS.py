#!/usr/bin/env python

'''
Heritage: CHARIS simulator by T. Brandt and his team
Code was adapted to match PISCES/WFIRST IFS characteristics
'''


import numpy as np
from astropy.io import fits as pyf
from params import Params
import tools
import time
from makeheader import Makeheader
import logging
from scipy import signal
import matplotlib.pyplot as plt
from image import Image



def propSingleWavelength(imageplane,lam,dlam,par=None):
	
    """
    Single wavelength propagation
    
    Propagate a single wavelength through the IFS
    
    Inputs:
    1. imageplane: 			complex 2D array representing the image at the lenslet array
    2. lam:					wavelength (microns)
    3. dlam:				delta wavelength (microns)
    4. par:					parameters instance
    """

    ###################################################################### 
    # Defines an array of times for performance monitoring
    ###################################################################### 
    t = {'Start program':time.time()}
    
    ###################################################################### 
    # load parameter instance, defined in params.py
    ###################################################################### 
    if par is None:
        try:
            par = Params()
            logging.debug('Parameters initialized:'+par.output())
        except:
            logging.exception('Issue with initializing the parameters')
    
    logging.info('Computing detector plane for wavelength %.3f and bandwidth %.3f (microns)' % (lam,dlam))
    
    ###################################################################### 
    # Go through the lenslet array.  We will then be in the detector
    # plane (more or less), we just have to do some convolutions and 
    # treat the dispersion as a wavelength-dependent translationp.
    ###################################################################### 
    t['Start lenslets'] = time.time()
    logging.info('Propagate through lenslet array')
    logging.info('Input plane is %dx%d' % imageplane.shape)

    # first, do some treatment of the input image:
    # rotate the image by the lenslet array angle
    # supersample the image to match lenslet array, perhaps some multiplier of that
    # use map_coordinates, I guess
    # at the end of this, we have an array corresponding to the flux for each lenslet
    lensletplane = tools.Lenslet(par, imageplane, lam, dlam,parallel=False)
    Image(data=lensletplane).write('lensletplane_%.2fum.fits' % (lam))
    #out = pyf.HDUList(pyf.PrimaryHDU(lensletplane.astype(np.float32)))
    #out.writeto('lensletplane_%.2fum.fits' % (lam), clobber=True)

    ###################################################################### 
    # Spectrograph include the dispersion and distortion.  It outputs the
    # convolution kernels to be applied with the detector resampling.
    ###################################################################### 

    t['Start spectrograph']=time.time()
    logging.info('Propagate through spectrograph')
    pinholeplane,  kernels, locations = tools.Spectrograph(par, lensletplane, lam)
    
    out = pyf.HDUList(pyf.PrimaryHDU(pinholeplane.astype(np.float32)))
    out.writeto('pinholeplane_%.2fum.fits' % (lam), clobber=True)

    ###################################################################### 
    # Convolve with detector response
    ###################################################################### 
    t['Start detector']=time.time()
    logging.info('Convolve with detector response')
    finalframe = tools.Detector(par, pinholeplane, kernels, locations)
    #finalframe = tools.simpleDetector(par,lensletplane)
    
    t['Start write to fits']=time.time()

    #head = Makeheader(par)
    out = pyf.HDUList(pyf.PrimaryHDU(finalframe.astype(np.float32)))
    out.writeto('test_image_%.2fum.fits' % (lam), clobber=True)
    
    string = "Performance:"
    string += "\n  %d seconds for initial calculations" % (t['Start lenslets'] - t['Start program'])
    string += "\n  %d seconds for lenslet transforms" % (t['Start spectrograph'] - t['Start lenslets'])
    string += "\n  %d seconds for spectrograph distortion, dispersion" % (t['Start detector'] - t['Start spectrograph'])
    string += "\n  %d seconds for convolutions, detector binning" % (t['Start write to fits'] - t['Start detector'])
    string += "\n  %d seconds total" % (t['Start write to fits'] - t['Start program'])    
    logging.info(string)
    
    # return array
    return finalframe


    
def main():

	tools.initLogger('IFS.log')
	
	logging.info('Starting computation.')
	
	# inputs should be an image cube, and a list of wavelengths for each cube slice
	
	imageplane = np.ones((16000,16000))+0.j
	dlam = 0.2
	frameList = [propSingleWavelength(imageplane,lam,dlam) for lam in np.arange(0.637,0.8,dlam)]
	frameList = np.asarray(frameList)
	print frameList.shape
	finalFrame = np.sum(frameList,axis=0)
	out = pyf.HDUList(pyf.PrimaryHDU(finalFrame.astype(np.float32)))
	out.writeto('test_image.fits', clobber=True)
	logging.info('Done.')
	logging.shutdown()

if __name__ == '__main__':
	main()
	
	

