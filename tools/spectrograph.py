#!/usr/bin/env python

import numpy as np
from scipy import interpolate, ndimage, signal
import glob
from astropy.io import fits as pyf
import re
from specutil import SpecWFE, Distortion, simpleDistortion
import logging as log
import matplotlib.pyplot as plt

def Spectrograph(par, image,  lam, simple=True):
    """
    Function Spectrograph

    Apply a translation for the dispersion, a distortion (if desired),
    and compute convolution kernels to account for wavefront errors,
    misalignments, and defocuses.  The function also computes some 
    transmissions from the prism and spectrograph optics; these are 
    appended to the list of transmissions.

    Inputs: 
    1. par:        parameters class
    2. image:      image plane after lenslet (array of PSF-lets)
    3. lam:        wavelength (microns)
    4. simple:     whether to use only the most important distortion coefficients

    Returns: 
    1. image:      image plane after dispersion, distortion
    2. kernels:    List of convolution kernels.  
    3. loc:        Locations of convolution kernels.
    
    """
    
   
    ###################################################################### 
    # Load the spot diagrams and their coordinates.  Scale the coordinates
    # to [0, 1]; values will be [0, 0.5, 1]x[0, 0.5, 1].  [0, 0] is bottom
    # left, [1, 1] is top right, etc.
    ###################################################################### 

    log.info('Loading spot diagrams.')
    # first, select which wavelength PSF to use
    wavel = 660
    kernelscale = 1.265e-6
    spotfields = glob.glob(par.prefix + '/simpsf/%dPSF_*.fits' % wavel)
    kernels = [pyf.open(ifile)[1].data for ifile in spotfields]
    locations = np.zeros((len(spotfields), 2))
    for i in range(len(spotfields)):
        name = re.sub('.fits', '', re.sub('.*PSF_', '', spotfields[i]))
        locations[i, 0] = float(name.split('_')[0])
        locations[i, 1] = float(name.split('_')[1])

    locations /= 2.
    locations += 0.5

    ###################################################################### 
    # Compute dispersion and distorsion from 2D polynomial fit
    ###################################################################### 
    
    #log.info('Apply distortion and dispersion.')
    
#     if simple:
#         image = simpleDistortion(par,image,lam)
#         log.info('Applied simplified model of the distortion/dispersion polynomial.')
#     else:
#         image = Distortion(par, image, lam)
#         log.info('Applied full model of the distortion/dispersion polynomial.')
    
    ###################################################################### 
    # Several convolution kernels here: 
    # Simulated misalignment (includes defocus, spherical and coma)
    # Manufacturing WFEs
    # We will apply all of these together in the detector plane.  The 
    # rationale for this is that the detector will down-sample, so by 
    # only convolving at these points, we save ourselves a computational
    # factor roughly equivalent to the down-sampling.
    # After this step, the kernels are convolved with everything: wavefront
    # errors, spot diagrams of all types.  We only need to convolve them 
    # with the pixel response function (which we will do later).
    # Convolve with the lenslet defocus spot diagram now if we didn't do
    # so in Lenslet()--set by par.vardefoc
    ###################################################################### 

    for i in range(len(locations)):

        ##################################################################
        # Now resample the kernels to the appropriate resolution
        ##################################################################
        
        
        # the scale in the incoming plane is par.pitch/par.pxprlens
        # the scale in the kernels is kernelscale
        # remap kernel to match incoming plate scale
        
        ratio = kernelscale/(par.pitch/par.pxprlens)
        nx = kernels[i].shape[0] * ratio
        ny = kernels[i].shape[1] * ratio
        
        # the factor of 94 is believed to be related to the magnification of the relay
        # the spot diagrams
        #nx = par.pxprlens*kernels[i].shape[1]/94.
        #ny = par.pxprlens*kernels[i].shape[0]/94.

        #x = np.arange(nx)*94./par.pxprlens
        #y = np.arange(ny)*94./par.pxprlens
        x = (np.arange(nx)- nx//2)/ratio 
        y = (np.arange(ny) - ny//2)/ratio

        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        x = r*np.cos(theta + par.philens) + kernels[i].shape[0]//2
        y = r*np.sin(theta + par.philens) + kernels[i].shape[1]//2

        kernels[i] = ndimage.map_coordinates(kernels[i], [y, x])
#         plt.imshow(kernels[i],interpolation='nearest')
#         plt.show()
    log.info('Remapped kernels')


    return image, kernels, locations
