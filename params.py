#!/usr/bin/env python

import os
from numpy import sqrt,arcsin

class Params():

    def __init__(self):
        '''
        Main class containing all the sim parameters
        '''

        self.saveRotatedInput = False
        self.saveLensletPlane = False   # not used
        self.saveDetector = True
        self.prefix = os.path.abspath('./ReferenceFiles') 
        self.exportDir = os.path.abspath('./SimResults')
        self.unitTestsOutputs = os.path.abspath('./unitTestsOutputs')

        ###################################################################### 
        # Basic resolution/configuration parameters
        ###################################################################### 

        self.nlens = 108            # Number of lenslets across array
        self.dlam = 0.005           # less than the IFS resolution (not yet used

        ###################################################################### 
        # Detector stuff
        ###################################################################### 
        
        self.npix = 1024            # Number of pixels in final detector
        self.pixsize = 13e-6        # Pixel size (meters)
        self.pxperdetpix = 25       # Oversampling of the final detector pixels
        self.convolve = True        # whether to convolve the existing kernels with
                                    # gaussian kernel (simulating defocus)
        self.FWHM = 1.5             # FWHM of gaussian kernel

        ###################################################################### 
        # Spectrograph stuff
        ###################################################################### 
        
        self.distort = True         # apply distortion or not
        self.pitch = 174e-6         # Lenslet pitch (meters)
        self.interlace = 2          # Interlacing
        self.philens = -arcsin(1./sqrt(self.interlace**2+1)) # Rotation angle of the lenslets (radians)

        self.pinhole = True         # Use a pinhole grid?
        self.pin_dia = 25e-6        # Diameter of pinholes (m)

