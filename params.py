#!/usr/bin/env python

import os
from numpy import sqrt,arcsin

class Params():

    def __init__(self):
        '''
        Main class containing all the sim parameters
        '''

        self.saveRotatedInput = False
        self.saveLensletPlane = False 
        self.saveDetector = True
        self.prefix = os.path.abspath('./ReferenceFiles') 
        self.exportDir = os.path.abspath('./SimResults')
        self.unitTestsOutputs = os.path.abspath('./unitTestsOutputs')
        self.wavecalDir = self.prefix+'/wavecal/'
#         self.wavecalDir = self.prefix+'/HighSNRWavecal/'
#         self.wavecalDir = self.prefix+'/wavecal/polychrome/'
#         self.wavecalName = 'VARIAcalCube.fits'
#         self.wavecalName = 'LLTFcalCube.fits'
        self.wavecalName = 'simCalCube.fits'
        self.filelist = []
        self.lamlist = []

        ###################################################################### 
        # Basic resolution/configuration parameters
        ###################################################################### 

        self.nlens = 108            # Number of lenslets across array (account for rotation)
        self.dlam = 0.005           # Resolution of recovered cube (microns)
        self.R = 110                # Determine the resolution of final cube
        self.pitch = 174e-6         # Lenslet pitch (meters)
        self.interlace = 2          # Interlacing
        self.philens = arcsin(1./sqrt(self.interlace**2+1)) # Rotation angle of the lenslets (radians)
        self.pinhole = True         # Use a pinhole grid?
        self.pin_dia = 25e-6        # Diameter of pinholes (m)

        ###################################################################### 
        # Detector stuff
        ###################################################################### 
        
        self.npix = 1024            # Number of pixels in final detector
        self.pixsize = 13e-6        # Pixel size (meters)
        self.pxperdetpix = 11       # Oversampling of the final detector pixels
        self.convolve = True        # whether to convolve the existing kernels with
                                    # gaussian kernel (simulating defocus)
        self.FWHM = 2               # FWHM of gaussian kernel
        self.gaussian = True        # Use standard Gaussian kernels instead of library

        ###################################################################### 
        # Spectrograph stuff
        ###################################################################### 
        
        self.distort = True         # apply distortion or not

    def output(self):
        string = ""
        string += "\nNumber of lenslets: "+str(self.nlens)
        string += "\nLenslet pitch in meters: "+str(self.pitch)
        string += "\nInterlacing: "+str(self.interlace)
        string += "\nRotation angle: "+str(self.philens)
        string += "\nPinhole? "+str(self.pinhole)
        string += "\nPinhole diameter: "+str(self.pin_dia)
        string += "\nNumber of pixels in final detector: "+str(self.npix)
        string += "\nDetector pixel size in meters: "+str(self.pixsize)
        string += "\nOversampling: "+str(self.pxperdetpix)
        string += "\nConvolve with Gaussian as defocus? "+str(self.convolve)
        string += "\nFWHM of Gaussian convolution "+str(self.FWHM)+' detector pixel'
        return string+"\n"
        