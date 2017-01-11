#!/usr/bin/env python

import os
from numpy import sqrt,arcsin,arctan
import logging

class Params():

    def __init__(self):
        """
        
        """

        self.saveRotatedInput = True
        self.saveDetector = True
        self.saveLensletPlane = True
        self.parallel = True
        self.maxcpus=6        
        # ******************************************************************
        # Basic resolution/configuration parameters
        # To make this run faster, decrease pxprlens and/or nlens.
        # Decreasing nlens reduces number of lenslets/FOV
        # ******************************************************************

        self.nlens = 108            # Number of lenslets across array
        self.dlam = 0.005           # less than the IFS resolution

        # ******************************************************************
        # Detector stuff
        # ******************************************************************
        
        self.npix = 1024            # Number of pixels in final detector
        self.pixsize = 13e-6        # Pixel size (meters)
        self.pxperdetpix = 5       # Oversampling of the final detector pixels
        self.convolve = True        # whether to convolve the existing kernels with
                                    # gaussian kernel (simulating defocus)
        self.FWHM = 1.5             # FWHM of gaussian kernel

        # ******************************************************************
        # Spectrograph stuff
        # ******************************************************************
        
        self.pitch = 174e-6        # Lenslet pitch (meters)
        self.interlace = 2         # Interlacing
        self.philens = -arcsin(1./sqrt(self.interlace**2+1)) # Rotation angle of the lenslets (radians)

        self.pinhole = True        # Use a pinhole grid?
        self.pin_dia = 25e-6       # Diameter of pinholes (m)
        self.flens = 5.6*self.pitch   # Focal length of lenslet (meters)
        self.distort = 0           # Magnitude of distortion (fiducial=1)
        self.vardefoc = False      # Account for variable defocusing 

        self.prefix = os.path.abspath('./ReferenceFiles') 
        self.exportDir = os.path.abspath('./SimResults')
        self.unitTestsOutputs = os.path.abspath('./unitTestsOutputs')


    def output(self):
        string = ""
        string += "\nNumber of pixels for pupil plane: "+str(self.n_pupil)
        string += "\nNumber of pixels per lenslet: "+str(self.pxprlens)
        string += "\nNumber of lenslets: "+str(self.nlens)
        string += "\nDetector pixel size in meters: "+str(self.pixsize)
        string += "\nLenslet pitch in meters: "+str(self.pitch)
        string += "\nInterlacing: "+str(self.interlace)
        string += "\nAngle of rotation of lenslet array (radians): "+str(self.philens)
        string += "\nUse pinhole mask? "+str(self.pinhole)
        string += "\nPinhole mask diameter (meters): "+str(self.pin_dia)
        string += "\nFocal length of lenslet: "+str(self.flens)
        string += "\nMagnitude of distortion: "+str(self.distort)
        string += "\nVariable defocusing: "+str(self.vardefoc)
        return string+"\n"
        
