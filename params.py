#!/usr/bin/env python

import os
from numpy import sqrt,arcsin,arctan
import logging

class Params():

    def __init__(self):
        """
        
        """

        # ******************************************************************
        # Basic resolution/configuration parameters
        # To make this run faster, decrease pxprlens and/or nlens.
        # Decreasing nlens reduces number of lenslets/FOV
        # Reducing pxprlens below ~60 introduces aliasing (don't do this)
        # Should generally leave n_pupil = 1000 (doesn't matter that much)
        # ******************************************************************

        self.n_pupil = 1024        # Number of pixels for pupil plane
        self.pxprlens = 60        # Pixels (not detector pixels!) per lenslet
        self.nlens = 108           # Number of lenslets across array

        # ******************************************************************
        # Detector stuff
        # ******************************************************************
        
        self.pixsize = 13e-6       # Pixel size (meters)

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
        
