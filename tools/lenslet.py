#!/usr/bin/env python

import numpy as np
from astropy.io import fits as pyf
from scipy import signal
import multiprocessing
from ft import FT
from rotate import Rotate
from lensrow import LensRow
from parallel_utils import Task, Consumer
import logging as log
import matplotlib.pyplot as plt
from detutils import frebin
import glob
import re
from scipy import ndimage
import codecs
import pickle


def processImagePlane(par,imagePlane):
    imagePlaneRot = Rotate(imagePlane,par.philens,clip=False)
    n = par.pixperlenslet
    newShape = (imagePlaneRot.shape[0]/n,imagePlaneRot.shape[1]/n)
    imagePlaneRot = frebin(imagePlaneRot,newShape)
    log.debug('Input plane is %dx%d' % imagePlaneRot.shape)
    return imagePlaneRot


def Lenslet(par, imageplane, lam, allweights,kernels,locations):
    """
    Function Lenslet

    Apply a Fourier transform to each lenslet's patch of the incident 
    electric field.  If specified in input parameters, apply a convolution
    to account for each lenslet's defocus.  Otherwise, account for this
    later when taking care of the other convolutions in the detector and 
    rebinning step.  Also, update the transmission and background lists
    by appending values for new optics.

    Inputs:
    1. par:           parameters class
    2. image:         image plane incident on lenslets (complex E-field)
    3. lam:           wavelength (microns)
    4. dlam:          delta wavelength (microns)
    
    Outputs:
    1. lensletplane:  image plane after lenslet (array of PSF-lets)

    """

    n = par.pxprlens
    
    #TODO: reduce the lenslet plane only to what is needed, if there are zeros in most locations
    # on the input
    lensletplane = np.zeros((n*(par.nlens + 2), n*(par.nlens + 2)))
    
    # select row values
    nx,ny = imageplane.shape
    rowList = np.arange(-nx//2,nx//2)
    colList = np.arange(-ny//2,ny//2)

    # simplified distortion and dispersion
    cx = np.zeros(1)
    cy = np.zeros(4)
    cx[0]=-0.96764187
    #cx[1]=-0.063628939
    cy[0]=-2.962499600000000E+00
    cy[1]=-9.907069600000000E-01
    cy[2]=6.343124200000000E+00
    cy[3]=-2.979901200000000E+00

    for i in range(nx):
        for j in range(ny):
            jcoord = colList[j]
            icoord = rowList[i]
            theta = np.arctan2(jcoord,icoord)
            r = np.sqrt(icoord**2 + jcoord**2)*par.pxprlens
            # determine the coordinate of that lenslet on the pinhole mask
            #x = int(r*np.cos(theta+par.philens)+n*par.nlens//2+n//2)
            #y = int(r*np.sin(theta+par.philens)+n*par.nlens//2+n//2)
            
            # center is at zero
            x = r*np.cos(theta+par.philens)
            y = r*np.sin(theta+par.philens)
            
            # transform this coordinate including the distortion and dispersion
            factor = 1000*par.pitch/par.pxprlens
            X = x*factor # this is now in millimeters
            Y = y*factor # this is now in millimeters
            
            
            # apply polynomial transform
            sy = -np.sum(cx*np.array([Y]))/factor+lensletplane.shape[0]//2
            sx = -np.sum(cy*np.array([1,X,lam,lam**2]))/factor+lensletplane.shape[1]//2
            #sy = np.sum(cy*np.array([1,Y]))/factor+lensletplane.shape[1]//2
            
            # according to the location x,y, select the correct PSF as a combination
            # of kernels
            # use bilinear interpolation of kernels
            
            #if sx>0 and sx<lensletplane.shape[0] and sy>0 and sy<lensletplane.shape[1]:
                #lensletplane[int(sx),int(sy)]=imageplane[jcoord+imageplane.shape[0]//2,icoord+imageplane.shape[0]//2]
            kx,ky = kernels[0].shape
            if sx>kx//2 and sx<lensletplane.shape[0]-kx//2 \
                and sy>ky//2 and sy<lensletplane.shape[1]-ky//2:
                isx = int(sx)
                isy = int(sy)
                # check i vs j
                val = imageplane[jcoord+imageplane.shape[0]//2,icoord+imageplane.shape[0]//2]
                
                #checkWeight = 0.0
                for k in range(len(locations)):
                    wx = int(isx/lensletplane.shape[0]*allweights[:,:,k].shape[0])
                    wy = int(isy/lensletplane.shape[1]*allweights[:,:,k].shape[1])
                    weight = allweights[wx,wy,k]
                    #checkWeight += weight
                    lensletplane[isy-ky/2:isy+ky/2,isx-kx/2:isx+kx/2]+=val*weight*kernels[k]
                #print 'checkWeight=',checkWeight



    return lensletplane

