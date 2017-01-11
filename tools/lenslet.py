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

    
def propagate(par, imageplane, lam, allweights,kernels,locations,lensletplane):
#def propagate(par, imageplane, lam, allweights,kernels,locations):
    """
    Function propagate

    
    Inputs:
    1. par:             parameters class
    2. image:           image plane incident on lenslets (complex E-field)
    3. lam:             wavelength (microns)
    4. allweights:      cube with weights for each kernel
    5. kernels:         kernels at locations on the detector
    6. locations:       locations where the kernels are sampled
    
    Outputs:
    1. lensletplane:  image plane after lenslet (array of PSF-lets)

    """

    #lensletplane = np.zeros((par.npix*par.pxperdetpix,par.npix*par.pxperdetpix))
    
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
            val = imageplane[jcoord+imageplane.shape[0]//2,icoord+imageplane.shape[0]//2]
            if val==0:
                continue
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
                    xlow = isy-ky/2
                    xhigh = xlow+ky
                    ylow = isx-kx/2
                    yhigh = ylow+kx
                    lensletplane[xlow:xhigh,ylow:yhigh]+=val*weight*kernels[k]
                #print 'checkWeight=',checkWeight

    #return lensletplane

