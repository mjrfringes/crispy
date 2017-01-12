#!/usr/bin/env python

import numpy as np
from astropy.io import fits as pyf
from tools.rotate import Rotate
import logging as log
import matplotlib.pyplot as plt
from tools.detutils import frebin
from scipy import ndimage
from tools.spectrograph import distort


def processImagePlane(par,imagePlane):
    '''
    Function processImagePlane
    
    Rotates an image or slice, and rebins in a flux-conservative way
    on an array of lenslets, using the plate scale provided in par.pixperlenslet.
    Each pixel represents the flux within a lenslet.
    
    Inputs:
    1. par:             parameters class
    2. imagePlane       2D array (image or slice)
    
    '''
    imagePlaneRot = Rotate(imagePlane,par.philens,clip=False)
    n = par.pixperlenslet
    newShape = (imagePlaneRot.shape[0]/n,imagePlaneRot.shape[1]/n)
    imagePlaneRot = frebin(imagePlaneRot,newShape)
    log.debug('Input plane is %dx%d' % imagePlaneRot.shape)
    return imagePlaneRot

    
def propagate(par, imageplane, lam, allweights,kernels,locations,lensletplane):
    """
    Function propagate
    
    Inputs:
    1. par:             parameters class
    2. image:           image plane incident on lenslets (complex E-field)
    3. lam:             wavelength (microns)
    4. allweights:      cube with weights for each kernel
    5. kernels:         kernels at locations on the detector
    6. locations:       locations where the kernels are sampled
    7. lensletplane:    densified detector plane
    
    """

    # select row values
    nx,ny = imageplane.shape
    rowList = np.arange(-nx//2,-nx//2+nx)
    colList = np.arange(-ny//2,-nx//2+nx)

    I = 64
    J = 35
    # loop on all lenslets; there's got to be a way to do this faster
    for i in range(nx):
        for j in range(ny):
            jcoord = colList[j]
            icoord = rowList[i]
            val = imageplane[jcoord+imageplane.shape[0]//2,icoord+imageplane.shape[0]//2]
            
            # exit early where there is no flux
            if val==0:
                continue
            theta = np.arctan2(jcoord,icoord)
            r = np.sqrt(icoord**2 + jcoord**2)
            x = r*np.cos(theta+par.philens)
            y = r*np.sin(theta+par.philens)
            #if i==I and j==J: print x,y
            
            # transform this coordinate including the distortion and dispersion
            factor = 1000*par.pitch
            X = x*factor # this is now in millimeters
            Y = y*factor # this is now in millimeters
            
            # apply polynomial transform
            if par.distort:
                ytmp,xtmp = distort(Y,X,lam)
                sy = -ytmp/factor*par.pxprlens+lensletplane.shape[0]//2
                sx = -xtmp/factor*par.pxprlens+lensletplane.shape[1]//2
            else:
                sy = y+lensletplane.shape[0]//2
                sx = x+lensletplane.shape[1]//2
            #if i==I and j==J: print sx/par.pxperdetpix,sy/par.pxperdetpix
            
            # put the kernel in the correct spot with the correct weight
            kx,ky = kernels[0].shape
            if sx>kx//2 and sx<lensletplane.shape[0]-kx//2 \
                and sy>ky//2 and sy<lensletplane.shape[1]-ky//2:
                isx = int(sx)
                isy = int(sy)
                
                for k in range(len(locations)):
                    wx = int(isx/lensletplane.shape[0]*allweights[:,:,k].shape[0])
                    wy = int(isy/lensletplane.shape[1]*allweights[:,:,k].shape[1])
                    weight = allweights[wx,wy,k]
                    if weight ==0:
                        continue
                    xlow = isy-ky/2
                    xhigh = xlow+ky
                    ylow = isx-kx/2
                    yhigh = ylow+kx
                    lensletplane[xlow:xhigh,ylow:yhigh]+=val*weight*kernels[k]

