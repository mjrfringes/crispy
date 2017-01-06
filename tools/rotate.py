#!/usr/bin/env python

import numpy as np
from scipy import ndimage

def Rotate(image, phi, clip=True,order=1):
    
    """
    Rotate the input image by phi about its center.  Do not resize the 
    image, but pad with zeros.  
    
    Inputs: 
    1. image:   2D square array
    2. phi:     rotation angle in radians
    3. clip:    boolean (optional): clip array by sqrt(2) to remove
                fill values?  Default True.

    Outputs:
    rotated image of the same shape as the input image, with zero-padding

    """

    x = np.arange(image.shape[0])
    med_n = np.median(x)
    x -= int(med_n)	
    x, y = np.meshgrid(x, x)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    x = r*np.cos(theta + phi) + med_n
    y = r*np.sin(theta + phi) + med_n

    imageout = ndimage.map_coordinates(image, [y, x], order=order, prefilter=False)
        
    if clip:
        i = int(imageout.shape[0]*(1. - 1./np.sqrt(2.))/2.)
        imageout = imageout[i:-i, i:-i]
    
    return imageout


def RotateAndScale(image, phi, par, clip=False):
    
    """
    Rotate the input image by phi about its center.  
    Resize the image to match the lenslet scale.
    At the end of this, each pixel corresponds to a lenslet.
    
    Inputs: 
    1. image:   2D square array
    2. phi:     rotation angle in radians
    3. par:     parameter instance containing the scaling factor
    4. clip:    boolean (optional): clip array by sqrt(2) to remove
                fill values?  Default True.

    Outputs:
    rotated and scaled image

    """

    pixperlenslet = par.pitch//par.mmperpix  # e.g. if there are 3 input pixel per lenslet, then the resulting image should be 3 times smaller
    
    x = np.arange(image.shape[0]//pixperlenslet)
    med_n = np.median(x)
    x -= int(med_n)
    x *= pixperlenslet
    x, y = np.meshgrid(x, x)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    x = r*np.cos(theta + phi) + image.shape[0]//2
    y = r*np.sin(theta + phi) + image.shape[0]//2

    imageout = ndimage.map_coordinates(image, [y, x], order=1, prefilter=False)
        
    if clip:
        i = int(imageout.shape[0]*(1. - 1./np.sqrt(2.))/2.)
        imageout = imageout[i:-i, i:-i]
    
    return imageout
