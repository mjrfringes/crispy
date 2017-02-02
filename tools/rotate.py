#!/usr/bin/env python

import numpy as np
from scipy import ndimage

def Rotate(image, phi, clip=True,order=1):
    
    """
    Rotate the input image by phi about its center.  Do not resize the 
    image, but pad with zeros.  Function originally from Tim Brandt
    
    Parameters
    ----------
    image : 2D square array
            Image to rotate
    phi : float
            Rotation angle in radians
    clip :  boolean (optional)
            Clip array by sqrt(2) to remove fill values?  Default True.
    order : integer (optional)
            Order of interpolation when rotating. Default is 1.

    Returns
    -------
    imageout: 2D array
            Rotated image of the same shape as the input image, with zero-padding

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
