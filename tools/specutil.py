#!/usr/bin/env python

import numpy as np
from scipy import interpolate, io, ndimage
import pyfits as pyf
from ft import FT
import logging

def simpleDistortion(par,imageplane,lam):
    
    N = imageplane.shape[0]

    pixelPitch = par.pitch/par.pxprlens
    fx = np.linspace(-N/2.*1000., N/2.*1000., N)*pixelPitch
    fx, fy = np.meshgrid(fx, fx)

    # subselection of coefficients which contribute the most
    cx = np.zeros(2)
    cy = np.zeros(4)

    cx[0]=-0.96764187
    cx[1]=-0.063628939
    cy[0]=-2.962499600000000E+00
    cy[1]=-9.907069600000000E-01
    cy[2]=6.343124200000000E+00
    cy[3]=-2.979901200000000E+00

    
    ones = np.ones(imageplane.shape)

    # Interpolation step
    # Use the polynomial fit that Jorge/Qian provided
    # simultaneously compute the values for the entire array
    X = 0.001*np.sum(cx[:,np.newaxis,np.newaxis]*np.array([fx,fx* lam]),axis=0)
    Y = 0.001*np.sum(cy[:,np.newaxis,np.newaxis]*np.array([1.*ones,fy,lam*ones,lam**2.*ones]),axis=0)

    # Scaling
    X /= pixelPitch
    Y /= pixelPitch
    
    # Re-center
    X += N/2.
    Y += N/2.
    
    # re-map original array to distorted & dispersed coordinates
    logging.info('Mapping to new coordinates...')
    imageplane = ndimage.map_coordinates(imageplane,[Y,X],order=1, prefilter=False)
    return imageplane


def Distortion(par, imageplane, lam):
    
    """
    Function Distortion

    Applies a distortion map to the input image using bicubic splines
    
    Inputs: 
    1. par:        parameters class
    2. imageplane: image plane after lenslet (array of PSF-lets)
    3. pixshift:   dispersion in pixels

    Outputs:
    1. imageplane: distorted, dispersed image plane  

    """

    N = imageplane.shape[0]

    pixelPitch = par.pitch/par.pxprlens
    fx = np.linspace(-N/2.*1000., N/2.*1000., N)*pixelPitch
    fx, fy = np.meshgrid(fx, fx)

    cx = np.zeros(20)
    cy = np.zeros(20)
    cx[0]=-0.00000000036463819
    cx[1]=-0.96764187
    cx[2]=2.9532635E-14
    cx[3]=0.0000000016635338
    cx[4]=-2.0986347E-15
    cx[5]=0.000117021
    cx[6]=-2.2672132E-15
    cx[7]=-0.063628939
    cx[8]=-8.187448500000000E-14
    cx[9]=-2.524745200000000E-09
    cx[10]=-3.346803500000000E-04
    cx[11]= 3.312787500000000E-17
    cx[12]=-3.582555400000000E-04
    cx[13]= 2.204885100000000E-17
    cx[14]= 3.043899600000000E-15
    cx[15]=-1.187753400000000E-04
    cx[16]= 3.301599300000000E-15
    cx[17]= 4.349654500000000E-02
    cx[18]= 5.381647600000000E-14
    cx[19]= 1.274761900000000E-09
    cy[0]=-2.962499600000000E+00
    cy[1]=-7.983890700000000E-14
    cy[2]=-9.907069600000000E-01
    cy[3]=6.343124200000000E+00
    cy[4]=-2.579091100000000E-03
    cy[5]=-5.548179600000000E-16
    cy[6]=-3.231052700000000E-03
    cy[7]=2.416302700000000E-13
    cy[8]=-2.116432700000000E-02
    cy[9]=-2.979901200000000E+00
    cy[10]=8.941723000000000E-18
    cy[11]=-3.690345100000000E-04
    cy[12]=1.272463800000000E-17
    cy[13]=-3.495699500000000E-04
    cy[14]=-8.848836700000000E-05
    cy[15]=7.928802600000000E-16
    cy[16]=-9.424257500000000E-05
    cy[17]=-1.844725700000000E-13
    cy[18]=2.163655100000000E-02
    cy[19]=2.627917300000000E-01

    ones = np.ones(imageplane.shape)

    # Interpolation step
    # Use the polynomial fit that Jorge/Qian provided
    # simultaneously compute the values for the entire array
    staticArray = np.array([1.*ones,fx,fy,lam*ones,fx**2.,fx*fy,fy**2.,fx* lam,fy* lam, lam**2.*ones,fx**3.,fx**2.*fy,fx*fy**2.,fy**3.,fx**2.* lam,fx*fy* lam,fy**2.* lam,fx* lam**2.,fy* lam**2., lam**3.*ones])
    X = 0.001*np.sum(cx[:,np.newaxis,np.newaxis]*staticArray,axis=0)
    Y = 0.001*np.sum(cy[:,np.newaxis,np.newaxis]*staticArray,axis=0)

    # Scaling
    X /= pixelPitch
    Y /= pixelPitch
    
    # Re-center
    X += N/2.
    Y += N/2.
    
    # re-map original array to distorted & dispersed coordinates
    logging.info('Mapping to new coordinates...')
    imageplane = ndimage.map_coordinates(imageplane,[Y,X],order=1, prefilter=False)
    return imageplane

    
def SpecWFE(par, lam):
    
    """
    Function SpecWFE

    Computes the spot diagram corresponding to the wavefront map
    appropriate for the input lam.
    
    Inputs: 
    1. par:         parameters class
    2. lam:         lam (microns)

    Outputs:
    1. FTwavefront: spot diagram/intensity map suitable for convolutions

    """

    inputF = io.loadmat(par.prefix + '/SpotDiagrams/Spectrograph/wavefronts_b.mat')
    onaxis = inputF['onaxis']
    offaxis = inputF['offaxis']
    wavefront = 1.4*(onaxis + offaxis)

    ###################################################################### 
    # Not sure what this F number is for (i.e. which optic), or what 
    # the rest of these are.  They were copied from matlab; maybe Mary
    # Anne can clean this up and/or comment it better.
    ###################################################################### 
    
    Fnum = 9.4
    maparea = float(inputF['d'])
    f = maparea*Fnum
    dim = max(wavefront.shape)
    mag = 5e-3
    pix = int(maparea*mag/(2*par.pitch/93) + 0.5)*2 - 1
    
    FTwavefront = FT(wavefront, f, maparea, maparea, mag*maparea, mag*maparea, lam*1e-6, pix, pix)

    FTwavefront = np.abs(FTwavefront)**2
    FTwavefront /= np.sum(FTwavefront)

    return FTwavefront

