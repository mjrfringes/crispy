#!/usr/bin/env python

import numpy as np
try:
    from astropy.io import fits as pyf
except:
    import pyfits as pyf
from tools.rotate import Rotate
from tools.initLogger import getLogger
log = getLogger('crispy')
import matplotlib.pyplot as plt
from tools.detutils import frebin
from scipy import ndimage
from tools.spectrograph import distort
from tools.locate_psflets import initcoef,transform


def processImagePlane(par,imagePlane):
    '''
    Function processImagePlane
    
    Rotates an image or slice, and rebins in a flux-conservative way
    on an array of lenslets, using the plate scale provided in par.pixperlenslet.
    Each pixel represents the flux within a lenslet. Starts by padding the original
    image to avoid cropping edges when rotating. This step necessarily involves an
    interpolation, so one needs to be cautious.
    
    Parameters
    ----------
    par :   Parameters instance
            Contains all IFS parameters
    imagePlane : Image instance containing 3D input cube
            Input cube to IFS sim, first dimension of data is wavelength

    Returns
    -------
    imagePlaneRot : 2D array
            Rotated image plane on same sampling as original.
    '''
    
    paddedImagePlane = np.zeros((int(imagePlane.shape[0]*np.sqrt(2)),int(imagePlane.shape[1]*np.sqrt(2))))
    
    xdim,ydim = paddedImagePlane.shape
    xpad = xdim-imagePlane.shape[0]
    ypad = ydim-imagePlane.shape[1]
    xpad //=2
    ypad //=2
    paddedImagePlane[xpad:-xpad,ypad:-ypad] = imagePlane
    
    imagePlaneRot = Rotate(paddedImagePlane,par.philens,clip=False)
    
    ###################################################################### 
    # Flux conservative rebinning
    ###################################################################### 
    newShape = (int(imagePlaneRot.shape[0]/par.pixperlenslet),int(imagePlaneRot.shape[1]/par.pixperlenslet))
    imagePlaneRot = frebin(imagePlaneRot,newShape)
    log.debug('Input plane is %dx%d' % imagePlaneRot.shape)
    
    return imagePlaneRot


def _psflet(par,size,y,x,lam):
    '''
    Function psflet
    
    Computes a PSFLet template to put in the right place
    
    '''
    
def Lenslets(par, imageplane, lam,lensletplane, allweights=None,kernels=None,locations=None):
    """
    Function Lenslets
    
    Creates the IFS map on a 'dense' detector array where each pixel is smaller than the
    final detector pixels by a factor par.pxperdetpix. Adds to lensletplane array to save
    memory.
    
    Parameters
    ----------
    par :   Parameters instance
            Contains all IFS parameters
    image : 2D array
            Image plane incident on lenslets.
    lam : float
            Wavelength (microns)
    lensletplane : 2D array
            Densified detector plane; the function updates this variable
    allweights : 3D array
            Cube with weights for each kernel
    kernels : 3D array
            Kernels at locations on the detector
    locations : 2D array
            Locations where the kernels are sampled
    
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
            
            if par.distortPISCES:
                # in this case, the lensletplane array is oversampled by a factor par.pxperdetpix
                theta = np.arctan2(jcoord,icoord)
                r = np.sqrt(icoord**2 + jcoord**2)
                x = r*np.cos(theta+par.philens)
                y = r*np.sin(theta+par.philens)
                #if i==I and j==J: print x,y
            
                # transform this coordinate including the distortion and dispersion
                factor = 1000.*par.pitch
                X = x*factor # this is now in millimeters
                Y = y*factor # this is now in millimeters
            
                # apply polynomial transform
                ytmp,xtmp = distort(Y,X,lam)
                sy = ytmp/1000.*par.pxperdetpix/par.pixsize+lensletplane.shape[0]//2
                sx = xtmp/1000.*par.pxperdetpix/par.pixsize+lensletplane.shape[1]//2
            else:
                order = 3
                dispersion = par.npixperdlam*par.R*(lam*1000.-par.FWHMlam)/par.FWHMlam
                ### NOTE THE NEGATIVE SIGN TO PHILENS
                coef = initcoef(order, scale=par.pitch/par.pixsize, phi=-par.philens, x0=0, y0=dispersion)
                sy, sx = transform(i-nx//2, j-nx//2, order, coef)
                sx+=par.npix//2
                sy+=par.npix//2
                
            
            if not par.gaussian:
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
                        xlow = isy-ky//2
                        xhigh = xlow+ky
                        ylow = isx-kx//2
                        yhigh = ylow+kx
                        lensletplane[xlow:xhigh,ylow:yhigh]+=val*weight*kernels[k]
            else:
                size = int(3*par.pitch/par.pixsize)
                if sx>size//2 and sx<lensletplane.shape[0]-size//2 \
                    and sy>size//2 and sy<lensletplane.shape[1]-size//2:
                    x = np.arange(size)-size//2 
                    y = np.arange(size)-size//2 
                    _x, _y = np.meshgrid(x, y)
                    isx = int(sx)
                    isy = int(sy)
                    rsx = sx-isx
                    rsy = sy-isy
                    sig = par.FWHM/2.35
                    psflet = np.exp(-((_x- rsx)**2+(_y- rsy)**2)/(2*(sig*lam*1000/par.FWHMlam)**2))
                    psflet /= np.sum(psflet)
                    xlow = isy-size//2
                    xhigh = xlow+size
                    ylow = isx-size//2
                    yhigh = ylow+size
                    lensletplane[xlow:xhigh,ylow:yhigh]+=val*psflet

    
