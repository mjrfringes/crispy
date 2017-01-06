#!/usr/bin/env python

import numpy as np
from conv2d import Conv2d
from ft import FT
import logging

### all comments are from MjR Nov 22nd, 2016

def LensRow(par, imageplane, i, mask, lam):
    
    n = par.pxprlens
    lensletplane = np.zeros((n*3, n*(par.nlens + 2)))
    #bkgnd_lenslet = np.zeros((n*3, n*(par.nlens + 2)))
    #owa = 3*par.pitch      # Include nearest-neighbor crosstalk

    for j in range(par.nlens):
        
        ############################################################## 
        # Ignore points that will be clipped out later.  We will clip
        # by sqrt(2) to make sure the lenslets fill the detector
        # after applying the lenslet rotation.
        ############################################################## 
		
        #cen = (par.nlens - 1.)/2
        #r = np.sqrt((i - cen)**2 + (j - cen)**2)
        #phi = np.arctan2(j - cen, i - cen)
        #x, y = [r*np.cos(phi + par.philens), r*np.sin(phi + par.philens)]
        
        #if np.abs(x) > cen/np.sqrt(2) + 1 or np.abs(y) > cen/np.sqrt(2) + 1:
        #    logging.debug('continue')
        #    continue
        
        ############################################################## 
        # Fourier transform, immediately convert to intensity.
        ############################################################## 

        #logging.debug('spaxel: j,n=%d,%d,index range:i*n=%d,(i+1)*n=%d' % (j,n,j*n,(j+1)*n))
        #spaxel = imageplane[:, j*n:(j + 1)*n]
        #image = np.abs(FT(spaxel, par.flens, par.pitch, par.pitch,
        #                  owa, owa, lam*1e-6, 3*n, 3*n))**2
        #logging.debug('FT completed')
        
        #image *= np.sum(np.abs(spaxel)**2)/np.sum(image)
        
#        lensletplane[:, j*n:(j + 3)*n] += image
        lensletplane[n+n//2, (j+1)*n+n//2] += imageplane[j] # just add one pixel
        #logging.debug('Lenslet image %dx%d added with dimensions %dx%d' % (i,j,lensletplane.shape[0],lensletplane.shape[1]))
        #if par.vardefoc:
        #    bkgnd_lenslet[:, j*n:(j + 3)*n] += bkgnd_all[k]
        #else:
        #    bkgnd_lenslet[:, j*n:(j + 3)*n] += bkgnd1lens
        
    return lensletplane #[lensletplane, bkgnd_lenslet]
    
