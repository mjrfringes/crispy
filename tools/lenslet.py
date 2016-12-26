#!/usr/bin/env python

import numpy as np
from astropy.io import fits as pyf
from scipy import signal
import multiprocessing
from ft import FT
from conv2d import Conv2d
from rotate import Rotate
from lensrow import LensRow
from parallel_utils import Task, Consumer
import logging

def Lenslet(par, imageplane, lam, dlam, parallel=True,
            maxcpus=6):
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
    5. parallel:      boolean--run in parallel?  Default True
    6. maxcpus:       maximum number of cpus to use.  Default 6, only used
                      if parallel=True
    
    Outputs:
    1. lensletplane:  image plane after lenslet (array of PSF-lets)

    """

    partialdefocus = True  # Take a small shortcut with convolutions

    owa = 3*par.pitch      # Include nearest-neighbor crosstalk
    
    n = par.pxprlens

    lensletplane = np.zeros((n*(par.nlens + 2), n*(par.nlens + 2)))
    
    if par.pinhole:

        ################################################################## 
        # 3x3 pinhole grid.  First get a remainder, then treat 
        # periodicity by forcing all values to (-0.5, 0.5].
        # circular pinholes of fractional diameter par.pin_dia/par.pitch
        ################################################################## 

        x = np.linspace(-1.5, 1.5, 3*n)%1
        x[np.where(x > 0.5)] -= 1
        x, y = np.meshgrid(x, x)

        mask = 1.*(np.sqrt(x**2 + y**2) <= 0.5*par.pin_dia/par.pitch)
        logging.debug('Created 3x3 pinhole grid mask of size %dx%d' % mask.shape)
#    else:
#    	mask = np.ones(lensletplane.shape)

    ###################################################################### 
    # Background after lenslets, summed over optics and transmissions.
    ###################################################################### 
        
#     trans_arr = np.asarray(trans)
#     bkgnd_arr = np.asarray(bkgnd)
#     bkgnd_in = np.sum(bkgnd_arr*np.cumprod(trans_arr[::-1]))
# 
#     spaxel = np.ones((n, n))*bkgnd_in/n**2
#     bkgnd1lens = np.abs(FT(spaxel, par.flens, par.pitch, par.pitch, 
#                            owa, owa, lam*1e-6, 3*n, 3*n))**2
#     if par.pinhole:
#         bkgnd1lens *= mask

    ###################################################################### 
    # Now load the lenslet defocus spot diagrams.  Require wavelength to 
    # be within 0.45 microns of the actual wavelength. 
    ###################################################################### 

#     lam_nm = int(lam*10 + 0.5)*100
#     defocus = [pyf.open(par.prefix + '/SpotDiagrams/Spot%04d.fits' % (i))[0].data
#                for i in range(lam_nm - 400, lam_nm + 401, 100)]
# 
#     bkgnd_all = [Conv2d(bkgnd1lens, defocus[k]) for k in range(len(defocus))]
#     bkgnd_all = np.asarray(bkgnd_all)

    ###################################################################### 
    # Perform the lenslet Fourier transforms.  Move this into another file
    # row-by-row to allow for parallelization.
    ###################################################################### 
    
    if not parallel:
        logging.info('No parallel processing! Processing each lenslet linearly...')
        for i in range(par.nlens):
            try:
                dlenslet = LensRow(par, imageplane[i*n:(i + 1)*n],i, mask, lam)
            except:
                logging.debug('LensRow error. i,n=%d,%d,index range:i*n=%d,(i+1)*n=%d' % (i,n,i*n,(i+1)*n))
            lensletplane[i*n:(i + 3)*n] += dlenslet
    else:
        logging.info('Starting parallel processing...')

        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        ncpus = min(multiprocessing.cpu_count(), maxcpus)
        consumers = [ Consumer(tasks, results)
                      for i in range(ncpus) ]
        for w in consumers:
            w.start()
        for i in range(par.nlens):
            tasks.put(Task(i, LensRow,
                           (par, imageplane[i*n:(i + 1)*n], i,
                            mask, lam)))
        for i in range(ncpus):
            tasks.put(None)
            
        for i in range(par.nlens):
            index, result = results.get()
            dlenslet = result
            lensletplane[index*n:(index + 3)*n] += dlenslet
            
    ###################################################################### 
    # Rotate.  The pupil should have been previously rotated in the other
    # direction, making this equivalent to a rotated lenslet array.
    # Clip by sqrt(2) so that the lenslets fill the array.
    ###################################################################### 
    
    lensletplane = Rotate(lensletplane, par.philens, clip=True)
    #lensletplane = Rotate(lensletplane, par.philens)
    #bkgnd_lenslet = Rotate(bkgnd_lenslet, par.philens)

    ###################################################################### 
    # Two lenslet surfaces.  Ignore thermal emission at cryo after 
    # lenslet array.
    ###################################################################### 
    
    #trans += 2*[par.trLens]

    return lensletplane

