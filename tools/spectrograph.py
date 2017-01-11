#!/usr/bin/env python

import numpy as np
from scipy import interpolate, ndimage, signal
import glob
from astropy.io import fits as pyf
import re
from specutil import SpecWFE, Distortion, simpleDistortion
import logging as log
import matplotlib.pyplot as plt
import codecs

def createAllWeightsArray(par,locations):

    # requires square array for now
    npix = par.nlens
    xfrac = np.linspace(0, npix, npix)/npix
    yfrac, xfrac = np.meshgrid(xfrac, xfrac)

    allweights = np.ones((npix, npix, len(locations)))*1/0.25
    for i in range(len(locations)):
        allweights[:, :, i] *= (np.abs(xfrac - locations[i, 0]) < 0.5)
        allweights[:, :, i] *= (np.abs(yfrac - locations[i, 1]) < 0.5)

    for i in range(npix):
        for k in range(len(locations)):
            if xfrac[i, 0] > locations[k, 0]:
                allweights[i, :, k] *= locations[k, 0] + 0.5 - xfrac[i, :]
            else:
                allweights[i, :, k] *= xfrac[i, :] - (locations[k, 0] - 0.5)

    for j in range(npix):
        for k in range(len(locations)):
            if yfrac[0, j] > locations[k, 1]:
                allweights[:, j, k] *= locations[k, 1] + 0.5 - yfrac[:, j]
            else:
                allweights[:, j, k] *= yfrac[:, j] - (locations[k, 1] - 0.5)
    return allweights


def selectKernel(par,lam,refWaveList,kernelList):
    # find lowest index that is greater than lam
    ind = 0
    for nwave in range(len(refWaveList)):
        if refWaveList[ind]>=lam*1000:
            break
        else:
            ind+=1
    
    if ind==0:
        kernels = kernelList[0]
        log.warning('Wavelength out of range of reference kernels')
    elif ind==len(refWaveList):
        kernels = kernelList[-1]
        log.warning('Wavelength out of range of reference kernels')
    else:
        # ind is the index of the wavelength immediately greater than lam
        wavelUp = refWaveList[ind]
        wavelDown = refWaveList[ind-1]
        kernels = (wavelUp - lam*1000)*kernelList[ind] + (lam*1000-wavelDown)*kernelList[ind-1]
        kernels /= (wavelUp-wavelDown)
    
    if par.convolve:
        # convolve the clipped kernel by a Gaussian to simulate defocus
        # this is inaccurate but is a placeholder for using the real defocussed kernels
        # scale is (par.pixsize/par.pxperdetpix)
        # we want kernel to be ~2 detector pixel FWHM so par.pixsize/(par.pixsize/par.pxperdetpix)
        sigma = par.FWHM/2.35*par.pixsize/(par.pixsize/par.pxperdetpix)
        for k in range(kernels.shape[0]):
            kernels[k] = ndimage.filters.gaussian_filter(kernels[k],sigma,order=0,mode='constant')
    return kernels

def loadKernels(par,wavel):
    
    log.info('Loading spot diagrams.')
    # first, select which wavelength PSF to use
    spotfields = glob.glob(par.prefix + '/simpsf/%dPSF_*.fits' % wavel)
    spotsizefiles = glob.glob(par.prefix+'/simpsf/%d_*.txt' % wavel)
    kernels = [pyf.open(ifile)[1].data for ifile in spotfields]
    locations = np.zeros((len(spotfields), 2))

    if len(spotfields) != len(spotsizefiles):
        log.error('Number of kernels should match number of textfiles')
    

    ##################################################################
    # Reading kernel scale from Zemax textfile
    ##################################################################
    kernelscale = 0.0
    for spotsizefile in spotsizefiles:
        readFile = codecs.open(spotsizefile,encoding='utf-16-le')
        for i, line in enumerate(readFile):
            # physical scale is located on line 9, element 4
            if i==9:
                kernelscale += float(line.split(' ')[3])
            elif i>9:
                break
    kernelscale /= len(spotsizefiles)
    log.info('kernel scale average is %.3f micron per pixel at %d nm' % (kernelscale,wavel))
    kernelscale *= 1e-6


    ##################################################################
    # Reading locations
    ##################################################################
    for i in range(len(spotfields)):
        name = re.sub('.fits', '', re.sub('.*PSF_', '', spotfields[i]))
        locations[i, 0] = float(name.split('_')[0])
        locations[i, 1] = float(name.split('_')[1])

    locations /= 2.
    locations += 0.5

    ##################################################################
    # Resample the kernels to the appropriate resolution
    ##################################################################
    log.info('Resampling kernels to match input')
    plateScale = par.pixsize/par.pxperdetpix
    log.debug('Lenslet plane plate scale is %.3f micron per pixel' % (plateScale*1e6))
    for i in range(len(locations)):
        # the scale in the incoming plane is par.pitch/par.pxprlens
        # the scale in the kernels is kernelscale
        # remap kernel to match incoming plate scale
        
        ratio = kernelscale/plateScale
        nx = int(kernels[i].shape[0] * ratio)
        ny = int(kernels[i].shape[1] * ratio)
        
        x = (np.arange(nx) - nx//2)/ratio 
        y = (np.arange(ny) - ny//2)/ratio

        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        x = r*np.cos(theta + par.philens) + kernels[i].shape[0]//2
        y = r*np.sin(theta + par.philens) + kernels[i].shape[1]//2

        kernels[i] = ndimage.map_coordinates(kernels[i], [y, x])
    
    if hasattr(par,'pxprlens'):
        log.info('Padding smaller kernels')
        newkernels = np.zeros((len(kernels),par.pxprlens,par.pxprlens))
        for k in range(len(locations)):
            kxm = par.pxprlens//2-kernels[k].shape[0]//2
            kxp = kxm+kernels[k].shape[0]
            #print kxm,kxp,kernels[k].shape
            newkernels[k,kxm:kxp,kxm:kxp] += kernels[k]
    else:
        par.pxprlens = kernels[0].shape[0]
        log.info('pxprlens: %.3f' % par.pxprlens)
        newkernels = kernels
        
        
    for k in range(len(locations)):
        newkernels[k] /= np.sum(newkernels[k])
        #print newkernels[k].shape
        if par.pinhole:
#             x = range(len(kernels[k]))
#             x -= np.median(x)
#             x, y = np.meshgrid(x, x)
            if kernels[k].shape[0]<2*par.pxprlens+par.pin_dia/plateScale:
                log.warning('Kernel too small to capture crosstalk')
            x = np.linspace(-1.5, 1.5, 3*par.pxprlens)%1
            x[np.where(x > 0.5)] -= 1
            x, y = np.meshgrid(x, x)

            mask = 1.*(np.sqrt(x**2 + y**2) <= 0.5*par.pin_dia/par.pitch)
            xc=newkernels[k].shape[0]//2
            yc=newkernels[k].shape[1]//2
            mx = mask.shape[0]//2
            my = mask.shape[1]//2
            
            if xc<mx:
                xlow = mx-xc
                xhigh = xlow+newkernels[k].shape[0]
                ylow = my-yc
                yhigh = ylow+newkernels[k].shape[1]
                newkernels[k] *= mask[xlow:xhigh,ylow:yhigh]
            else:
                xlow = xc-mx
                xhigh = xlow+mask.shape[0]
                ylow = yc-my
                yhigh = ylow+mask.shape[1]
                newkernels[k,xlow:xhigh,ylow:yhigh] *= mask

    return newkernels,locations




