#!/usr/bin/env python

'''
Heritage: CHARIS simulator by T. Brandt and his team
Code was adapted to match PISCES/WFIRST IFS characteristics
'''


import numpy as np
from astropy.io import fits as pyf
from params import Params
import tools
import time
#from makeheader import Makeheader
import logging as log
from scipy import signal,ndimage
import matplotlib.pyplot as plt
from tools.image import Image
from tools.rotate import RotateAndScale,Rotate
from tools.detutils import frebin
import glob
import re
import codecs

def propSingleWavelength(imagePlane,lam,dlam,par,allweights,kernels,locations):
    
    """
    Single wavelength propagation
    
    Propagate a single wavelength through the IFS
    
    Inputs:
    1. imageplane:          complex 2D array representing the image at the lenslet array
    2. lam:                 wavelength (microns)
    3. dlam:                delta wavelength (microns)
    4. par:                 parameters instance
    """

    
    log.info('Computing detector plane for wavelength %.3f and bandwidth %.3f (microns)' % (lam,dlam))
    
    hdu = pyf.PrimaryHDU()
    header = hdu.header

    #Image(data=imagePlaneRot2).write('imagePlaneRot2_%.2fum.fits' % (lam))
    
    ###################################################################### 
    # Go through the lenslet array.  We will then be in the detector
    # plane (more or less), we just have to do some convolutions and 
    # treat the dispersion as a wavelength-dependent translation.
    ###################################################################### 
    
    log.info('Propagate through lenslet array')

    # first, do some treatment of the input image:
    # rotate the image by the lenslet array angle
    # supersample the image to match lenslet array, perhaps some multiplier of that
    # use map_coordinates, I guess
    # at the end of this, we have an array corresponding to the flux for each lenslet
    
    lensletplane = tools.Lenslet(par, imagePlane, lam, dlam,allweights,kernels,locations)
    Image(data=lensletplane).write('lensletplane_%.2fum.fits' % (lam))

    ###################################################################### 
    # Spectrograph include the dispersion and distortion.  It outputs the
    # convolution kernels to be applied with the detector resampling.
    ###################################################################### 
    
#     t['Start spectrograph']=time.time()
#     log.info('Propagate through spectrograph')
#     pinholeplane,  kernels, locations = tools.Spectrograph(par, lensletplane, lam)
#     Image(data=pinholeplane).write('pinholeplane_%.2fum.fits' % (lam))
    
    ###################################################################### 
    # Convolve with detector response
    ###################################################################### 
#     t['Start detector']=time.time()
#     log.info('Convolve with detector response')
#     finalframe = tools.Detector(par, pinholeplane, kernels, locations)
    #finalframe = tools.simpleDetector(par,lensletplane)
    
    #t['Start write to fits']=time.time()

    #head = Makeheader(par)
#     Image(data=finalframe).write('test_image_%.2fum.fits' % (lam))
    
#     string = "Performance:"
#     string += "\n  %d seconds for initial calculations" % (t['Start lenslets'] - t['Start program'])
#     string += "\n  %d seconds for lenslet transforms" % (t['Start spectrograph'] - t['Start lenslets'])
#     string += "\n  %d seconds for spectrograph distortion, dispersion" % (t['Start detector'] - t['Start spectrograph'])
#     string += "\n  %d seconds for convolutions, detector binning" % (t['Start write to fits'] - t['Start detector'])
#     string += "\n  %d seconds total" % (t['Start write to fits'] - t['Start program'])    
#     log.info(string)
     
    # return array
    return lensletplane,header


    
def main():

    tools.initLogger('IFS.log')
    
    log.info('Starting computation.')
    
    ###################################################################### 
    # The code is organized as follows:
    # 1. Rotate the original image/cube
    # 2. Project the image/cube onto the lenslet array (rebin)
    # 3. Apply distortion, dispersion
    # 4. Convolve with PSFs at the detector array
    # 5. Add detector noise
    ###################################################################### 
    
    # inputs should be an image cube, and a list of wavelengths for each cube slice
    # first dimension is the number of wavelengths
    # ideally spacing between wavelengths is smaller than IFS resolving power
    wavelist = [0.7]#[0.800,0.820,0.840]
    imageplane = np.ones((len(wavelist),512,512))
    dlam = 0.1
    nframes = imageplane.shape[0]
    
    if nframes != len(wavelist):
        log.error('Number of wavelengths does not match the number of slices')
    
    frameList = []
    
    
    par = Params()
    log.debug('Parameters initialized:'+par.output())
    
    pixperlenslet = par.pitch//par.mmperpix
    
    allweights = None
    
    ###################################################################### 
    # Defines an array of times for performance monitoring
    ###################################################################### 
    t = {'Start program':time.time()}
    t['Start lenslets'] = time.time()
    
    log.info('Import all kernels and rescale them to same plate scale')
    kernels890,locations = loadKernels(par,890)
    numpix = kernels890[0].shape[0]
    kernels770,loc = loadKernels(par,770,numpix)
    kernels660,loc = loadKernels(par,660,numpix)
    refWaveList = [660,770,890]
    kernelList = np.array([kernels660,kernels770,kernels890])
    
    for i in range(nframes):
        
        lam = wavelist[i]
        log.info('Processing wavelength %f (%d out of %d)' % (lam,i,nframes))
        
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
            
        
        ###################################################################### 
        # First, rotate and scale the image so that it is in the same 
        # orientation and scale as the lenslet array
        # After this step, the pixels in the array each represents a lenslet
        ###################################################################### 
        log.info('Rotate and scale slice %d' % i)
        #imagePlaneRot2 = RotateAndScale(imagePlane,-par.philens,par,clip=False)
        imagePlane = imageplane[i]
        imagePlaneRot = Rotate(imagePlane,-par.philens,clip=False)
        newShape = (imagePlaneRot.shape[0]/pixperlenslet,imagePlaneRot.shape[1]/pixperlenslet)
        imagePlaneRot = frebin(imagePlaneRot,newShape)
        log.info('Input plane is %dx%d' % imagePlaneRot.shape)
        Image(data=imagePlaneRot).write('imagePlaneRot_%.2fum.fits' % (lam))
    
        log.info('Calculating/loading weights for bilinear interpolation')
        if allweights==None:
            log.info('Allweights array was never created. Creating now.')
            log.debug('Constructing location array for the first time')
            lensletplane = np.zeros((par.pxprlens*(par.nlens + 2), par.pxprlens*(par.nlens + 2)))
            allweights = createAllWeightsArray(imagePlaneRot,locations)
            
        frame,header = propSingleWavelength(imagePlaneRot,lam,dlam,par,allweights,kernels,locations)
        frameList.append(frame)
        
    finalFrame = np.sum(np.asarray(frameList),axis=0)
    
    detpixprlenslet = par.pitch/par.pixsize 
    log.info('Number of detector pixels per lenslet: %f' % detpixprlenslet)
    
    #log.info('Convolve with pixel response function (check validity of this)')
    #n = int(par.pxprlens/detpixprlenslet)
    #pixresponse = np.ones((n, n))*1./n**2
    #finalFrame = signal.convolve2d(finalFrame, pixresponse)
    
    newShape = (finalFrame.shape[0]//(par.pxprlens/detpixprlenslet),finalFrame.shape[1]//(par.pxprlens/detpixprlenslet))
    log.info('Rebinning final detector. Image has dimensions %dx%d' % newShape)
    finalFrame = frebin(finalFrame,newShape) 

    Image(data=finalFrame).write('finalframe.fits') 
    log.info('Done.')
    log.shutdown()

def createAllWeightsArray(plane,locations):

    # requires square array for now
    npix = plane.shape[0]
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



def loadKernels(par,wavel,numpix=None):
    
    log.info('Loading spot diagrams.')
    # first, select which wavelength PSF to use
    spotfields = glob.glob(par.prefix + '/simpsf/%dPSF_*.fits' % wavel)
    spotsizefiles = glob.glob(par.prefix+'/simpsf/%d_*.txt' % wavel)
    kernels = [pyf.open(ifile)[1].data for ifile in spotfields]
    locations = np.zeros((len(spotfields), 2))

    if len(spotfields) != len(spotsizefiles):
        log.error('Number of kernels should match number of textfiles')
    
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
    log.info('kernel scale average is %f micron per pixel at %d nm' % (kernelscale,wavel))
    kernelscale *= 1e-6
    
    for i in range(len(spotfields)):
        name = re.sub('.fits', '', re.sub('.*PSF_', '', spotfields[i]))
        locations[i, 0] = float(name.split('_')[0])
        locations[i, 1] = float(name.split('_')[1])

    locations /= 2.
    locations += 0.5

    log.info('Resampling kernels to match input')
    plateScale = par.pitch/par.pxprlens
    log.info('Incoming plate scale is %f micron per pixel' % (plateScale*1e6))
    for i in range(len(locations)):

        ##################################################################
        # Now resample the kernels to the appropriate resolution
        ##################################################################
        
        
        # the scale in the incoming plane is par.pitch/par.pxprlens
        # the scale in the kernels is kernelscale
        # remap kernel to match incoming plate scale
        
        ratio = kernelscale/plateScale
        nx = kernels[i].shape[0] * ratio
        ny = kernels[i].shape[1] * ratio
        
        x = (np.arange(nx) - nx//2)/ratio 
        y = (np.arange(ny) - ny//2)/ratio

        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        x = r*np.cos(theta + par.philens) + kernels[i].shape[0]//2
        y = r*np.sin(theta + par.philens) + kernels[i].shape[1]//2

        kernels[i] = ndimage.map_coordinates(kernels[i], [y, x])
    
    if numpix!=None:
        newkernels = np.zeros((len(kernels),numpix,numpix))
        for k in range(len(locations)):
            newkernels[k,numpix//2-kernels[k].shape[0]//2:numpix//2+kernels[k].shape[0]//2, \
                numpix//2-kernels[k].shape[1]//2:numpix//2+kernels[k].shape[1]//2] += kernels[k]
    else:
        newkernels = kernels
        
        
    for k in range(len(locations)):
        newkernels[k] /= np.sum(newkernels[k])
        
        if par.pinhole:
#             x = range(len(kernels[k]))
#             x -= np.median(x)
#             x, y = np.meshgrid(x, x)
            if kernels[k].shape[0]<par.pxprlens+par.pin_dia/plateScale:
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
                newkernels[k] *= mask[mx-xc:mx+xc,my-yc:my+yc]
            else:
                newkernels[k,xc-mc:mx+xc,yc-mc:my+yc] *= mask[mx-xc:mx+xc,my-yc:my+yc]

    return newkernels,locations




if __name__ == '__main__':
    main()
    
    

