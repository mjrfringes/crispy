#!/usr/bin/env python

'''
Standalone IFS simulation code
MJ Rizzo and the IFS team

Originally inspired by T. Brandt's code for CHARIS
'''


import numpy as np
from params import Params
from astropy.io import fits as pyf
#import tools
import time
import logging as log
import matplotlib.pyplot as plt
import tools
from tools.image import Image
from tools.lenslet import propagate,processImagePlane
from tools.spectrograph import createAllWeightsArray,selectKernel,loadKernels
from tools.detector import rebinDetector
from tools.initLogger import initLogger
from tools.plotting import plotKernels
from tools.reduction import simpleReduction,apertureReduction,densifiedSimpleReduction

def propagateIFS(par,wavelist,inputcube):
    '''
    takes in a parameter class, a list of wavelengths, and a cube for which each slice
    represents the PSF at a different wavelength
    
    Inputs:
    1. par          Parameter instance with at least the key IFS parameters, interlacing and scale
    2. lamlist      list of wavelengths in microns
    
    '''
    
    log.info('The number of input pixels per lenslet is %f' % par.pixperlenslet)    
    log.info('The plate scale of the input cube is %f um/pixel' % (par.mperpix*1e6))    
    nframes = inputcube.shape[0]
    allweights = None
    
    if inputcube.shape[0] != len(wavelist):
        log.error('Number of wavelengths does not match the number of slices')

    ###################################################################### 
    # Create cube that is interpolated to the correct level if necessary
    ###################################################################### 
    waveList,interpolatedInputCube = prepareCube(par,wavelist,inputcube)

    ###################################################################### 
    # Defines an array of times for performance monitoring
    ###################################################################### 
    t = {'Start':time.time()}

    ###################################################################### 
    # Load kernels from Zemax
    ###################################################################### 
    
    log.info('Import all kernels and rescale them to same plate scale')
    kernels890,locations = loadKernels(par,890)
    kernels770,loc = loadKernels(par,770)
    kernels660,loc = loadKernels(par,660)
    refWaveList = [660,770,890]
    kernelList = np.array([kernels660,kernels770,kernels890])

    ###################################################################### 
    # Creating kernel weight map (bilinear interpolation)
    ###################################################################### 
    allweights = createAllWeightsArray(par,locations)
    
    ###################################################################### 
    # Allocate an array
    ###################################################################### 
    finalFrame=np.zeros((par.npix*par.pxperdetpix,par.npix*par.pxperdetpix))

    log.info('Small pixels per lenslet: %f' % (par.pxprlens))    
    log.info('Final detector pixel per lenslet: %f' % (par.pxprlens/par.pxperdetpix))
    
    for i in range(len(waveList)):
        lam = wavelist[i]
        log.info('Processing wavelength %f (%d out of %d)' % (lam,i,nframes))        
        ###################################################################### 
        # Interpolate kernel at wavelength lam
        ###################################################################### 
        kernel = selectKernel(par,lam,refWaveList,kernelList)
    
        ###################################################################### 
        # Rotate and scale the image so that it is in the same 
        # orientation and scale as the lenslet array
        # After this step, the pixels in the array each represents a lenslet
        ###################################################################### 
        log.info('Rotate and scale slice %d' % i)
        imagePlaneRot = processImagePlane(par,interpolatedInputCube[i])
        if par.saveRotatedInput: Image(data=imagePlaneRot).write(par.exportDir+'/imagePlaneRot_%.3fum.fits' % (lam))

        ###################################################################### 
        # Generate high-resolution detector map for wavelength lam
        ###################################################################### 
        log.info('Propagate through lenslet array')
        propagate(par, imagePlaneRot, lam, allweights,kernel,locations,finalFrame)
            
    if par.saveLensletPlane: Image(data=finalFrame).write(par.exportDir+'/lensletPlane_%.3fum.fits' % (lam))

    ###################################################################### 
    # Rebinning to detector resolution
    ###################################################################### 
    detectorFrame = rebinDetector(par,finalFrame,clip=False)
    if par.saveDetector: Image(data=detectorFrame).write(par.exportDir+'/detectorFrame.fits') 
    log.info('Done.')
    t['End'] = time.time()
    log.info("Performance: %d seconds total" % (t['End'] - t['Start']))

    return detectorFrame

def main():

    ###################################################################### 
    # Load parameters from params.py
    ###################################################################### 
    par = Params()

    ###################################################################### 
    # Initialize logging function; both to console and to file
    ###################################################################### 
    tools.initLogger(par.exportDir+'/IFS.log')
    
    ###################################################################### 
    # Load input
    ###################################################################### 
    log.info('Loading input')
#     wavelist = np.arange(0.6,0.7,0.005) #[0.800,0.820,0.840]
#     wavelist = [0.7] #[0.800,0.820,0.840]
#     inputcube = np.ones((len(wavelist),512,512),dtype=float)/9.
#     mperpix = 58e-6
    BW = 0.18
    Nlam = 51
    clam = 0.77
    wavelist= clam*np.linspace(1.-BW/2.,1.+BW/2.,Nlam)
    fname = './Inputs/PSF_SPLC_Nwvl51_BW18pct_star.fits'
    hdu = pyf.open(fname)
    inputcube = hdu[0].data    
    mperpix = 3./5.*par.pitch # 5 pixels per lambda/D
    par.pixperlenslet = par.pitch/mperpix
    par.mperpix = mperpix
    
    propagateIFS(par,wavelist,inputcube)


    
def reduceIFSMap(par,IFSimageName,method='simple'):

    hdulist = pyf.open(IFSimageName,ignore_missing_end=True)
    if hdulist[0].header['NAXIS']!=2:
        IFSimage = pyf.open(IFSimageName,ignore_missing_end=True)[1].data
    else:
        IFSimage = pyf.open(IFSimageName,ignore_missing_end=True)[0].data
    IFSimage +=1.
    reducedName = IFSimageName.split('/')[-1].split('.')[0]
    if method == 'simple':
        reducedName += '_red_simple'
        simpleReduction(par,par.exportDir+'/'+reducedName,IFSimage)
    elif method == 'dense':
        reducedName += '_red_dense'
        densifiedSimpleReduction(par,par.exportDir+'/'+reducedName,IFSimage)
    elif method == 'apphot':
        reducedName += '_red_apphot'
        apertureReduction(par,par.exportDir+'/'+reducedName,IFSimage)

    
def prepareCube(par,wavelist,inputcube):
    '''
    Takes an input cube and interpolates it down to the required
    spectral resolution
    For now, just returns the inputs with no modification
    Watch out for energy conservation!!
    '''
    
    return wavelist,inputcube


