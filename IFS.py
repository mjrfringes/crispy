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
from tools.reduction import simpleReduction,apertureReduction,densifiedSimpleReduction,testReduction,lstsqExtract,intOptimalExtract
import multiprocessing
from tools.par_utils import Task, Consumer


def propagateSingleWavelength(par,i,wavelist,refWaveList,kernelList,interpolatedInputCube,allweights,locations,finalFrame):

    lam = wavelist[i]
    
#     if ~par.parallel: log.info('Processing wavelength %f (%d out of %d)' % (lam,i,nframes))        
    ###################################################################### 
    # Interpolate kernel at wavelength lam
    ###################################################################### 
    kernel = selectKernel(par,lam,refWaveList,kernelList)

    ###################################################################### 
    # Rotate and scale the image so that it is in the same 
    # orientation and scale as the lenslet array
    # After this step, the pixels in the array each represents a lenslet
    ###################################################################### 
#     if ~par.parallel: log.info('Rotate and scale slice %d' % i)
    imagePlaneRot = processImagePlane(par,interpolatedInputCube[i])
    if par.saveRotatedInput: Image(data=imagePlaneRot).write(par.exportDir+'/imagePlaneRot_%3.1fnm.fits' % (lam*1000.))

    ###################################################################### 
    # Generate high-resolution detector map for wavelength lam
    ###################################################################### 
#     if ~par.parallel: log.info('Propagate through lenslet array')
    propagate(par, imagePlaneRot, lam, allweights,kernel,locations,finalFrame)
    if par.saveLensletPlane: Image(data=finalFrame).write(par.exportDir+'/lensletPlane_%3.1fnm.fits' % (lam*1000.))
    return True

def propagateIFS(par,wavelist,inputcube,name='detectorFrame',parallel=True,cpus=6):
    '''
    takes in a parameter class, a list of wavelengths, and a cube for which each slice
    represents the PSF at a different wavelength
    
    Parameters
    ----------
    par :   Parameter instance
            with at least the key IFS parameters, interlacing and scale
    lamlist : list of floats
            List of wavelengths in microns
    inputcube : 3D ndarray
            First dimension needs to be the same length as lamlist
                
    Returns
    -------
    detectorFrame : 2D array
            Return the detector frame
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
    
    if not parallel:
        for i in range(len(waveList)):
            log.info('Processing wavelength %f (%d out of %d)' % (waveList[i],i,nframes))
            propagateSingleWavelength(par,i,wavelist,refWaveList,kernelList,interpolatedInputCube,allweights,locations,finalFrame)
#             lam = wavelist[i]
#             
#             log.info('Processing wavelength %f (%d out of %d)' % (lam,i,nframes))        
#             ###################################################################### 
#             # Interpolate kernel at wavelength lam
#             ###################################################################### 
#             kernel = selectKernel(par,lam,refWaveList,kernelList)
#     
#             ###################################################################### 
#             # Rotate and scale the image so that it is in the same 
#             # orientation and scale as the lenslet array
#             # After this step, the pixels in the array each represents a lenslet
#             ###################################################################### 
#             log.info('Rotate and scale slice %d' % i)
#             imagePlaneRot = processImagePlane(par,interpolatedInputCube[i])
#             if par.saveRotatedInput: Image(data=imagePlaneRot).write(par.exportDir+'/imagePlaneRot_%.3fum.fits' % (lam))
# 
#             ###################################################################### 
#             # Generate high-resolution detector map for wavelength lam
#             ###################################################################### 
#             log.info('Propagate through lenslet array')
#             propagate(par, imagePlaneRot, lam, allweights,kernel,locations,finalFrame)
    
    else:
        log.info('Starting parallel IFS propagation! Watchout for memory...')
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        ncpus = min(multiprocessing.cpu_count(),cpus)
        consumers = [ Consumer(tasks, results)
                      for i in range(ncpus) ]
        for w in consumers:
            w.start()
    
        for i in range(len(waveList)):
            tasks.put(Task(i, propagateSingleWavelength, (par,i,waveList,refWaveList,
                            kernelList,interpolatedInputCube,allweights,locations,finalFrame)))
    
        for i in range(ncpus):
            tasks.put(None)
        for i in range(len(waveList)):
            index,result = results.get()
            log.info('Done with wavelength %.3f' % waveList[index])
#             for i in range(len(lam)):
#                 index, hiresarr = results.get()
#                 hires_arrs += [hiresarr]

           

    ###################################################################### 
    # Rebinning to detector resolution
    ###################################################################### 
    detectorFrame = rebinDetector(par,finalFrame,clip=False)
    if par.saveDetector: Image(data=detectorFrame).write(par.exportDir+'/'+name+'.fits') 
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
    initLogger(par.exportDir+'/IFS.log')
    
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


    
def reduceIFSMap(par,IFSimageName,method='apphot',ivar=False):
    '''
    Main reduction function
    
    Uses various routines to extract an IFS detector map into a spectral-spatial cube.
    
    Parameters
    ----------
    par :   Parameter instance
    IFSimageName : string
            Path of image file
    method : 'simple', 'dense', 'apphot', 'test', 'lstsq', 'opt'
            Method used for reduction.
            'simple': brute force photometry, adds up the fluxes in a line of pixels
            centered where the centroid at that wavelength falls. Not very accurate.
            'dense': does the same but after interpolating/densifying the image by a factor of 10,
            which gives the previous method more accuracy (but depends on the interpolation scheme).
            'apphot': use an aperture photometry routine from the photutils package, for better accuracy. Default setting.
            'test': use the method defined in the testReduction function, for experimenting.
            'lstsq': use the knowledge of the PSFs at each location and each wavelength and fits
            the microspectrum as a weighted sum of these PSFs in the least-square sense. Can weigh the data by its variance.
            'opt': use a matched filter to appropriately weigh each pixel and assign the fluxes, making use of the inverse
            wavlength calibration map. Then remap each microspectrum onto the desired wavelengths 
    ivar : Boolean
            Uses the variance information. If the original image doesn't have a variance HDU, then
            use the image itself as its own variance (Poisson noise). Default False.
         
    '''
    
#     hdulist = pyf.open(IFSimageName,ignore_missing_end=True)
#     if hdulist[0].header['NAXIS']!=2:
#         IFSimage = pyf.open(IFSimageName,ignore_missing_end=True)[1]
#     else:
#         IFSimage = pyf.open(IFSimageName,ignore_missing_end=True)[0]
#     IFSimage.data +=1.
    IFSimage = Image(filename = IFSimageName)
    reducedName = IFSimageName.split('/')[-1].split('.')[0]
    if method == 'simple':
        reducedName += '_red_simple'
        cube = simpleReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'dense':
        reducedName += '_red_dense'
        cube = densifiedSimpleReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'apphot':
        reducedName += '_red_apphot'
        cube = apertureReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'test':
        reducedName += '_red_test'
        cube = testReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'lstsq':
        reducedName += '_red_lstsq'
        cube = lstsqExtract(par,par.exportDir+'/'+reducedName,IFSimage,ivar)
    elif method == 'intopt':
        reducedName += '_red_intopt'
        if ivar==False: IFSimage.ivar = np.ones(IFSimage.data.shape)
        cube = intOptimalExtract(par,par.exportDir+'/'+reducedName,IFSimage)
    else:
        log.info("Method not found")
    return cube




def prepareCube(par,wavelist,inputcube):
    '''
    Takes an input cube and interpolates it down to the required
    spectral resolution
    For now, just returns the inputs with no modification
    Watch out for energy conservation!!
    '''
    
    return wavelist,inputcube

if __name__ == '__main__':
    main()

