#!/usr/bin/env python

'''
Standalone IFS simulation code
MJ Rizzo and the IFS team

Originally inspired by T. Brandt's code for CHARIS
'''


import numpy as np
from params import Params
from astropy.io import fits as pyf
import time
import logging
import matplotlib.pyplot as plt
import tools
from tools.image import Image
from tools.lenslet import Lenslets,processImagePlane
from tools.spectrograph import createAllWeightsArray,selectKernel,loadKernels
from tools.detector import rebinDetector
from tools.initLogger import initLogger
from tools.plotting import plotKernels
from tools.reduction import simpleReduction,densifiedSimpleReduction,testReduction,lstsqExtract,intOptimalExtract,GPImethod2
import multiprocessing
from tools.par_utils import Task, Consumer

log = logging.getLogger('main')

def propagateSingleWavelength(par,i,wavelist,refWaveList,kernelList,interpolatedInputCube,allweights,locations,finalFrame):
    '''
    Propagates a single wavelength through the Integral Field Spectrograph
    
    Parameters
    ----------
    par :   Parameter instance
            with at least the key IFS parameters, interlacing and scale
    i : int
            Slice number within the cube
    wavlist : list of floats
            List of wavelengths in microns
    refWaveList : list of floats
            List of wavelengths in microns that correspond to the loaded kernels
            representing the PSF at each wavelength
    kernelList : list of 3D arrays
            List of kernels representing the wavelength for each reference wavelength and
            each field location
    wavlist : list of floats
            List of wavelengths in microns
    interpolatedInputCube : 3D ndarray
            Represents the input cube already pre-processed to match the desired wavelist,
            spatial sampling, orientation, etc.
    allweights : 3D ndarray
            Weights for bilinear interpolation between kernels in the final image. This is
            passed as an argument to avoid having to recalculate them each time.
    locations : 3D ndarray
            Detector fractional locations corresponding to each kernel. Bottom left is (0,0)
            while top right is (1,1). Same size as kernelList
    finalFrame : 2D ndarray
            Densified detector map of size par.npix*par.pxperdetpix square, which is passed
            as argument to save memory space. This is in lieu of a shared memory array that
            could be used for parallelization. The function modifies this array.
                
    '''

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
    imagePlaneRot = processImagePlane(par,interpolatedInputCube.data[i])
    if par.saveRotatedInput: Image(data=imagePlaneRot).write(par.exportDir+'/imagePlaneRot_%3.1fnm.fits' % (lam*1000.))

    ###################################################################### 
    # Generate high-resolution detector map for wavelength lam
    ###################################################################### 
#     if ~par.parallel: log.info('Propagate through lenslet array')
    Lenslets(par, imagePlaneRot, lam, allweights,kernel,locations,finalFrame)
    if par.saveLensletPlane: Image(data=finalFrame).write(par.exportDir+'/lensletPlane_%3.1fnm.fits' % (lam*1000.))
    return True

def propagateIFS(par,wavelist,inputcube,name='detectorFrame',parallel=False,cpus=6):
    '''
    Propagates an input cube through the Integral Field Spectrograph
    
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
    par.makeHeader()
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*'*60), end=True)
    par.hdr.append(('comment', '*'*22 + ' IFS Simulation ' + '*'*18), end=True)
    par.hdr.append(('comment', '*'*60), end=True)    
    par.hdr.append(('comment', ''), end=True)

    try:
        input_sampling = inputcube.header['PIXSIZE']
        input_wav = inputcube.header['LAM_C']
    except:
        log.error('Missing header information in input file')
        raise

    ###################################################################### 
    # Calculate sampling ratio to resample rotated image and match the lenslet sampling
    ###################################################################### 

    par.pixperlenslet = (par.lenslet_sampling/par.lenslet_wav)/(input_sampling/input_wav)
    log.info('The number of input pixels per lenslet is %f' % par.pixperlenslet)
    par.hdr.append(('SCALE',par.pixperlenslet,'Factor by which the input slice is rescaled'), end=True) 

#    log.info('The plate scale of the input cube is %f um/pixel' % (par.mperpix*1e6))    
    nframes = inputcube.data.shape[0]
    allweights = None
    
    if inputcube.data.shape[0] != len(wavelist):
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
                
    else:
        # This is not yet working because of shared memory issues. Need to fix it...
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

           

    ###################################################################### 
    # Rebinning to detector resolution
    ###################################################################### 
    detectorFrame = rebinDetector(par,finalFrame,clip=False)
    if par.saveDetector: Image(data=detectorFrame,header=par.hdr).write(par.exportDir+'/'+name+'.fits') 
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

    log.shutdown()

    
def reduceIFSMap(par,IFSimageName,method='optext',ivar=False):
    '''
    Main reduction function
    
    Uses various routines to extract an IFS detector map into a spectral-spatial cube.
    
    Parameters
    ----------
    par :   Parameter instance
    IFSimageName : string
            Path of image file
    method : 'simple', 'dense', 'apphot', 'test', 'lstsq', 'optext'
            Method used for reduction.
            'simple': brute force photometry, adds up the fluxes in a line of pixels
            centered where the centroid at that wavelength falls. Not very accurate.
            'dense': does the same but after interpolating/densifying the image by a factor of 10,
            which gives the previous method more accuracy (but depends on the interpolation scheme).
            'test': use the method defined in the testReduction function, for experimenting.
            'lstsq': use the knowledge of the PSFs at each location and each wavelength and fits
            the microspectrum as a weighted sum of these PSFs in the least-square sense. Can weigh the data by its variance.
            'optext': use a matched filter to appropriately weigh each pixel and assign the fluxes, making use of the inverse
            wavlength calibration map. Then remap each microspectrum onto the desired wavelengths 
    ivar : Boolean
            Uses the variance information. If the original image doesn't have a variance HDU, then
            use the image itself as its own variance (Poisson noise). Default False.
         
    '''
    # reset header (in the case where we do simulation followed by extraction)
    #par.makeHeader()
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*'*60), end=True)
    par.hdr.append(('comment', '*'*22 + ' Cube Extraction ' + '*'*21), end=True)
    par.hdr.append(('comment', '*'*60), end=True)    
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('R',par.R,'Spectral resolution of final cube'), end=True) 
    par.hdr.append(('CALDIR',par.wavecalDir.split('/')[-2],'Directory in which the wavelength solution is kept'), end=True) 

    IFSimage = Image(filename = IFSimageName)
    reducedName = IFSimageName.split('/')[-1].split('.')[0]
    if method == 'simple':
        reducedName += '_red_simple'
        cube = simpleReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'dense':
        reducedName += '_red_dense'
        cube = densifiedSimpleReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
#     elif method == 'apphot':
#         reducedName += '_red_apphot'
#         cube = apertureReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'test':
        reducedName += '_red_test'
        cube = testReduction(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'GPI2':
        reducedName += '_red_gpi2'
        cube = GPImethod2(par,par.exportDir+'/'+reducedName,IFSimage.data)
    elif method == 'lstsq':
        reducedName += '_red_lstsq'
        cube = lstsqExtract(par,par.exportDir+'/'+reducedName,IFSimage)
    elif method == 'optext':
        reducedName += '_red_optext'
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
    
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*'*60), end=True)
    par.hdr.append(('comment', '*'*22 + ' Innput info ' + '*'*25), end=True)
    par.hdr.append(('comment', '*'*60), end=True)    
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('INSLICES',len(wavelist),'Number of wavelengths in input cube'), end=True) 

#    par.hdr.append(('INTERPSL',len(wavinterp),'Number of wavelengths in interpolated input cube'), end=True) 

    outcube = Image(data=inputcube.data,header=inputcube.header)
    return wavelist,outcube

if __name__ == '__main__':
    main()

