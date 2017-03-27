#!/usr/bin/env python

'''
Standalone IFS simulation code
MJ Rizzo and the IFS team

Originally inspired by T. Brandt's code for CHARIS
'''


import numpy as np
from params import Params
try:
    from astropy.io import fits as pyf
except:
    import pyfits as pyf
import time
import matplotlib.pyplot as plt
import tools
from tools.image import Image
from tools.lenslet import Lenslets,processImagePlane
from tools.spectrograph import createAllWeightsArray,selectKernel,loadKernels
from tools.detector import rebinDetector
from tools.plotting import plotKernels
from tools.reduction import simpleReduction,densifiedSimpleReduction,testReduction,lstsqExtract,intOptimalExtract,GPImethod2
import multiprocessing
from tools.par_utils import Task, Consumer

from tools.initLogger import getLogger
log = getLogger('crispy')

def propagateSingleWavelength(par,i,wavelist,interpolatedInputCube,finalFrame,
            refWaveList=None,kernelList=None,allweights=None,locations=None):
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
    if not par.gaussian:
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
    if not par.gaussian:
        Lenslets(par, imagePlaneRot, lam,finalFrame, allweights,kernel,locations)
    else:
        Lenslets(par, imagePlaneRot, lam,finalFrame)
    if par.saveLensletPlane: Image(data=finalFrame).write(par.exportDir+'/lensletPlane_%3.1fnm.fits' % (lam*1000.))
    return True


def propagateIFS(par,wavelist,inputcube,name='detectorFrame'):
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
        input_wav = inputcube.header['LAM_C']*1000.
    except:
        log.error('Missing header information in input file')
        raise

    ###################################################################### 
    # Calculate sampling ratio to resample rotated image and match the lenslet sampling
    ###################################################################### 

    par.pixperlenslet = par.lenslet_sampling/(input_sampling * input_wav/par.lenslet_wav)
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

    if not par.gaussian:
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
    else:
        log.info('Using PSFlet gaussian approximation')
        
    
    ###################################################################### 
    # Allocate an array
    ###################################################################### 
    finalFrame=np.zeros((par.npix*par.pxperdetpix,par.npix*par.pxperdetpix))

    if not par.gaussian:
        log.info('Small pixels per lenslet: %f' % (par.pxprlens))    
        log.info('Final detector pixel per lenslet: %f' % (par.pxprlens/par.pxperdetpix))
    else:
        log.info('Final detector pixel per PSFLet: %f' % (int(3*par.pitch/par.pixsize)))
    
    if not parallel:
        for i in range(len(waveList)):
            log.info('Processing wavelength %f (%d out of %d)' % (waveList[i],i,nframes))
            if not par.gaussian:
                propagateSingleWavelength(par,i,wavelist,interpolatedInputCube,finalFrame,
                                            refWaveList,kernelList,allweights,locations)
            else:
                propagateSingleWavelength(par,i,wavelist,interpolatedInputCube,finalFrame)
                
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

    
def reduceIFSMap(par,IFSimageName,method='optext'):
    '''
    Main reduction function
    
    Uses various routines to extract an IFS detector map into a spectral-spatial cube.
    
    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    IFSimageName : string
            Path of image file
    method : 'lstsq', 'optext'
            Method used for reduction.
            'lstsq': use the knowledge of the PSFs at each location and each wavelength and fits
            the microspectrum as a weighted sum of these PSFs in the least-square sense. Can weigh the data by its variance.
            'optext': use a matched filter to appropriately weigh each pixel and assign the fluxes, making use of the inverse
            wavlength calibration map. Then remap each microspectrum onto the desired wavelengths 
    
    Returns
    -------
    cube: 3D ndarray
        Reduced IFS cube
    
    '''
    start = time.time()

    # reset header (in the case where we do simulation followed by extraction)
    if 'CALDIR' in par.hdr:
        pass
    else:
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*'*60), end=True)
        par.hdr.append(('comment', '*'*22 + ' Cube Extraction ' + '*'*21), end=True)
        par.hdr.append(('comment', '*'*60), end=True)    
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('R',par.R,'Spectral resolution of final cube'), end=True) 
        par.hdr.append(('CALDIR',par.wavecalDir.split('/')[-2],'Directory of wavelength solution'), end=True) 

    IFSimage = Image(filename = IFSimageName)
    reducedName = IFSimageName.split('/')[-1].split('.')[0]
    if method == 'lstsq':
        reducedName += '_red_lstsq'
        cube = lstsqExtract(par,par.exportDir+'/'+reducedName,IFSimage)
    elif method == 'optext':
        reducedName += '_red_optext'
        cube = intOptimalExtract(par,par.exportDir+'/'+reducedName,IFSimage)
    else:
        log.info("Method not found")
        
    log.info('Elapsed time: %fs' % (time.time()-start))

    return cube


def reduceIFSMapList(par,IFSimageNameList,method='optext',parallel=True):
    '''
    Main reduction function
    
    Uses various routines to extract an IFS detector map into a spectral-spatial cube.
    
    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    IFSimageNameList : list
            List of strings containing the paths to the image files
    method : 'lstsq', 'optext'
            Method used for reduction.
            'lstsq': use the knowledge of the PSFs at each location and each wavelength and fits
            the microspectrum as a weighted sum of these PSFs in the least-square sense. Can weigh the data by its variance.
            'optext': use a matched filter to appropriately weigh each pixel and assign the fluxes, making use of the inverse
            wavlength calibration map. Then remap each microspectrum onto the desired wavelengths 
             
    '''
    # reset header (in the case where we do simulation followed by extraction)
    #par.makeHeader()
    start = time.time()

    if 'CALDIR' in par.hdr:
        pass
    else:
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*'*60), end=True)
        par.hdr.append(('comment', '*'*22 + ' Cube Extraction ' + '*'*21), end=True)
        par.hdr.append(('comment', '*'*60), end=True)    
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('R',par.R,'Spectral resolution of final cube'), end=True) 
        par.hdr.append(('CALDIR',par.wavecalDir.split('/')[-2],'Directory of wavelength solution'), end=True) 

    
    if parallel:
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        ncpus = multiprocessing.cpu_count()
        consumers = [ Consumer(tasks, results)
                      for i in range(ncpus) ]
        for w in consumers:
            w.start()

        # you call the function here, with all its arguments in a list
        for i in range(len(IFSimageNameList)):
            IFSimage = Image(filename = IFSimageNameList[i])
            reducedName = IFSimageNameIFSimageNameList[i].split('/')[-1].split('.')[0]
            elif method == 'lstsq':
                reducedName += '_red_lstsq'
                tasks.put(Task(i, lstsqExtract, (par, par.exportDir+'/'+reducedName,IFSimage)))
            elif method == 'optext':
                reducedName += '_red_optext'
                tasks.put(Task(i, intOptimalExtract, (par, par.exportDir+'/'+reducedName,IFSimage)))
            else:
                log.info("Method not found")
    
        for i in range(ncpus):
            tasks.put(None)

        for i in range(len(IFSimageNameList)):
            index, result = results.get()

    else:
        for i in range(len(IFSimageNameList)):
            IFSimage = Image(filename = IFSimageNameList[i])
            reducedName = IFSimageNameList[i].split('/')[-1].split('.')[0]
            elif method == 'lstsq':
                reducedName += '_red_lstsq'
                cube = lstsqExtract(par,par.exportDir+'/'+reducedName,IFSimage)
            elif method == 'optext':
                reducedName += '_red_optext'
                cube = intOptimalExtract(par,par.exportDir+'/'+reducedName,IFSimage)
            else:
                log.info("Method not found")

    log.info('Elapsed time: %fs' % (time.time()-start))



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

