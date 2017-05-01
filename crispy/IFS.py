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
from tools.lenslet import Lenslets,processImagePlane,propagateLenslets
from tools.spectrograph import createAllWeightsArray,selectKernel,loadKernels
from tools.detector import rebinDetector
from tools.plotting import plotKernels
from tools.reduction import simpleReduction,densifiedSimpleReduction,testReduction,lstsqExtract,intOptimalExtract,GPImethod2
import multiprocessing
from tools.par_utils import Task, Consumer
from tools.wavecal import get_sim_hires
from scipy.interpolate import interp1d


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


def polychromeIFS(par,wavelist,inputcube,name='detectorFrame',parallel=True, QE = True, wavelist_endpts=None,dlambda=None,lam_arr=None):
    '''
    Propagates an input cube through the Integral Field Spectrograph
    
    Parameters
    ----------
    par :   Parameter instance
            with at least the key IFS parameters, interlacing and scale
    wavelist : list of floats
            List of wavelengths in nm corresponding to the center of each bin
    inputcube : Image
            or HDU. data is 3D ndarray with first dimension the same length as lamlist
            header needs to contain the 'PIXSIZE' and 'LAM_C' keywords
    name: string
            Name of the output file (without .fits extension)
    parallel: boolean
            Whether to use parallel computing for this (recommended)
    QE: boolean
            Whether to take into account wavelength-dependent detector QE (from file defined in par.QE)
    wavelist_endpts: list of floats
            List of the wavelengths in nm corresponding to the endpoints of the bins (array has to be one longer than wavelist)
    dlambda: float
            In case all the bins have the same size, use this parameter in nm. Replaces wavelist_endpts if set
    lam_arr: list of floats
            Temporary input vector of the wavelengths used to construct the polychrome. This is necessary in order to construct
            the wavelength calibration files. If the bandpass changes, one needs to pass an array of wavelengths covering the 
            new bandpass. Need to work on this.
            
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

    nframes = inputcube.data.shape[0]
    allweights = None
    
    if inputcube.data.shape[0] != len(wavelist):
        log.error('Number of wavelengths does not match the number of slices')

    ###################################################################### 
    # Create cube that is interpolated to the correct level if necessary
    ###################################################################### 
    waveList,interpolatedInputCube = prepareCube(par,wavelist,inputcube,QE=QE)

    ###################################################################### 
    # Defines an array of times for performance monitoring
    ###################################################################### 
    t = {'Start':time.time()}

    if not par.gaussian:
        log.info('Using templates PSFLets')
    else:
        log.info('Using PSFlet gaussian approximation')
        
    ###################################################################### 
    # Allocate arrays, make sure you have abundant memory
    ###################################################################### 
    finalFrame=np.zeros((par.npix*par.pxperdetpix,par.npix*par.pxperdetpix))
    polyimage=np.zeros((len(waveList), par.npix*par.pxperdetpix, par.npix*par.pxperdetpix))
    
    
    ###################################################################### 
    # Determine wavelength endpoints
    ###################################################################### 
    if wavelist_endpts is None:
        log.warning('Assuming slices are evenly spread in wavelengths')
        if len(waveList)>1:
            dlam = waveList[1]-waveList[0]
            wavelist_endpts = np.zeros(len(waveList)+1)
            wavelist_endpts[:-1] = waveList-dlam/2.
            wavelist_endpts[-1] = waveList[-1]+dlam/2.
        else:
            if dlambda is None:
                log.error('No bandwidth specified')
            else:
                wavelist_endpts=np.array([waveList[0]-dlambda/2.,waveList[0]+dlambda/2.])
    else:
        log.warning('Assuming endpoints wavelist is given')
#     print wavelist_endpts
    
    ###################################################################### 
    # Load template PSFLets
    ###################################################################### 
    if par.gaussian:
        hires_arrs = []
        if lam_arr is None:
            lam_arr=np.arange(700.,845.,10.)  # hard coded for now, need to modify this
        for i in range(len(lam_arr)):
            hiresarr = get_sim_hires(par, lam_arr[i])   
            hires_arrs += [hiresarr]
    else:
        log.error('Importing PSFLets is not yet implemented')
    
    inputCube = []
    
    if parallel==False:
        for i in range(len(waveList)):
            imagePlaneRot = (wavelist_endpts[i + 1]-wavelist_endpts[i])*processImagePlane(par,interpolatedInputCube.data[i])
            inputCube += [imagePlaneRot]
            polyimage[i] = propagateLenslets(par,imagePlaneRot, 
                            wavelist_endpts[i], wavelist_endpts[i + 1],
                            hires_arrs,lam_arr,10)
    else:
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        ncpus = multiprocessing.cpu_count()
        consumers = [ Consumer(tasks, results)
                      for i in range(ncpus) ]
        for w in consumers:
            w.start()
    
        for i in range(len(waveList)):
            imagePlaneRot = (wavelist_endpts[i + 1]-wavelist_endpts[i])*processImagePlane(par,interpolatedInputCube.data[i])
            inputCube += [imagePlaneRot]
            tasks.put(Task(i, propagateLenslets, (par,imagePlaneRot,
                        wavelist_endpts[i], wavelist_endpts[i + 1],
                        hires_arrs,lam_arr,10)))
    
        for i in range(ncpus):
            tasks.put(None)
        for i in range(len(waveList)):
            index, poly = results.get()
            polyimage[index] = poly
    
    if par.saveRotatedInput: Image(data=np.array(inputCube),header=par.hdr).write(par.exportDir+'/imagePlaneRot.fits')
    if par.savePoly: Image(data=polyimage,header=par.hdr).write(par.exportDir+'/'+name+'poly.fits') 
    
    finalFrame = np.sum(polyimage,axis=0)
    
    detectorFrame = rebinDetector(par,finalFrame,clip=False)
    if par.saveDetector: Image(data=detectorFrame,header=par.hdr).write(par.exportDir+'/'+name+'.fits') 
    log.info('Done.')
    t['End'] = time.time()
    log.info("Performance: %d seconds total" % (t['End'] - t['Start']))

    return detectorFrame

        
    

def propagateIFS(par,wavelist,inputcube,name='detectorFrame'):
    '''
    Propagates an input cube through the Integral Field Spectrograph
    
    Parameters
    ----------
    par :   Parameter instance
            with at least the key IFS parameters, interlacing and scale
    lamlist : list of floats
            List of wavelengths in microns
    inputcube : Image
            or HDU. data is 3D ndarray with first dimension the same length as lamlist
            header needs to contain the 'PIXSIZE' and 'LAM_C' keywords
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
        log.info('Final detector pixel per lenslet: %f' % (par.pxprlens/par.pxperdetpix))
    else:
        log.info('Final detector pixel per PSFLet: %f' % (int(3*par.pitch/par.pixsize)))
    
    for i in range(len(waveList)):
        log.info('Processing wavelength %f (%d out of %d)' % (waveList[i],i,nframes))
        if not par.gaussian:
            propagateSingleWavelength(par,i,wavelist,interpolatedInputCube,finalFrame,
                                        refWaveList,kernelList,allweights,locations)
        else:
            propagateSingleWavelength(par,i,wavelist,interpolatedInputCube,finalFrame)
                       

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

    
def reduceIFSMap(par,IFSimageName,method='optext',smoothbad = True,name=None):
    '''
    Main reduction function
    
    Uses various routines to extract an IFS detector map into a spectral-spatial cube.
    
    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    IFSimageName : string or 2D ndarray
            Path of image file, of 2D ndarray. 
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

    if isinstance(IFSimageName,basestring):
        IFSimage = Image(filename = IFSimageName)
        reducedName = IFSimageName.split('/')[-1].split('.')[0]
    else:
        IFSimage = Image(data=IFSimageName)
        if name is None:
            reducedName = time.strftime("%Y%m%d-%H%M%S")
        else:
            reducedName = name

    if method == 'lstsq':
        reducedName += '_red_lstsq'
        cube = lstsqExtract(par,par.exportDir+'/'+reducedName,IFSimage,smoothandmask=smoothbad)
    elif method == 'optext':
        reducedName += '_red_optext'
        cube = intOptimalExtract(par,par.exportDir+'/'+reducedName,IFSimage,smoothandmask=smoothbad)
    else:
        log.info("Method not found")
        
    log.info('Elapsed time: %fs' % (time.time()-start))

    return cube


def reduceIFSMapList(par,IFSimageNameList,method='optext',parallel=True,smoothbad=True):
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
            reducedName = IFSimageNameList[i].split('/')[-1].split('.')[0]
            if method == 'lstsq':
                reducedName += '_red_lstsq'
                tasks.put(Task(i, lstsqExtract, (par, par.exportDir+'/'+reducedName,IFSimage,smoothbad)))
            elif method == 'optext':
                reducedName += '_red_optext'
                tasks.put(Task(i, intOptimalExtract, (par, par.exportDir+'/'+reducedName,IFSimage,smoothbad)))
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
            if method == 'lstsq':
                reducedName += '_red_lstsq'
                cube = lstsqExtract(par,par.exportDir+'/'+reducedName,IFSimage,smoothbad)
            elif method == 'optext':
                reducedName += '_red_optext'
                cube = intOptimalExtract(par,par.exportDir+'/'+reducedName,IFSimage,smoothbad)
            else:
                log.info("Method not found")

    log.info('Elapsed time: %fs' % (time.time()-start))



def prepareCube(par,wavelist,inputcube,QE=True,adjustment=0.98898):
    '''
    Processes input cubes
    '''
    if not 'INSLICES' in par.hdr:
        par.hdr.append(('INSLICES',len(wavelist),'Number of wavelengths in input cube'), end=True) 
        par.hdr.append(('ADJUST',adjustment,'Adjustment factor for rebinning error'), end=True) 

    if QE:
        loadQE = np.loadtxt(par.codeRoot+"/"+par.QE)
        QEinterp = interp1d(loadQE[:,0],loadQE[:,1])
        QEvals = QEinterp(wavelist)

        if not "APPLYQE" in par.hdr:
            par.hdr.append(('APPLYQE',QE,'Applied quantum efficiency?'), end=True) 
        for iwav in range(len(wavelist)):
            inputcube.data[iwav] *= QEvals[iwav]
    # adjust for small error in rebinning function
    inputcube.data *= adjustment
    outcube = Image(data=inputcube.data,header=inputcube.header)
    return wavelist,outcube



def createWavecalFiles(par,lamlist,lamc=770.,dlam=1.):
    '''
    Creates a set of monochromatic IFS images to be used in wavelength calibration step
    '''
    
    par.saveDetector=False
    inputCube = np.ones((1,512,512),dtype=float)
    inCube = pyf.HDUList(pyf.PrimaryHDU(inputCube))
    inCube[0].header['LAM_C'] = lamc/1000.
    inCube[0].header['PIXSIZE'] = 0.5
    filelist = []
    for wav in lamlist:
#         detectorFrame = propagateIFS(par,[wav*1e-3],inCube[0])
        detectorFrame = polychromeIFS(par,[wav],inCube[0],dlambda=dlam,parallel=False)
        filename = par.wavecalDir+'det_%3d.fits' % (wav)
        filelist.append(filename)
        Image(data=detectorFrame,header=par.hdr).write(filename)
    par.lamlist = lamlist
    par.filelist = filelist

