#!/usr/bin/env python

'''
Standalone IFS simulation code
MJ Rizzo and the IFS team

Originally inspired by T. Brandt's code for CHARIS
'''


import numpy as np
try:
    from astropy.io import fits as pyf
except BaseException:
    import pyfits as pyf
import time
import matplotlib.pyplot as plt
from crispy.tools.image import Image
from crispy.tools.lenslet import processImagePlane, propagateLenslets
from crispy.tools.spectrograph import createAllWeightsArray, selectKernel, loadKernels
from crispy.tools.detector import rebinDetector
from crispy.tools.plotting import plotKernels
from crispy.tools.reduction import testReduction, lstsqExtract, intOptimalExtract
import multiprocessing
from crispy.tools.par_utils import Task, Consumer
from crispy.tools.wavecal import get_sim_hires
from scipy.interpolate import interp1d
import glob
import astropy.units as u
from astropy.stats import sigma_clipped_stats

# the following code snippet is supposed to deal with the basestring having
# disappeared in Python 3
import types
try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring


from crispy.tools.initLogger import getLogger
log = getLogger('crispy')


def polychromeIFS(par, inWavelist, inputcube,
                  name='detectorFrame',
                  parallel=True,
                  QE=True,
                  wavelist_endpts=None,
                  dlambda=None,
                  lam_arr=None,
                  wavecalDir=None,
                  noRot=False,
                  dx=0.0,
                  upsample=3, # need to make this part of the header in the templates
                  npix=13,
                  nlam=10,
                  order=3
                  ):
    '''
    Propagates an input cube through the Integral Field Spectrograph

    Parameters
    ----------
    par :   Parameter instance
            with at least the key IFS parameters, interlacing and scale
    inWavelist : list of floats
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
    wavecal: string
        This can be used to add a distortion already measured from lab data, for example.
        Put in there the full folder name where we can find a 'lamsol.dat' file.
    noRot: boolean
        A rarely used option that allows to NOT rotate the input cube, if we want to simulate sending
        a input map aligned with the lenslets

    Returns
    -------
    detectorFrame : 2D array
            Return the detector frame
    '''
    
    
    par.makeHeader()
    par.hdr.append(('comment', ''), end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(
        ('comment',
         '*' *
         22 +
         ' IFS Simulation ' +
         '*' *
         18),
        end=True)
    par.hdr.append(('comment', '*' * 60), end=True)
    par.hdr.append(('comment', ''), end=True)

    try:
        input_sampling = inputcube.header['PIXSIZE']
        input_wav = inputcube.header['LAM_C'] * 1000.
    except BaseException:
        log.error('Missing header information in input file')
        raise

    if isinstance(inWavelist, u.Quantity):
        wavelist = inWavelist.to(u.nm).value
    else:
        # assume it is in nm
        wavelist = inWavelist

    ######################################################################
    # Calculate sampling ratio to resample rotated image and match the lenslet sampling
    ######################################################################

    par.pixperlenslet = par.lenslet_sampling / \
        (input_sampling * input_wav / par.lenslet_wav)
    log.info(
        'The number of input pixels per lenslet is %f' %
        par.pixperlenslet)
    par.hdr.append(
        ('SCALE',
         par.pixperlenslet,
         'Factor by which the input slice is rescaled'),
        end=True)

    nframes = inputcube.data.shape[0]
    allweights = None

    if inputcube.data.shape[0] != len(wavelist):
        log.error('Number of wavelengths does not match the number of slices')

    ######################################################################
    # Create cube that is interpolated to the correct level if necessary
    ######################################################################
    waveList, interpolatedInputCube = prepareCube(
        par, wavelist, inputcube, QE=QE)

    ######################################################################
    # Defines an array of times for performance monitoring
    ######################################################################
    t = {'Start': time.time()}

    if not par.gaussian:
        log.info('Using templates PSFLets')
    else:
        log.info('Using PSFlet gaussian approximation')

    ######################################################################
    # Allocate arrays, make sure you have abundant memory
    ######################################################################
    finalFrame = np.zeros(
        (par.npix * par.pxperdetpix,
         par.npix * par.pxperdetpix))
    polyimage = np.zeros(
        (len(waveList),
         par.npix *
         par.pxperdetpix,
         par.npix *
         par.pxperdetpix))

    ######################################################################
    # Determine wavelength endpoints
    ######################################################################
    if wavelist_endpts is None:
        log.warning('Assuming slices are evenly spread in wavelengths')
        if len(waveList) > 1:
            dlam = waveList[1] - waveList[0]
            wavelist_endpts = np.zeros(len(waveList) + 1)
            wavelist_endpts[:-1] = waveList - dlam / 2.
            wavelist_endpts[-1] = waveList[-1] + dlam / 2.
        else:
            if dlambda is None:
                log.error('No bandwidth specified')
            else:
                wavelist_endpts = np.array(
                    [waveList[0] - dlambda / 2., waveList[0] + dlambda / 2.])
    else:
        log.warning('Assuming endpoints wavelist is given')
#     print wavelist_endpts

    ######################################################################
    # Load template PSFLets
    ######################################################################
    # lam_arr needs to be provided the first time you create monochromatic
    # flats!
    if lam_arr is None:
        lam_arr = np.loadtxt(par.wavecalDir + "lamsol.dat")[:, 0]

    hires_arrs = []
    if par.gaussian:
        for i in range(len(lam_arr)):
            hiresarr = get_sim_hires(par, lam_arr[i])
            hires_arrs += [hiresarr]
        log.info('Creating Gaussian PSFLet templates')
        upsample=10
    else:
        try:
            hires_list = np.sort(
                glob.glob(
                    par.wavecalDir +
                    'hires_psflets_lam???.fits'))
            hires_arrs = [pyf.getdata(filename) for filename in hires_list]
            log.info('Loaded PSFLet templates')
        except BaseException:
            log.error('Failed loading the PSFLet templates')
            raise

    inputCube = []

    if not parallel:
        for i in range(len(waveList)):
            imagePlaneRot = (wavelist_endpts[i + 1] - wavelist_endpts[i]) * \
                processImagePlane(par, interpolatedInputCube.data[i], noRot)
            inputCube += [imagePlaneRot]
            polyimage[i] = propagateLenslets(par,
                                             imagePlaneRot,
                                             wavelist_endpts[i],
                                             wavelist_endpts[i + 1],
                                             hires_arrs,
                                             lam_arr,
                                             upsample,
                                             nlam,
                                             npix,
                                             order,
                                             dx)
    else:
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        ncpus = multiprocessing.cpu_count()
        consumers = [Consumer(tasks, results)
                     for i in range(ncpus)]
        for w in consumers:
            w.start()

        for i in range(len(waveList)):
            imagePlaneRot = (wavelist_endpts[i + 1] - wavelist_endpts[i]) * \
                processImagePlane(par, interpolatedInputCube.data[i], noRot)
            inputCube += [imagePlaneRot]
            tasks.put(Task(i,
                           propagateLenslets,
                           (par,
                            imagePlaneRot,
                            wavelist_endpts[i],
                               wavelist_endpts[i + 1],
                               hires_arrs,
                               lam_arr,
                               upsample,
                               nlam,
                               npix,
                               order,
                               dx)))

        for i in range(ncpus):
            tasks.put(None)
        for i in range(len(waveList)):
            index, poly = results.get()
            polyimage[index] = poly

    if par.saveRotatedInput:
        Image(
            data=np.array(inputCube),
            header=par.hdr).write(
            par.exportDir +
            '/imagePlaneRot.fits')
    if par.savePoly:
        Image(
            data=polyimage,
            header=par.hdr).write(
            par.exportDir +
            '/' +
            name +
            'poly.fits')

    detectorFrame = np.sum(polyimage, axis=0)

    if par.pxperdetpix != 1.:
        detectorFrame = rebinDetector(par, detectorFrame, clip=False)
    if par.saveDetector:
        Image(
            data=detectorFrame,
            header=par.hdr).write(
            par.exportDir +
            '/' +
            name +
            '.fits')
    log.info('Done.')
    t['End'] = time.time()
    log.info("Performance: %d seconds total" % (t['End'] - t['Start']))

    return detectorFrame


def reduceIFSMap(
        par,
        IFSimageName,
        method='optext',
        smoothbad=True,
        name=None,
        hires=False,
        dy=3,
        fitbkgnd=True,
        specialPolychrome=None,
        returnall=False,
        niter=10,
        pixnoise=None,
        medsub=True,
        normpsflets=False,
        gain=0.5):
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
        par.hdr.append(('comment', '*' * 60), end=True)
        par.hdr.append(
            ('comment',
             '*' *
             22 +
             ' Cube Extraction ' +
             '*' *
             21),
            end=True)
        par.hdr.append(('comment', '*' * 60), end=True)
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(
            ('R', par.R, 'Spectral resolution of final cube'), end=True)
        par.hdr.append(('CALDIR', par.wavecalDir.split(
            '/')[-2], 'Directory of wavelength solution'), end=True)
            


    if isinstance(IFSimageName, basestring):
        IFSimage = Image(filename=IFSimageName)
        reducedName = IFSimageName.split('/')[-1].split('.')[0]
    else:
        IFSimage = Image(data=IFSimageName)
        if name is None:
            reducedName = time.strftime("%Y%m%d-%H%M%S")
        else:
            reducedName = name

    mean, median, std = sigma_clipped_stats(IFSimage.data, sigma=3.0, iters=5)
    log.info("Mean, median, std: {:}".format((mean,median,std)))
    par.hdr.append(
        ('MEAN', mean, 'Mean of image'), end=True)
    par.hdr.append(
        ('MED', median, 'Median of image'), end=True)
    par.hdr.append(
        ('STD', std, 'Std of image'), end=True)
    
    if medsub:
        IFSimage.data -= median
        par.hdr.append(
            ('MEDSUB', True, 'Subtract median from image'), end=True)
        log.info('Subtracting median from image')
    else:
        par.hdr.append(
            ('MEDSUB', False, 'Subtract median from image'), end=True)

    if pixnoise is None:
        pixnoise=std**2


    if method in ['lstsq', 'lstsq_conv', 'RL', 'RL_conv']:
        reducedName += '_red_' + method
        cube = lstsqExtract(
            par,
            par.exportDir +
            '/' +
            reducedName,
            IFSimage,
            smoothandmask=smoothbad,
            hires=hires,
            dy=dy,
            fitbkgnd=fitbkgnd,
            specialPolychrome=specialPolychrome,
            returnall=returnall,
            mode=method,
            niter=niter,
            pixnoise=pixnoise,
            normpsflets=normpsflets,
            gain=gain)
    elif method == 'optext':
        reducedName += '_red_optext'
        cube = intOptimalExtract(
            par,
            par.exportDir +
            '/' +
            reducedName,
            IFSimage,
            smoothandmask=smoothbad)
    elif method == 'sum':
        reducedName += '_red_sum'
        cube = intOptimalExtract(
            par,
            par.exportDir +
            '/' +
            reducedName,
            IFSimage,
            smoothandmask=smoothbad,
            sum=True)
        
    else:
        log.info("Method not found")

    log.info('Elapsed time: %fs' % (time.time() - start))

    return cube


def reduceIFSMapList(
        par,
        IFSimageNameList,
        method='optext',
        parallel=True,
        smoothbad=True):
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
    # par.makeHeader()
    start = time.time()

    if 'CALDIR' in par.hdr:
        pass
    else:
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*' * 60), end=True)
        par.hdr.append(
            ('comment',
             '*' *
             22 +
             ' Cube Extraction ' +
             '*' *
             21),
            end=True)
        par.hdr.append(('comment', '*' * 60), end=True)
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(
            ('R', par.R, 'Spectral resolution of final cube'), end=True)
        par.hdr.append(('CALDIR', par.wavecalDir.split(
            '/')[-2], 'Directory of wavelength solution'), end=True)

    if parallel:
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        ncpus = multiprocessing.cpu_count()
        consumers = [Consumer(tasks, results)
                     for i in range(ncpus)]
        for w in consumers:
            w.start()

        # you call the function here, with all its arguments in a list
        for i in range(len(IFSimageNameList)):
            IFSimage = Image(filename=IFSimageNameList[i])
            reducedName = IFSimageNameList[i].split('/')[-1].split('.')[0]
            if method == 'lstsq':
                reducedName += '_red_lstsq'
                tasks.put(
                    Task(
                        i,
                        lstsqExtract,
                        (par,
                         par.exportDir +
                         '/' +
                         reducedName,
                         IFSimage,
                         smoothbad)))
            elif method == 'optext':
                reducedName += '_red_optext'
                tasks.put(
                    Task(
                        i,
                        intOptimalExtract,
                        (par,
                         par.exportDir +
                         '/' +
                         reducedName,
                         IFSimage,
                         smoothbad)))
            else:
                log.info("Method not found")

        for i in range(ncpus):
            tasks.put(None)

        for i in range(len(IFSimageNameList)):
            index, result = results.get()

    else:
        for i in range(len(IFSimageNameList)):
            IFSimage = Image(filename=IFSimageNameList[i])
            reducedName = IFSimageNameList[i].split('/')[-1].split('.')[0]
            if method == 'lstsq':
                reducedName += '_red_lstsq'
                cube = lstsqExtract(
                    par,
                    par.exportDir +
                    '/' +
                    reducedName,
                    IFSimage,
                    smoothbad)
            elif method == 'optext':
                reducedName += '_red_optext'
                cube = intOptimalExtract(
                    par, par.exportDir + '/' + reducedName, IFSimage, smoothbad)
            else:
                log.info("Method not found")

    log.info('Elapsed time: %fs' % (time.time() - start))

def getQE(par,wavelist):
    if isinstance(par.QE, basestring):
        loadQE = np.loadtxt(par.codeRoot + "/" + par.QE)
        QEinterp = interp1d(loadQE[:, 0], loadQE[:, 1])
        QEvals = QEinterp(wavelist)
    else:
        if hasattr(wavelist,'__len__'):
            QEvals = par.QE * np.ones(len(wavelist))
        else:
            QEvals = par.QE
    return QEvals
    

def prepareCube(par, wavelist, incube, QE=True, adjustment=1.0):
    # def prepareCube(par,wavelist,incube,QE=True,adjustment=1.0):
    '''
    Processes input cubes
    '''
    if 'INSLICES' not in par.hdr:
        par.hdr.append(
            ('INSLICES',
             len(wavelist),
                'Number of wavelengths in input cube'),
            end=True)
        par.hdr.append(
            ('ADJUST',
             adjustment,
             'Adjustment factor for rebinning error'),
            end=True)

    inputcube = Image(data=incube.data.copy(), header=incube.header)
    if QE:
        QEvals = getQE(par,wavelist)

        if "APPLYQE" not in par.hdr:
            par.hdr.append(
                ('APPLYQE', QE, 'Applied quantum efficiency?'), end=True)
            par.hdr.append(('QEFILE', par.QE, 'QE file/value used'), end=True)
        for iwav in range(len(wavelist)):
            inputcube.data[iwav] *= QEvals[iwav]
        # print(QEvals)
    # adjust for small error in rebinning function
    inputcube.data *= adjustment
    outcube = Image(data=inputcube.data, header=inputcube.header)
    return wavelist, outcube


def createWavecalFiles(par, lamlist, dlam=1., flux=None, background=0.0):
    '''
    Creates a set of monochromatic IFS images to be used in wavelength calibration step

    Parameters
    ----------
    par:   Parameter instance
            Contains all IFS parameters
    lamlist: list or array of floats
            List of discrete wavelengths in nm at which to create a monochromatic flat
    dlam:  float
            Width in nm of the small band for each of the monochromatic wavelengths.
            Default is 1 nm. This has no effect unless we are trying to add any noise.
    flux: float
            Number of counts per lenslet. If a noiseless
            image is preferred, leave this to None
    background: float
            Adds Poisson-distributed background to the image. Leave to None to ignore.

    Notes
    -----
    This function populates the fields par.lamlist and par.filelist which are subsequently
    used by the buildcalibrations function. If this createWavecalFiles is not called, the
    two fields need to be populated manually with the set of files and wavelengths that
    you want to use for the calibration.

    '''

    par.saveDetector = False
    inputCube = np.ones((1, 512, 512), dtype=float)
    inCube = pyf.HDUList(pyf.PrimaryHDU(inputCube))
    inCube[0].header['LAM_C'] = 0.5 * (lamlist[-1] + lamlist[0]) / 1000.
    inCube[0].header['PIXSIZE'] = 0.1
    filelist = []
    for wav in lamlist:
        # note the argument lam_arr, necessary when computing things for the
        # first time
        detectorFrame = polychromeIFS(
                                    par,
                                    [wav],
                                    inCube[0],
                                    dlambda=dlam,
                                    parallel=False,
                                    lam_arr=lamlist)
        if flux is not None:
            detectorFrame /= getQE(par,wav)*(par.lensletsampling/inCube[0].header['PIXSIZE'])**2
            detectorFrame = np.random.poisson(flux*detectorFrame+background)
        filename = par.wavecalDir + 'det_%3d.fits' % (wav)
        filelist.append(filename)
        Image(data=detectorFrame, header=par.hdr).write(filename)
    par.lamlist = lamlist
    par.filelist = filelist
    return filelist

from crispy.tools.locate_psflets import transform
from crispy.tools.imgtools import gausspsf

def quickMonochromatic(par=None, 
                        fwhm =2.0,
                        coefs = None,
                        Dx = 0.0,
                        Dy = 0.0,
                        flux = 1.0,
                        gsize = 5,
                        order = 3,
                        nlens = 108,
                        npix = 1024,
                        returnCoords = False):

    if coefs is None:
        if par is None: 
            raise
        else:
            scale = par.pitch / par.pixsize
            cphi = np.cos(par.philens)
            sphi = np.sin(par.philens)
            Xcoefs = np.array([par.npix//2+Dx,cphi*scale,0.0,0.,-sphi*scale,0.0,0.0,0.0,0.0,0.0])
            Ycoefs = np.array([par.npix//2+Dy,sphi*scale,0.0,0.,cphi*scale,0.0,0.0,0.0,0.0,0.0])
            coefs = np.concatenate([Xcoefs,Ycoefs])
    
    if par is not None:
        nlens = par.nlens
        npix = par.npix
    
    xindx = np.arange(-nlens / 2, nlens / 2)
    xindx, yindx = np.meshgrid(xindx, xindx)
    
    Xc, Yc = transform(xindx,yindx,order,coefs)
    
    lx = npix+2*gsize
    detectorFrame = np.zeros((lx,lx))
    ry = np.reshape(Yc,-1)
    rx = np.reshape(Xc,-1)
    for i in range(len(ry)):
            yi = ry[i]+gsize
            xi = rx[i]+gsize
            xmin = int(xi)-gsize
            xmax = xmin+2*gsize
            ymin = int(yi)-gsize
            ymax = ymin+2*gsize
            if ymin>0 and xmin>0 and xmax<lx and ymax<lx:
                dx = xi - np.floor(xi) + 0.5
                dy = yi - np.floor(yi) + 0.5
                detectorFrame[ymin:ymax,xmin:xmax]+=flux*gausspsf(size=2*gsize,fwhm=fwhm,offx=-dx,offy=-dy)

    detectorFrame = detectorFrame[gsize:-gsize,gsize:-gsize]
    if returnCoords:
        return detectorFrame,(Xc, Yc)
    return detectorFrame