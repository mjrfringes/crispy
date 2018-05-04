#!/usr/bin/env python

from scipy import signal
import scipy.interpolate
import scipy.ndimage

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from detutils import frebin
from par_utils import Task, Consumer

from initLogger import getLogger
log = getLogger('crispy')
from image import Image
import multiprocessing
from astropy.io import fits
import pkg_resources

def rebinDetector(par, finalFrame, clip=False):
    '''
    Rebins the dense detector map with the correct scaling while conserving flux.
    This also works with non-integer ratios.

    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    finalFrame : 2D ndarray
            Dense detector map to be rebinned.

    Returns
    -------
    detectorFrame : 2D array
            Return the detector frame with correct pixel scale.

    '''
    detpixprlenslet = par.pitch / par.pixsize
    log.info('Number of detector pixels per lenslet: %f' % detpixprlenslet)

    newShape = (finalFrame.shape[0] //
                (par.pxperdetpix), finalFrame.shape[1] //
                (par.pxperdetpix))
    log.info('Rebinning final detector. Image has dimensions %dx%d' % newShape)
    detectorFrame = frebin(finalFrame, newShape)

    if clip:
        i = int(detectorFrame.shape[0] * (1. - 1. / np.sqrt(2.)) / 2.)
        detectorFrame = detectorFrame[i:-i, i:-i]

    return detectorFrame


def readDetector(par, IFSimage, inttime=100):
    '''
    Read noise, CIC, dark current; NO TRAPS
    Input is IFSimage in average photons per second
    Quantum efficiency considerations are already taken care of when
    generating IFSimage images
    '''

    if 'RN' not in par.hdr:
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*' * 60), end=True)
        par.hdr.append(
            ('comment',
             '*' *
             22 +
             ' Detector readout ' +
             '*' *
             20),
            end=True)
        par.hdr.append(('comment', '*' * 60), end=True)
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(
            ('TRANS',
             par.losses,
             'IFS Transmission factor'),
            end=True)
        par.hdr.append(('POL', par.pol, 'Polarization losses'), end=True)
        par.hdr.append(
            ('PHCTEFF',
             par.PhCountEff,
             'Photon counting efficiency'),
            end=True)
        par.hdr.append(
            ('NONOISE',
             par.nonoise,
             'Ignore all noises?'),
            end=True)
        if ~par.nonoise:
            par.hdr.append(
                ('POISSON', par.poisson, 'Poisson noise?'), end=True)
            par.hdr.append(
                ('RN', par.RN, 'Read noise (electrons/read)'), end=True)
            par.hdr.append(('CIC', par.CIC, 'Clock-induced charge'), end=True)
            par.hdr.append(('DARK', par.dark, 'Dark current'), end=True)
            par.hdr.append(('Traps', par.Traps, 'Use traps? T/F'), end=True)
            par.hdr.append(
                ('EMSTATS', par.EMStats, 'EM statistics?'), end=True)
            par.hdr.append(
                ('EMGAIN', par.EMGain, 'Gain of the EM stage'), end=True)
            par.hdr.append(
                ('PCBIAS', par.PCbias, 'To make RN zero-mean '), end=True)
            par.hdr.append(
                ('PCMODE', par.PCmode, 'Photon counting mode?'), end=True)
            if par.PCmode:
                par.hdr.append(
                    ('THRESH', par.threshold, 'Photon counting threshold'), end=True)
            par.hdr.append(
                ('LIFEFRAC',
                 par.lifefraction,
                 'Mission life fraction (changes CTE if >0)'),
                end=True)
    par.hdr['INTTIME'] = inttime

    # just to deal with small numerical errors - normally there is nothing
    # there
    IFSimage.data[IFSimage.data < 0] = 0.0

    eff = par.losses * par.PhCountEff * par.pol

    photoelectrons = IFSimage.data * eff * inttime

    if par.nonoise:
        return photoelectrons
    else:

        # verify with Bijan that the CIC/dark doesn't contribute to this
        # formula
        if par.lifefraction > 0.0:
            photoelectrons[photoelectrons > 0] *= np.minimum(np.ones(photoelectrons[photoelectrons > 0].shape),
                                                             1 + par.lifefraction * 0.51296 * (np.log10(photoelectrons[photoelectrons > 0]) + 0.0147233))

        average = photoelectrons + par.dark * inttime + par.CIC

        # calculate electron generation in the CCD frame
        if par.poisson:
            atEMRegister = np.random.poisson(average)
        else:
            atEMRegister = average

        # calculate the number of electrons after the EM register
        if par.EMStats:
            EMmask = atEMRegister > 0
            afterEMRegister = np.zeros(atEMRegister.shape)
            afterEMRegister[EMmask] = np.random.gamma(
                atEMRegister[EMmask], par.EMGain, atEMRegister[EMmask].shape)
        else:
            afterEMRegister = par.EMGain * atEMRegister

        # add read noise
        if par.EMStats and par.RN > 0:
            afterRN = afterEMRegister + \
                np.random.normal(par.PCbias, par.RN, afterEMRegister.shape)
            # clip at zero
            afterRN[afterRN < 0] = 0
        else:
            afterRN = afterEMRegister + par.PCbias

        # add photon counting thresholding
        if par.PCmode:
            PCmask = afterRN > par.PCbias + par.threshold * par.RN
            afterRN[PCmask] = 1.0
            afterRN[~PCmask] = 0.
        else:
            afterRN -= par.PCbias
            afterRN /= par.EMGain

        return afterRN


def averageDetectorReadout(
        par,
        filelist,
        detectorFolderOut,
        suffix='detector',
        offaxis=None,
        averageDivide=False,
        factor=1.0,
        zodi=None,
        forced_inttime=None,
        forced_tottime=None):
    '''
    Process a list of files and creates individual detector readouts
    If we want only one file, we can just make a list of 1
    '''
    det_outlist = []

    for reffile in filelist:
        log.info('Apply detector readout on ' + reffile.split('/')[-1])
        img = Image(filename=reffile)
        if offaxis is not None:
            off = Image(offaxis)
            img.data *= factor  # Multiplies by post-processing factor
            img.data += off.data
        if zodi is not None:
            z = Image(zodi)
            img.data += z.data
        par.makeHeader()

        if forced_inttime is None:
            inttime = par.timeframe / par.Nreads
            nreads = int(par.Nreads)
            exptime = int(par.timeframe)
        else:
            inttime = forced_inttime
            nreads = int(forced_tottime / forced_inttime)
            exptime = int(forced_tottime)
        log.info("Nreads: %d" % nreads)

        frame = np.zeros(img.data.shape)
        varframe = np.zeros(img.data.shape)
        # averaging reads
        for i in range(nreads):
            newread = readDetector(par, img, inttime=inttime)
            frame += newread
            varframe += newread**2
        if averageDivide:
            frame /= nreads
            varframe /= nreads
            varframe -= frame**2
        if "NREADS" not in par.hdr:
            par.hdr.append(
                ('NREADS',
                 nreads,
                 'Number of subframes co-added  per image'),
                end=True)
            par.hdr.append(
                ('EXPTIME',
                 exptime,
                 'Total exposure time for number of frames'),
                end=True)
        par.hdr['NREADS'] = nreads
        par.hdr['EXPTIME'] = exptime

        outimg = Image(data=frame, ivar=1. / varframe, header=par.hdr)
        # append '_suffix' to the file name
        outimg.write(detectorFolderOut + '/' + reffile.split('/')
                     [-1].split('.')[0] + '_' + suffix + '.fits', clobber=True)
        det_outlist.append(detectorFolderOut + '/' + reffile.split('/')
                           [-1].split('.')[0] + '_' + suffix + '.fits')
    return det_outlist


def multipleReadouts(
        par,
        filename,
        detectorFolderOut,
        suffix='detector',
        offaxis=None,
        averageDivide=False,
        factor=1.0,
        zodi=None,
        forced_inttime=None,
        forced_tottime=None):

    log.info('Apply detector readout on ' + filename.split('/')[-1])
    img = Image(filename=filename)
    if offaxis is not None:
        off = Image(offaxis)
        img.data *= factor  # Multiplies by post-processing factor
        img.data += off.data
    if zodi is not None:
        z = Image(zodi)
        img.data += z.data
    par.makeHeader()

    if forced_inttime is None:
        inttime = par.timeframe / par.Nreads
        nreads = int(par.Nreads)
        exptime = int(par.timeframe)
    else:
        inttime = forced_inttime
        nreads = int(forced_tottime / forced_inttime)
        exptime = int(forced_tottime)
    #log.info("Nreads: %d" % nreads)

    frame = np.zeros(img.data.shape)
    varframe = np.zeros(img.data.shape)
    # averaging reads
    for i in range(nreads):
        newread = readDetector(par, img, inttime=inttime)
        frame += newread
        varframe += newread**2
    if averageDivide:
        frame /= nreads
        varframe /= nreads
        varframe -= frame**2
    if "NREADS" not in par.hdr:
        par.hdr.append(
            ('NREADS',
             nreads,
             'Number of subframes co-added  per image'),
            end=True)
        par.hdr.append(
            ('EXPTIME',
             exptime,
             'Total exposure time for number of frames'),
            end=True)
    par.hdr['NREADS'] = nreads
    par.hdr['EXPTIME'] = exptime

    outimg = Image(data=frame, ivar=1. / varframe, header=par.hdr)
    # append '_suffix' to the file name
    outimg.write(detectorFolderOut + '/' + filename.split('/')
                 [-1].split('.')[0] + '_' + suffix + '.fits', clobber=True)
    return detectorFolderOut + '/' + \
        filename.split('/')[-1].split('.')[0] + '_' + suffix + '.fits'


def averageDetectorReadoutParallel(
        par,
        filelist,
        detectorFolderOut,
        suffix='detector',
        offaxis=None,
        averageDivide=False,
        factor=1.0,
        zodi=None,
        forced_inttime=None,
        forced_tottime=None):
    '''
    Process a list of files and creates individual detector readouts
    If we want only one file, we can just make a list of 1
    '''
    det_outlist = []

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    ncpus = multiprocessing.cpu_count()
    consumers = [Consumer(tasks, results)
                 for i in range(ncpus)]
    for w in consumers:
        w.start()

    for i in range(len(filelist)):
        tasks.put(
            Task(
                i,
                multipleReadouts,
                (par,
                 filelist[i],
                 detectorFolderOut,
                 'detector',
                 offaxis,
                 averageDivide,
                 factor,
                 zodi,
                 forced_inttime,
                 forced_tottime)))

    for i in range(ncpus):
        tasks.put(None)
    for i in range(len(filelist)):
        index, strres = results.get()
        det_outlist.append(strres)

    return det_outlist


def calculateDark(par, filelist):
    '''
    Process a series of dark frames to get an estimate of the dark current frame that matches the exposures.
    '''

    fileshape = Image(filename=filelist[0]).data.shape
    darkframe = np.zeros(fileshape)
    blankframe = Image(data=np.zeros(fileshape))
    inttime = par.timeframe / par.Nreads
    par.makeHeader()
    for i in range(par.Nreads * len(filelist)):
        darkframe += readDetector(par, blankframe, inttime=inttime)
    return darkframe


from scipy import stats


def photonCounting(cimg,
                   EMGain=1.0,
                   RN=0.0,
                   ndark=0,
                   ncic=0,
                   PCbias=0.0,
                   threshold=6,
                   xs=2144,
                   ys=1137,
                   ncr=0,
                   nhpcnts=0,
                   hpidx=None,
                   poisson=True,
                   EMStats=True,
                   PCmode=True):
    '''
    WFIRST EMCCD readout function for a single frame
    
    Parameters
    ----------
    cimg: 2d ndarray
        Input map in average counts/pix
    ndark: float
        Number of dark current counts per active area frame
    ncic: float
        Number of CIC counts per frame (active area + overscan)
    eff: float
        Bulk adjustment in efficiency (does the same as QE)
    EMGain: float
        Gain of the multiplying register
    RN: float
        Read noise
    PCbias: float
        Bias that one can put to the readout register to read positive values.
        The readout noise is a Gaussian distribution centered on PCbias and of std RN
    ncr: float
        Number of cosmic ray counts per active area frame
    nhpcnts: float
        Number of hot pixel counts per active area frame
    hpidx: float
        Locations of the hot pixels in the flatten active area frame
    threshold: float
        Used only if PCmode=True. Selects counts that are higher than 
        PCbias + threshold*RN, and counts them at "1". Everything else is zero.
        Watch out for coincidence losses
    xs: int
        Size of the readout area including overscan (long size). This needs to be larger
        than the image
    ys: int
        Size of the readout area including overscan (short size). This needs to be larger
        than the image
    poisson: Boolean
        If False, ignore Poisson statistics of the photon flux
    EMStats: Boolean
        Use the gain register statistics. If False, then a simple gain is applied with no
        gain register noise and no read noise.
    PCmode: Boolean
        Uses photon-counting mode by thresholding post-gain register map
        
    Returns
    -------
    afterRN: 2D ndarray
        Resulting digital frame in counts, including overscan area
    '''


    yi, xi = cimg.shape
    
    ##########################################################################################
    # Calculate Hot Pixels contribution
    ##########################################################################################
    hpimg = np.zeros(xi*yi, dtype=np.float)
    if hpidx is not None and nhpcnts>0:
            idx1 = np.random.randint(0, len(hpidx), nhpcnts)
            hpimg[hpidx[idx1]] += 1

    hpimg = np.reshape(hpimg, (yi, xi))
    
    ##########################################################################################
    # Calculate Dark current contribution
    ##########################################################################################
    dkimg = np.zeros((yi, xi), dtype=np.float)
    if ndark>1:
        x0 = np.random.randint(0, xi, ndark)
        y0 = np.random.randint(0, yi, ndark)
        dkimg[y0, x0] += 1
  
    ##########################################################################################
    # Apply Poisson noise or not
    ##########################################################################################
    if poisson:
        average = np.random.poisson(cimg)

    ##########################################################################################
    # Add CIC if necessary
    ##########################################################################################
    atEMRegister = np.zeros((ys, xs), dtype=np.float)
    if ncic > 0:
        x0 = np.random.randint(0, xs, ncic)
        y0 = np.random.randint(0, ys, ncic)
        atEMRegister[y0, x0] += 1

    ##########################################################################################
    # Add in active area before EM gain register
    ##########################################################################################
    atEMRegister[:yi,-xi:] += average + hpimg + dkimg

    ##########################################################################################
    # Calculate the number of counts after the EM register
    ##########################################################################################
    afterEMRegister = np.zeros((ys, xs), dtype=np.float)
    if EMStats:
        EMmask = atEMRegister > 0
        afterEMRegister[EMmask] = np.random.gamma(
            atEMRegister[EMmask], EMGain, afterEMRegister[EMmask].shape)
    else:
        afterEMRegister = EMGain * atEMRegister

    ##########################################################################################
    # Calculate cosmic ray image in entire readout
    ##########################################################################################
    crimg = np.zeros(xs*ys, dtype=np.float)
    if ncr > 1:
        toyx = np.arange(xs, dtype=np.float)
        ncrpx = np.round(0.0323*EMGain+133.5)
        ncrpxidx = np.where(toyx == ncrpx)[0][0]
        # Empirical toy model
        crtoy = 5e3*np.exp(-toyx/300.)+3e4*np.exp(-toyx/30.) + \
            1e4*np.exp(-toyx/15.)+3e5*np.exp(-toyx/5.)
        # Scale to photon counting threshold
        crtoy = crtoy/crtoy[ncrpxidx]*5.*RN
        # Saturate leading pixels
        crtoy[0:2] = 65536
        # This is not Pythonized yet; I'd have to think about it a bit, but it should be possible
        for i in range(ncr.astype(int)):
            crx = np.random.randint(0, xi)
            crx += xs-xi
            cry = np.random.randint(0, yi)
            cridx = cry*xs+crx
            if cridx > len(crimg)-1-xs:
                cridx = len(crimg)-1-xs
            crimg[cridx:cridx+xs] = crtoy

    afterEMRegister += np.reshape(crimg, (ys, xs))

    ##########################################################################################
    # Add read noise with some arbitrary bias to yield positive values if desired
    ##########################################################################################
    if EMStats and RN > 0:
        afterRN = afterEMRegister + \
            np.random.normal(PCbias, RN, afterEMRegister.shape)
        # clip at zero
#         afterRN[afterRN < 0] = 0
    else:
        afterRN = afterEMRegister + PCbias

    ##########################################################################################
    # Add photon counting thresholding
    ##########################################################################################
    if PCmode:
        PCmask = afterRN > PCbias + threshold * RN
        afterRN[PCmask] = 1.0
        afterRN[~PCmask] = 0.
    else:
        afterRN -= PCbias
#         afterRN /= EMGain

    return afterRN


def readoutPhotonFluxMapWFIRST(
        fluxMap,
        tottime,
        inttime=None,
        QE=1.0,
        darkBOL=1.4e-4,
        darkEOL=2.8e-4,
        CIC=1e-2,
        eff=1.0,
        EMGain=5000.,
        RN=100.0,
        PCbias=1000.0,
        crrate=0.0, # per detector per second
        hprate=0.0,
        threshold=6.,
        lifefraction=0.0,
        dqeKnee=0.858,
        dqeFluxSlope=3.24,
        dqeKneeFlux=0.089,
        xs=2144,
        ys=1137,
        pixsize=0.0013,
        transpose=False,
        returnFullFrame=False,
        nonoise=False,
        poisson=True,
        EMStats=True,
        PCmode=True,
        PCcorrect=False,
        normalize=False,
        verbose=False):
    '''
    WFIRST EMCCD readout function for averaging a long observation
    
    Parameters
    ----------
    fluxMap: 2d ndarray
        Input map in photons/sec/pix; QE should already be included, although a bulk QE
        parameter is available too
    tottime: float
        Total exposure time
    inttime: float
        Exposure time per image. If None, the program calculates the exposure time needed
        to achieve 0.1 count on the maximum pixel of the image
    QE: float
        Bulk adjustment in QE
    darkBOL: float
        Dark current in counts/s/pix at the beginning of life of the mission
    darkEOL: float
        Dark current in counts/s/pix at the end of life of the mission
    CIC: float
        Clock-induced charge in counts/pix/frame
    eff: float
        Bulk adjustment in efficiency (does the same as QE)
    EMGain: float
        Gain of the multiplying register
    RN: float
        Read noise
    PCbias: float
        Bias that one can put to the readout register to read positive values.
        The readout noise is a Gaussian distribution centered on PCbias and of std RN
    crrate: float
        Cosmic ray rate in number per cm^2 per second
    hprate: float
        Hot pixel rate in number per second per frame (watch out, this could vary with
        frame size) 
    threshold: float
        Used only if PCmode=True. Selects counts that are higher than 
        PCbias + threshold*RN, and counts them at "1". Everything else is zero.
        Watch out for coincidence losses
    lifefraction: float
        Fraction of mission year (1.0 means 5 years)
    dqeKnee: float
        dQE parameter (set to 1.0 to ignore)
    dqeFluxSlope: float
        dQE parameter (set to 0.0 to ignore)
    dqeKneeFlux: float
        dQE parameter (if the two other parameters are set to 1.0 and 0.0, this one
        is automatically ignored)
    xs: int
        Size of the readout area including overscan (long size). This needs to be larger
        than the image
    ys: int
        Size of the readout area including overscan (short size). This needs to be larger
        than the image
    transpose: Boolean
        Transpose the input map
    returnFullFrame: Boolean
        If True, returns the entire frame including the frame transfer area. If False,
        only return the active area
    nonoise: Boolean
        Ignore all noise contributions
    poisson: Boolean
        If False, ignore Poisson statistics of the photon flux
    EMStats: Boolean
        Use the gain register statistics. If False, then a simple gain is applied with no
        gain register noise and no read noise.
    PCmode: Boolean
        Uses photon-counting mode by thresholding post-gain register map
    PCcorrect: Boolean
        Implements the Basler et al correction to keep photometry legit after photon
        counting
    normalize: Boolean
        If set to True, the program averages frames instead of adding them, and divides
        by the exposure time
    verbose: Boolean
        Display text content to monitor program evolution
        
    Returns
    -------
    frame: 2D ndarray
        Resulting integrated frame of size (ys,xs) including overscan area
    '''

    if transpose: photoElectronsRate = QE * eff * fluxMap.T
    else: photoElectronsRate = QE * eff * fluxMap
    
    yi, xi = photoElectronsRate.shape

    ##########################################################################################
    # Allow to turn all noises off
    ##########################################################################################
    if nonoise:
        return photoElectronsRate * tottime
    else:
        ######################################################################################
        # if inttime is None, determine the exposure time so that the brightest
        # pixel is only 0.1 electrons
        ######################################################################################
        if inttime is None:
            exptime = 0.1 / np.amax(photoElectronsRate)
            if verbose: print("Individual exposure time: %.3f" % exptime)
        else:
            exptime = inttime

        nreads = int(tottime / exptime)
        if verbose: print("Number of reads: %d" % nreads)

        ######################################################################################
        # Number of photoelectrons detected
        ######################################################################################
        photoElectrons = photoElectronsRate * exptime

        ######################################################################################
        # QE degradation as a function of mission lifetime (due to charge traps)
        ######################################################################################
        if lifefraction > 0.0:
            photoElectrons *= np.maximum(
                np.zeros(
                    photoElectrons.shape),
                np.minimum(
                    np.ones(
                        photoElectrons.shape) + lifefraction * (
                        dqeKnee - 1.),
                    np.ones(
                        photoElectrons.shape) + lifefraction * (
                        dqeKnee - 1) + lifefraction * dqeFluxSlope * (
                        photoElectrons - dqeKneeFlux)))

        ######################################################################################
        # Compute dark counts, cic counts at epoch and average count rate
        ######################################################################################
        ndark = np.round((darkBOL + lifefraction * (darkEOL - darkBOL))*exptime*xi*yi).astype(int)
        if verbose: print('{:} dark counts per exposure'.format(ndark))
        ncic = np.round(CIC*xs*ys).astype(int)
        if verbose: print('{:} CIC counts per exposure'.format(ncic))

        average = photoElectrons# + dark * exptime

        frame = np.zeros((ys,xs),dtype=np.float)
        
        ######################################################################################
        # Hot pixel map and cosmic ray counts
        ######################################################################################
        ncr = np.round(crrate*exptime*(xi*pixsize)*(yi*pixsize)).astype(int)
        if verbose: print('{:} cosmic rays per exposure'.format(ncr))
        yrs = np.round(lifefraction*5).astype(int)
        yi, xi = fluxMap.shape
        if yrs>0:
            datafile = pkg_resources.resource_filename('crispy', 'Inputs') + '/EMCCD/'
            datafile += 'emccd_hot_pixel_map_%dyr.fits' % np.round(yrs)
            hpmask = fits.getdata(datafile)
            hpmask = np.reshape(hpmask[:yi, :xi], -1)
            hpidx = np.array(np.where(hpmask > 0)[0])
            hpidx = np.concatenate([hpidx, hpidx+1, hpidx+2, hpidx+3])
            hpidx[hpidx > len(hpmask)] = len(hpmask)
            nhpcnts = np.round(len(hpidx)*hprate/3600.*exptime).astype(int)
        else:
            hpidx = None
            nhpcnts = 0
        if verbose: print('{:} hot pixel counts per exposure'.format(nhpcnts))

        ######################################################################################
        # Average reads
        ######################################################################################
        for n in range(nreads):
            newread = photonCounting(average,
                                     EMGain=EMGain,
                                     RN=RN,
                                     ndark=ndark,
                                     ncic=ncic,
                                     PCbias=PCbias,
                                     ncr=ncr,
                                     nhpcnts=nhpcnts,
                                     hpidx=hpidx,
                                     xs=xs,
                                     ys=ys,
                                     threshold=threshold,
                                     poisson=poisson,
                                     EMStats=EMStats,
                                     PCmode=PCmode)
            frame += newread

        ######################################################################################
        # Compensate for PC mode and/or normalize if desired
        ######################################################################################
        if normalize:
            frame /= float(nreads)
            if PCcorrect:
                frame *= np.exp(RN * threshold / EMGain)
                frame = -np.log(1. - frame)
            frame /= exptime
        else:
            if PCcorrect:
                frame *= np.exp(RN * threshold / EMGain)
                frame = -np.log(1. - frame)

        if returnFullFrame:
            return frame
        else:
            return frame[:yi, -xi:]

def mkemccd(modeln,
            gain=5000.,
            dark=0.5,
            cic=0.01,
            readn=99,
            crrate=5.,
            HPrate=3.,
            frmt=100.,
            yrs=3,
            xs=450,#2144,
            ys=250,#1137,
            transpose=False,
            debug=False,
            verbose=False):
    ''' EMCCD noise generator based on Patrick's IDL code
    
    Parameters
    ----------
    modeln: string
        Link to the input FITS file
    gain: float
        Gain of the multiplying register
    dark: float
        Dark current in counts/hour/pix
    cic: float
        Clock-induced charge in counts/pix/frame
    readn: float
        Read noise
    crrate: float
        Cosmic ray rate
    HPrate: float
        Hot pixel rate
    frmt: float
        Integration time
    yrs: int
        Number of years into the mission (0 - 5)
    xs: int
        Size of the readout area including overscan (long size)
    ys: int
        Size of the readout area including overscan (short size)
    transpose: Boolean
        Transpose the input map
    debug: Boolean
        Export intermediary maps
    verbose: Boolean
        Display text content to monitor program evolution
    '''



    # size of image area
    data = fits.getdata(modeln)
    if transpose:
        data = data.T

    yi, xi = data.shape

    ##########################################################################################
    # Dark current
    ##########################################################################################
    if verbose:
        print("Dark current image")
    dkimg = np.zeros((yi, xi), dtype=np.float)
    ndark = np.round(dark/3600.*frmt*xi*yi).astype(int)
    if verbose: print('{:} dark counts per exposure'.format(ndark))
    x0 = np.random.randint(0, xi, ndark)
    y0 = np.random.randint(0, yi, ndark)
    dkimg[y0, x0] += 1

    ##########################################################################################
    # Clock-Induced Charge
    ##########################################################################################
    if verbose:
        print("CIC image")
    cicimg = np.zeros((ys, xs), dtype=np.float)
    ncic = np.round(cic*xs*ys).astype(int)
    if verbose: print('{:} CIC counts per exposure'.format(ncic))
    x0 = np.random.randint(0, xs, ncic)
    y0 = np.random.randint(0, ys, ncic)
    cicimg[y0, x0] += 1

    ##########################################################################################
    # Hot pixels
    ##########################################################################################
    if verbose:
        print("Hot pixels image")
    hpimg = np.zeros(xi*yi, dtype=np.float)
    if yrs != 0:
#         hpmask = fits.getdata(
#             '/Users/mrizzo/IFS/mkemccd_v6_171207/emccd_hot_pixel_map_%dyr.fits' % np.round(yrs))
        datafile = pkg_resources.resource_filename('crispy', 'Inputs') + '/EMCCD/'
        datafile += 'emccd_hot_pixel_map_%dyr.fits' % np.round(yrs)
        hpmask = fits.getdata(datafile)
        hpmask = np.reshape(hpmask[:yi, :xi], -1)
        hpidx = np.array(np.where(hpmask > 0)[0])
        hpidx = np.concatenate([hpidx, hpidx+1, hpidx+2, hpidx+3])
        hpidx[hpidx > len(hpmask)] = len(hpmask)
        nhpcnts = np.round(len(hpidx)*HPrate/3600.*frmt).astype(int)
        if verbose: print('{:} Hot pixel counts per exposure'.format(nhpcnts))
        
        idx1 = np.random.randint(0, len(hpidx), nhpcnts)
        hpimg[hpidx[idx1]] += 1

    hpimg = np.reshape(hpimg, (yi, xi))

    ##########################################################################################
    # Cosmic rays
    ##########################################################################################
    if verbose:
        print("Cosmic ray image")
    crimg = np.zeros(xs*ys, dtype=np.float)
    ncr = np.round(crrate*frmt*(xi*0.0013)*(yi*0.0013))
    if verbose: print('{:} cosmic rays per exposure'.format(ncr))
    if ncr > 1:
        toyx = np.arange(xs, dtype=np.float)
        ncrpx = np.round(0.0323*gain+133.5)
        ncrpxidx = np.where(toyx == ncrpx)[0][0]
        # Empirical toy model
        crtoy = 5e3*np.exp(-toyx/300.)+3e4*np.exp(-toyx/30.) + \
            1e4*np.exp(-toyx/15.)+3e5*np.exp(-toyx/5.)
        # Scale to photon counting threshold
        crtoy = crtoy/crtoy[ncrpxidx]*5.*readn
        # Saturate leading pixels
        crtoy[0:2] = 65536
        # This is not Pythonized yet; I'd have to think about it a bit, but it should be possible
        for i in range(ncr.astype(int)):
            crx = np.random.randint(0, xi)
            crx += xs-xi
            cry = np.random.randint(0, yi)
            cridx = cry*xs+crx
            if cridx > len(crimg)-1-xs:
                cridx = len(crimg)-1-xs
            crimg[cridx:cridx+xs] = crtoy

    crimg = np.reshape(crimg, (ys, xs))

    ##########################################################################################
    # Photon counts
    ##########################################################################################
    if verbose:
        print("Photon counts image")
    phimg = np.random.poisson(data*frmt)

    ##########################################################################################
    # Add components together
    ##########################################################################################
    imgsum = cicimg.copy()
    imgsum[:yi, xs-xi:xs] += dkimg+hpimg+phimg

    ##########################################################################################
    # Gain register
    ##########################################################################################
    if verbose:
        print("Gain image")

    gainimg = np.zeros((ys, xs), dtype=np.float)
    EMmask = imgsum > 0
    gainimg[EMmask] = np.random.gamma(
        imgsum[EMmask], gain)

    outimg = gainimg+crimg+np.random.normal(0.0, readn, (ys, xs))
    if debug:
        return outimg, gainimg, crimg, dkimg, hpimg, phimg
    else:
        return outimg



def photonCounting_old(average,
                EMGain=1.0,
                RN=0.0,
                PCbias=0.0,
                threshold=6,
                poisson=True,
                EMStats=True,
                PCmode=True):


        # calculate electron generation in the CCD frame
        if poisson:
            atEMRegister = np.random.poisson(average)
        else:
            atEMRegister = average
        
    
        # calculate the number of electrons after the EM register
        if EMStats:
            EMmask = atEMRegister>0
            afterEMRegister = np.zeros(atEMRegister.shape)
            afterEMRegister[EMmask] = np.random.gamma(atEMRegister[EMmask],EMGain,atEMRegister[EMmask].shape)
        else:
            afterEMRegister = EMGain*atEMRegister
        
        # add read noise
        if EMStats and RN>0:
            afterRN = afterEMRegister+np.random.normal(PCbias,RN,afterEMRegister.shape)
            # clip at zero
            afterRN[afterRN<0]=0
        else:
            afterRN = afterEMRegister+PCbias

        # add photon counting thresholding
        if PCmode:
            PCmask = afterRN>PCbias+threshold*RN
            afterRN[PCmask]=1.0
            afterRN[~PCmask]=0.
        else:
            afterRN -= PCbias
            afterRN /= EMGain
    
        return afterRN


def readoutPhotonFluxMapWFIRST_old(
                fluxMap, 
                tottime,
                inttime=None,
                QE=1.0,
                darkBOL=1.4e-4, 
                darkEOL=2.8e-4, 
                CIC=1e-2,
                eff=1.0,
                EMGain=2500.,
                RN=100.0,
                PCbias=1000.0,
                threshold=6.,
                lifefraction=0.0,
                dqeKnee=0.858,
                dqeFluxSlope=3.24,
                dqeKneeFlux=0.089,
                nonoise=False,
                poisson=True,
                EMStats=True,
                PCmode=True,
                PCcorrect=False,
                normalize=False,
                verbose=False):
   
    photoElectronsRate = QE*eff*fluxMap
    
    if nonoise:
        return photoElectronsRate*tottime
    else:
        # if inttime is None, determine the exposure time so that the brightest pixel is only 0.1 electrons  
        if inttime is None:
            exptime = 0.1/np.amax(QE*eff*fluxMap)
            if verbose: print("Individual exposure time: %.3f" % exptime)
        else:
            exptime=inttime
            
        nreads = int(tottime/exptime)
        if verbose: print("Number of reads: %d" % nreads)
            
        photoElectrons = photoElectronsRate*exptime
        
        if lifefraction>0.0:
            photoElectrons *= np.maximum(np.zeros(photoElectrons.shape),np.minimum(np.ones(photoElectrons.shape)+lifefraction*(dqeKnee-1.),np.ones(photoelectrons.shape)+lifefraction*(dqeKnee-1)+lifefraction*dqeFluxSlope*(photoElectrons-dqeKneeFlux)))

        dark = darkBOL+lifefraction*(darkEOL-darkBOL)
        average = photoElectrons+dark*exptime+CIC
    
        frame = np.zeros(average.shape)
        
        for n in range(nreads):
            newread = photonCounting_old(average,
                                    EMGain=EMGain,
                                    RN=RN,
                                    PCbias=PCbias,
                                    threshold=threshold,
                                    poisson=poisson,
                                    EMStats=EMStats,
                                    PCmode=PCmode)
            frame += newread
        if normalize:
            frame/=float(nreads)
            if PCcorrect:
                frame*=np.exp(RN*threshold/EMGain)
                frame=-np.log(1.-frame)
            frame/=exptime
        else:
            if PCcorrect:
                frame*=np.exp(RN*threshold/EMGain)
                frame=-np.log(1.-frame)

            
        return frame
