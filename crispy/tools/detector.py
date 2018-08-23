#!/usr/bin/env python

from scipy import signal
import scipy.interpolate
import scipy.ndimage

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from crispy.tools.detutils import frebin
from crispy.tools.par_utils import Task, Consumer

from crispy.tools.initLogger import getLogger
log = getLogger('crispy')
from crispy.tools.image import Image
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
            photoElectrons *= np.maximum(np.zeros(photoElectrons.shape),np.minimum(np.ones(photoElectrons.shape)+lifefraction*(dqeKnee-1.),np.ones(photoElectrons.shape)+lifefraction*(dqeKnee-1)+lifefraction*dqeFluxSlope*(photoElectrons-dqeKneeFlux)))

        dark = darkBOL+lifefraction*(darkEOL-darkBOL)
        average = photoElectrons+dark*exptime+CIC
    
#         frame = np.zeros(average.shape)
        frame = []
        
        for n in range(nreads):
            newread = photonCounting_old(average,
                                    EMGain=EMGain,
                                    RN=RN,
                                    PCbias=PCbias,
                                    threshold=threshold,
                                    poisson=poisson,
                                    EMStats=EMStats,
                                    PCmode=PCmode)
#             frame += newread
            frame.append(newread)
        
        frame = np.array(frame)
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
#             afterRN /= EMGain
    
        return afterRN




