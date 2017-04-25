#!/usr/bin/env python

from scipy import signal
import scipy.interpolate
import scipy.ndimage

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from detutils import frebin
from initLogger import getLogger
log = getLogger('crispy')
from image import Image

def rebinDetector(par,finalFrame,clip=False):
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
    detpixprlenslet = par.pitch/par.pixsize 
    log.info('Number of detector pixels per lenslet: %f' % detpixprlenslet)
        
    newShape = (finalFrame.shape[0]//(par.pxperdetpix),finalFrame.shape[1]//(par.pxperdetpix))
    log.info('Rebinning final detector. Image has dimensions %dx%d' % newShape)
    detectorFrame = frebin(finalFrame,newShape) 
    
    if clip:
        i = int(detectorFrame.shape[0]*(1. - 1./np.sqrt(2.))/2.)
        detectorFrame = detectorFrame[i:-i, i:-i]

    return detectorFrame



def readDetector(par,IFSimage,inttime=100,nonoise=False):
    '''
    Read noise, CIC, dark current; NO TRAPS
    Input is IFSimage in average photons per second
    Quantum efficiency considerations are already taken care of when
    generating IFSimage images
    '''
    if not 'RN' in par.hdr:
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('comment', '*'*60), end=True)
        par.hdr.append(('comment', '*'*22 + ' Detector readout ' + '*'*20), end=True)
        par.hdr.append(('comment', '*'*60), end=True)    
        par.hdr.append(('comment', ''), end=True)
        par.hdr.append(('POISSON',par.poisson,'Poisson noise?'), end=True) 
        par.hdr.append(('RN',par.RN,'Read noise (electrons/read)'), end=True) 
        par.hdr.append(('CIC',par.CIC,'Clock-induced charge'), end=True) 
        par.hdr.append(('DARK',par.dark,'Dark current'), end=True) 
        par.hdr.append(('Traps',par.Traps,'Use traps? T/F'), end=True) 
        par.hdr.append(('PHCTEFF',par.PhCountEff,'Photon counting efficiency'),end=True)
        par.hdr.append(('EMSTATS',par.EMStats,'EM statistics?'),end=True)
        par.hdr.append(('EMGAIN',par.EMGain,'Gain of the EM stage'),end=True)
        par.hdr.append(('PCBIAS',par.PCbias,'To make RN zero-mean '),end=True)
        par.hdr.append(('PCMODE',par.PCmode,'Photon counting mode?'),end=True)
        if par.PCmode:
            par.hdr.append(('THRESH',par.threshold,'Photon counting threshold'),end=True)
        par.hdr.append(('LIFEFRAC',par.lifefraction,'Mission life fraction (changes CTE if >0)'),end=True)
        par.hdr.append(('TRANS',par.losses,'IFS Transmission factor'),end=True)
        par.hdr.append(('POL',par.pol,'Polarization losses'),end=True)
        par.hdr.append(('INTTIME',inttime,'Integration time per frame'),end=True)
                
        
    eff = par.losses*par.PhCountEff*par.pol
    
    photoelectrons = IFSimage.data*eff*inttime
    
    if nonoise:
        return photoelectrons
    else:
    
        # verify with Bijan that the CIC/dark doesn't contribute to this formula
        if par.lifefraction>0.0:
            photoelectrons[photoelectrons>0] *= np.minimum(np.ones(photoelectrons[photoelectrons>0].shape),1+par.lifefraction*0.51296*(np.log10(photoelectrons[photoelectrons>0])+0.0147233))    

        average = photoelectrons+par.dark*inttime+par.CIC
    
        # calculate electron generation in the CCD frame
        if par.poisson:
            atEMRegister = np.random.poisson(average)
        else:
            atEMRegister = average
        
    
        # calculate the number of electrons after the EM register
        if par.EMStats:
            EMmask = atEMRegister>0
            afterEMRegister = np.zeros(atEMRegister.shape)
            afterEMRegister[EMmask] = np.random.gamma(atEMRegister[EMmask],par.EMGain,atEMRegister[EMmask].shape)
        else:
            afterEMRegister = par.EMGain*atEMRegister
        
        # add read noise
        if par.RN>0:
            afterRN = afterEMRegister+np.random.normal(par.PCbias,par.RN,afterEMRegister.shape)
        else:
            afterRN = afterEMRegister+par.PCbias

        # add photon counting thresholding
        if par.PCmode:
            PCmask = afterRN>par.PCbias+par.threshold*par.RN
            afterRN[PCmask]=1.0 #(afterRN[PCmask]-par.PCbias)/par.EMGain
            afterRN[~PCmask]=0.
        else:
            afterRN -= par.PCbias
            afterRN /= par.EMGain
    
        return afterRN

def averageDetectorReadout(par,filelist,detectorFolderOut,suffix = 'detector',offaxis=None,averageDivide=False,factor=1.0,zodi=None):
    '''	
    Process a list of files and creates individual detector readouts
    If we want only one file, we can just make a list of 1
    '''
    det_outlist = []
	
    for reffile in filelist:
        log.info('Apply detector readout on '+reffile.split('/')[-1])
        img = Image(filename=reffile)
        if offaxis is not None:
            off = Image(offaxis)
            img.data*=factor # Multiplies by post-processing factor
            img.data+=off.data
        if zodi is not None:
            z = Image(zodi)
            img.data+=z.data
        inttime = par.timeframe/par.Nreads
        par.makeHeader()        

        frame = np.zeros(img.data.shape)
        varframe = np.zeros(img.data.shape)
        # averaging reads
        for i in range(par.Nreads):
            newread = readDetector(par,img,inttime=inttime)
            frame += newread
            varframe += newread**2
        if averageDivide:
            frame /= par.Nreads
            varframe /= par.Nreads
            varframe -= frame**2
        par.hdr.append(('NREADS',par.Nreads,'Number of frames averaged'),end=True)
        par.hdr.append(('EXPTIME',par.timeframe,'Total exposure time for number of frames'),end=True)
        outimg = Image(data=frame,ivar=1./varframe,header=par.hdr)
        # append '_suffix' to the file name
        outimg.write(detectorFolderOut+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits',clobber=True)
        det_outlist.append(detectorFolderOut+'/'+reffile.split('/')[-1].split('.')[0]+'_'+suffix+'.fits')
    return det_outlist


